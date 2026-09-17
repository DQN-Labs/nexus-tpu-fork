# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-name mapping: HF checkpoint → JAX Qwen4Exp model.

Upstream references:
- ``vllm/models/qwen4_exp/nvidia/model.py``: ``_EXTRA_WEIGHTS_MAPPER``
  (hyper-connection down/inject merge, PLE kv merge),
  ``packed_modules_mapping`` (qkv_proj / gate_up_proj / kv_proj /
  in_proj_qkvz / in_proj_ba / down_block_inject), ``_remap_qsa_cache_scale_name``,
  ``_QWEN4_EXP_IGNORED_MISSING_SUFFIXES``, PLE shard validation
  (``split_ngram_parts`` + ``copy_ple_embedding_shard_``).
- ``vllm/model_executor/models/qwen3_5.py``: ``Qwen3_5Model.hf_to_vllm_mapper``
  base (``model.language_model.`` prefix strip, GDN in_proj stacking, dense
  q/k/v and gate/up stacking, shared-expert mapping).
- ``vllm/models/qwen4_exp/nvidia/mtp.py``: MTP remap
  (``model.mtp.`` → ``mtp.``, ``shared_head.head.`` → ``lm_head.``).

JAX parameter naming follows ``tpu_inference`` conventions (``JaxEinsum``
aliases ``kernel`` → ``weight``; ``JaxRmsNorm`` aliases ``scale`` →
``weight``; ``JaxEmbed`` aliases ``embedding`` → ``weight``), so mapped
names end in ``.weight`` and match ``named_parameters()``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# HF -> JAX prefix map (ordered; first match wins for exact prefixes).
PREFIX_MAP: List[Tuple[str, str]] = [
    ("model.language_model.", "model."),
    ("language_model.", "model."),
    ("model.mtp.", "mtp."),
    ("model.lm_head.", "lm_head."),
    ("model.embed_tokens.", "model.embed_tokens."),
]

# Stacked (fused) projections: ckpt shard -> (JAX param, shard_index).
# Mirrors upstream packed_modules_mapping + _EXTRA_WEIGHTS_MAPPER.
STACKED_MAP: Dict[str, Tuple[str, int]] = {
    # Hyper-connection merged down projection.
    "hyper_connection.input_mix_weight_down.weight": (
        "hyper_connection.input_mix_weight_down_block_inject.weight", 0),
    "hyper_connection.block_inject_weight.weight": (
        "hyper_connection.input_mix_weight_down_block_inject.weight", 1),
    # PLE merged KV.
    "ple.key_proj.weight": ("ple.kv_proj.weight", 0),
    "ple.value_proj.weight": ("ple.kv_proj.weight", 1),
    # Dense attention fused qkv / gate-up (split at load; see split_rows()).
    # GDN fused projections.
    "linear_attn.in_proj_qkv.weight": ("linear_attn.in_proj_qkvz.weight", 0),
    "linear_attn.in_proj_z.weight": ("linear_attn.in_proj_qkvz.weight", 1),
    "linear_attn.in_proj_b.weight": ("linear_attn.in_proj_ba.weight", 0),
    "linear_attn.in_proj_a.weight": ("linear_attn.in_proj_ba.weight", 1),
    # MoE fused gate_up.
    "mlp.experts.gate_proj.weight": ("mlp.experts.gate_up_proj.weight", 0),
    "mlp.experts.up_proj.weight": ("mlp.experts.gate_up_proj.weight", 1),
    "mlp.shared_expert.gate_proj.weight": ("mlp.shared_expert.gate_up_proj.weight", 0),
    "mlp.shared_expert.up_proj.weight": ("mlp.shared_expert.gate_up_proj.weight", 1),
}

# QSA cache-scale remap suffixes (upstream _remap_qsa_cache_scale_name).
_QSA_SCALE_SUFFIXES = {
    "k_proj.k_scale": "_k_scale",
    "k_proj.output_scale": "_k_scale",
    "attn.k_scale": "_k_scale",
    "attn._k_scale": "_k_scale",
    "k_scale": "_k_scale",
    "_k_scale": "_k_scale",
    "v_proj.v_scale": "_v_scale",
    "v_proj.output_scale": "_v_scale",
    "attn.v_scale": "_v_scale",
    "attn._v_scale": "_v_scale",
    "v_scale": "_v_scale",
    "_v_scale": "_v_scale",
}


def strip_mtp_prefix(name: str) -> str:
    """Apply MTP remaps (``mtp.py``): shared head → lm_head, etc."""
    # Order matters: more specific first.
    name = name.replace("model.language_model.", "model.")
    name = re.sub(r"^language_model\.", "model.", name)
    name = re.sub(r"^model\.mtp\.", "mtp.", name)
    name = re.sub(r"^mtp\.", "mtp.", name)
    name = re.sub(r"\.shared_head\.head\.", ".", name)
    name = re.sub(r"^model\.lm_head\.", "lm_head.", name)
    return name


def remap_qsa_scale_name(name: str, qsa_layer_ids: frozenset) -> str:
    for lid in qsa_layer_ids:
        marker = f"layers.{lid}.self_attn."
        i = name.find(marker)
        if i < 0 or (i > 0 and name[i - 1] != "."):
            continue
        suffix = name[i + len(marker):]
        if suffix in _QSA_SCALE_SUFFIXES:
            return f"{name[: i + len(marker)]}{_QSA_SCALE_SUFFIXES[suffix]}"
    return name


def map_checkpoint_name(
    name: str,
    qsa_layer_ids: frozenset = frozenset(),
) -> str:
    """Map one HF/vLLM checkpoint tensor name to the JAX model namespace."""
    orig = name
    name = strip_mtp_prefix(name)
    for src, dst in PREFIX_MAP:
        if name.startswith(src):
            name = dst + name[len(src):]
            break
    name = remap_qsa_scale_name(name, qsa_layer_ids)
    # Normalize torch Linear ".weight" (JAX kernels alias weight).
    # nnx.Einsum kernel shape [in, out] matches torch [out, in]^T; the
    # transpose is applied in StandardWeightLoader transpose_map, not here.
    _ = orig
    return name


def stacked_target(name: str) -> Tuple[str, int] | None:
    """Return (merged_param_suffix, shard) if ``name`` is a stacked shard."""
    for suffix, (merged, shard) in STACKED_MAP.items():
        if name.endswith(suffix):
            return (name[: -len(suffix)] + merged, shard)
    return None


def is_ignored_missing(name: str) -> bool:
    from .quant import IGNORED_MISSING_SUFFIXES

    if "hyper_connection_mixer.block_inject_weight" in name:
        return True
    # MTP draft-model tensors (mtp.*): our MTP is a stub sharing the target
    # trunk (model.py::Qwen4ExpMTP), so draft-only weights such as
    # mtp.layers.N.mlp.experts.{down_proj,gate_up_proj} are never loaded.
    # Diagnosed 2026-09-16 (v44 inspect: 2 of the 5 GAPS).
    if name.startswith("mtp.") or ".mtp." in name:
        return True
    return any(name.endswith(s) for s in IGNORED_MISSING_SUFFIXES)


# GPTQ auxiliary tensors live alongside each Linear weight (qweight/qzeros/scales/g_idx
# per expert shard, plus per-channel scales for fused gate_up). They are not
# separate JAX parameters — the loader dequantizes them into the single
# float weight (see quant.py::dequantize_q4_packed) — so for coverage they
# count as mapped, not unknown. Likewise GDN's A_log / dt_bias have no
# ".weight" suffix but are real params (gdn.py).
_GPTQ_SUFFIXES = (
    ".qweight",
    ".qzeros",
    ".scales",
    ".g_idx",
    ".weight_scale",  # compressed-tensors alias
    ".input_scale",
)


def is_gptq_aux(name: str) -> bool:
    return any(name.endswith(s) for s in _GPTQ_SUFFIXES)


_GDN_NOWEIGHT_PARAMS = (
    ".A_log",
    ".dt_bias",
    ".A_log.weight",
    ".dt_bias.weight",
)


def is_gdn_param(name: str) -> bool:
    return any(name.endswith(s) for s in _GDN_NOWEIGHT_PARAMS)


# JAX-side transpose/reshape hints consumed by StandardWeightLoader.
# torch nn.Linear weight [out, in] -> nnx.Einsum kernel [in, out].
TRANSPOSE_SUBSTR = (
    "qkv_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    "kv_proj",
    "key_proj",
    "value_proj",
    "in_proj_qkvz",
    "in_proj_ba",
    "input_mix_weight_down_block_inject",
    "input_mix_weight_up",
    "index_qk_proj",
    "lm_head",
)

__all__ = [
    "PREFIX_MAP",
    "STACKED_MAP",
    "TRANSPOSE_SUBSTR",
    "is_ignored_missing",
    "map_checkpoint_name",
    "remap_qsa_scale_name",
    "stacked_target",
    "strip_mtp_prefix",
]
