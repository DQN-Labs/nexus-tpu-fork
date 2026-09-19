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


# Checkpoint module names that differ from the JAX attribute names
# (applied after prefix-strip; first match wins, anchored on ".<src>.").
# The loader's per-param reshape/permute heuristics are keyed on name
# substrings, so JAX names must dodge them (e.g. fused ``qkv`` must not
# contain "v_proj", which would route it into the 3D k/v path).
_JAX_RENAMES = (
    (".self_attn.qkv_proj.", ".self_attn.qkv."),
    (".ple.ple_embedding.", ".ple.embedding."),
    (".ple.kv_proj.", ".ple.kv."),
    (".index_qk_proj.", ".index_qk."),
)


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
    for src, dst in _JAX_RENAMES:
        if src in name:
            name = name.replace(src, dst)
            break
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

# ---------------------------------------------------------------------------
# Load-time GPTQ assembly: checkpoint stream -> JAX-named torch tensors.
#
# vLLM's own GPTQ path (auto_gptq/marlin kernels) is CUDA-only and its
# ModelConfig gate rejects GPTQ on TPU outright, so the fork bypasses it
# entirely: ``--quantization`` is NOT passed, the checkpoint's
# ``quantization_config`` is neutralized in hf_config.py, and this iterator
# dequantizes GPTQ shards on CPU (exact GPTQModel-v2 semantics, see
# quant.dequantize_gptq_torch) and assembles fused JAX params (MoE experts,
# stacked linears) before the standard auto-loader assigns them.
#
# Output convention (must match JaxAutoWeightsLoader per-param handling):
# - 2D linears: torch layout (out, in); the loader transposes to JAX [in, out]
#   (except ``embed_tokens``/``lm_head``-style heuristics, which behave the
#   same for torch-layout input).
# - 3D fused experts (exp_gate_up [H,E,2I], exp_down [E,I,H]): exact JAX
#   layout (the loader leaves ndim != 2 untouched).
# - 1D norms/scales/biases: as stored.
# - PLE n-gram table ``ple.embedding.weight`` [rows, hd]: the loader would
#   transpose any 2D tensor, so it is yielded pre-transposed (a view, free).
# ---------------------------------------------------------------------------

# JAX 1D-norm params whose checkpoint names differ from the JAX attribute
# names: ordered candidate suffixes, first hit wins. (Upstream nvidia layout
# uses RMSNorm submodules; the JAX side keeps functional scale vectors.)
_NORM_CANDIDATES = (
    ("q_norm_w", (".q_norm.weight", ".q_layernorm.weight",
                  ".query_norm.weight", ".q_norm_w")),
    ("k_norm_w", (".k_norm.weight", ".k_layernorm.weight",
                  ".key_norm.weight", ".k_norm_w")),
    ("norm_key_w", (".norm_key.weight", ".key_norm.weight",
                    ".norm_key_w", ".ple_norm_key.weight")),
    ("norm_query_w", (".norm_query.weight", ".query_norm.weight",
                      ".norm_query_w", ".ple_norm_query.weight")),
    ("norm_conv_w", (".norm_conv.weight", ".conv_norm.weight",
                     ".norm_conv_w", ".ple_norm_conv.weight")),
    ("norm_w", (".norm.weight", ".norm_w", ".rms_norm.weight")),
    ("conv_w", (".conv_w", ".conv.weight", ".depthwise_conv.weight")),
    ("conv_weight", (".conv_weight", ".conv.weight",
                     ".causal_conv.weight", ".conv_weight.weight")),
)

# Module-prefix variants: JAX per-role names vs possibly-shared checkpoint
# names. Each entry maps a JAX infix to the checkpoint infixes to try.
_HC_PREFIX_CANDIDATES = (
    (".attn_hyper_connection.", (".attn_hyper_connection.",
                                 ".hyper_connection.")),
    (".mlp_hyper_connection.", (".mlp_hyper_connection.",
                                ".hyper_connection.")),
)

# Linears whose torch (out, in) must be concatenated along rows to form the
# single JAX param (ckpt shard suffix -> position). Mirrors STACKED_MAP but
# operates on dequantized/plain torch tensors at load time.
_CONCAT_GROUPS = {
    # merged PLE kv [E, 5H] from key+value rows (JAX attr ``kv``, see ngram.py)
    "ple.kv": (("ple.key_proj", "ple.value_proj"), 0),
    # merged GDN in_proj_qkvz from qkv+z rows
    "linear_attn.in_proj_qkvz": (("linear_attn.in_proj_qkv",
                                  "linear_attn.in_proj_z"), 0),
    # merged GDN in_proj_ba from b+a rows
    "linear_attn.in_proj_ba": (("linear_attn.in_proj_b",
                                "linear_attn.in_proj_a"), 0),
    # merged HC down_block_inject [(L+HC), ...] from down+inject rows
    "down_block_inject": (("input_mix_weight_down",
                           "block_inject_weight"), 0),
}

# JAX params kept in bf16 (everything else is yielded fp32).
_BF16_JAX_SUFFIXES = (".embed_tokens.weight", ".lm_head.weight")


def expected_jax_names(arch):
    """Full JAX parameter inventory for an arch (mirrors layers.py/model.py).

    Used by the loader-heuristic audit test and as documentation of the
    load contract. Names use the JaxModule dotted convention
    (``...weight`` for linears/embeds/norms, bare attr for nnx.Param
    vectors, no ``.weight`` on fused 3D expert params).
    """
    from .config import is_moe_layer

    names = ["model.embed_tokens.weight", "lm_head.weight",
             "model.hyper_connection_mixer.hc_norm.weight",
             "model.hyper_connection_mixer.down_block_inject.weight",
             "model.hyper_connection_mixer.up.weight"]
    n = int(arch.num_hidden_layers)
    ple_ids = set(arch.ple_layer_ids or [])
    for i in range(n):
        p = f"model.layers.{i}"
        lt = list(arch.layer_types)[i] if arch.layer_types else None
        if (i + 1) in ple_ids:
            names += [f"{p}.ple.embedding.weight", f"{p}.ple.kv.weight",
                      f"{p}.ple.norm_key_w", f"{p}.ple.norm_query_w",
                      f"{p}.ple.norm_conv_w", f"{p}.ple.conv_w"]
        if lt == "linear_attention":
            names += [f"{p}.linear_attn.in_proj_qkvz.weight",
                      f"{p}.linear_attn.in_proj_ba.weight",
                      f"{p}.linear_attn.conv_weight",
                      f"{p}.linear_attn.A_log", f"{p}.linear_attn.dt_bias",
                      f"{p}.linear_attn.norm_w",
                      f"{p}.linear_attn.out_proj.weight"]
        else:
            names += [f"{p}.self_attn.qkv.weight",
                      f"{p}.self_attn.o_proj.weight",
                      f"{p}.self_attn.q_norm_w", f"{p}.self_attn.k_norm_w"]
            if getattr(arch, "use_qsa", False):
                names += [f"{p}.self_attn.indexer.index_qk.weight",
                          f"{p}.self_attn.indexer.q_norm_w",
                          f"{p}.self_attn.indexer.k_norm_w"]
        if is_moe_layer(arch, i):
            names += [f"{p}.mlp.gate.weight",
                      f"{p}.mlp.exp_gate_up", f"{p}.mlp.exp_down"]
            if int(getattr(arch, "shared_expert_intermediate_size", 0) or 0) > 0:
                names += [f"{p}.mlp.shared_expert.gate_proj.weight",
                          f"{p}.mlp.shared_expert.up_proj.weight",
                          f"{p}.mlp.shared_expert.down_proj.weight"]
        else:
            names += [f"{p}.mlp.gate_proj.weight",
                      f"{p}.mlp.up_proj.weight",
                      f"{p}.mlp.down_proj.weight"]
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            names += [f"{p}.{hc}.hc_norm.weight",
                      f"{p}.{hc}.down_block_inject.weight",
                      f"{p}.{hc}.up.weight"]
    return names


# Already-merged checkpoint linears whose JAX attribute has a shorter name.
_MERGED_RENAMES = (
    (".input_mix_weight_down_block_inject", ".down_block_inject"),
    (".input_mix_weight_up", ".up"),
)


def _jax_name_for(mapped, jax_set):
    """Resolve a mapped checkpoint tensor/linear name to its JAX param name.

    Exact membership wins; renames (norm suffixes, HC role infixes, merged
    HC names) are only accepted when the result is an actual JAX param.
    Returns None when nothing matches (caller reports it loud).
    """
    if mapped in jax_set:
        return mapped
    for jax_key, ckpt_suffixes in _NORM_CANDIDATES:
        for suf in ckpt_suffixes:
            if mapped.endswith(suf):
                cand = mapped[: -len(suf)] + "." + jax_key
                if cand in jax_set:
                    return cand
    for jax_infix, ckpt_infixes in _HC_PREFIX_CANDIDATES:
        for infix in ckpt_infixes:
            if infix in mapped:
                for tail in ("", ".weight"):
                    cand = mapped.replace(infix, jax_infix) + tail
                    if cand in jax_set:
                        return cand
    for src, dst in _MERGED_RENAMES:
        if src in mapped:
            for tail in ("", ".weight"):
                cand = mapped.replace(src, dst) + tail
                if cand in jax_set:
                    return cand
    return None


def _maybe_bf16(jax_name, tensor):
    """Match the JAX param dtype: bf16 for embed/lm_head, fp32 otherwise.

    The auto-loader assigns without casting, so the yielded dtype must be
    exact (this also upcasts a bf16 PLE table to the fp32 JAX param).
    """
    import torch

    if jax_name.endswith(_BF16_JAX_SUFFIXES):
        return tensor.to(torch.bfloat16)
    if tensor.dtype != torch.float32:
        return tensor.to(torch.float32)
    return tensor


def iter_jax_named_weights(weights, arch, jax_names, report=None, *,
                           bits=4, group_size=128, log=None):
    """Yield ``(jax_name, torch_tensor)`` ready for JaxAutoWeightsLoader.

    Args:
        weights: iterable of ``(ckpt_name, torch_tensor)`` (raw names).
        arch: Qwen4ExpArch (layer counts, experts, dims for assembly).
        jax_names: iterable of expected JAX param names (from the live
            model) — used to verify full coverage at the end.
        report: dict filled with ``filled``, ``missing``, ``unconsumed``,
            ``dropped``, ``dequantized`` for the load report artifact.
        bits/group_size: pinned GPTQ contract (INT4 / 128).

    The generator buffers only incomplete linear groups (a few tensors) and
    per-layer expert accumulators; everything else streams through.
    """
    from .quant import dequantize_gptq_torch

    def _log(msg):
        if log is not None:
            log(msg)

    rep = {"filled": [], "missing": [], "unconsumed": [], "dropped": {},
           "dequantized": 0, "assembled": 0}
    filled = set()
    # linear-prefix -> {"qweight": t, "qzeros": t, "scales": t, "g_idx": t,
    #                   "weight": t, "bias": t}
    groups = {}
    # (layer_idx,) -> {"gate": {e: t}, "up": {e: t}, "down": {e: t}} torch (O,I)
    experts = {}
    num_experts = int(getattr(arch, "num_experts", 0) or 0)
    # raw buffered single tensors needing JAX-side rename (norms etc.)
    singles = {}

    def _linear_piece(mapped_prefix, parts):
        """Assemble one linear's torch-layout (out, in) tensor from shards.

        Returns (tensor, used_identity_g_idx). Plain .weight wins over GPTQ
        shards when both exist (warns).
        """
        if "weight" in parts and "qweight" in parts:
            _log(f"LOAD-WARN {mapped_prefix}: has both .weight and GPTQ "
                 f"shards; preferring .weight")
        if "weight" in parts:
            return parts["weight"], False
        need = ("qweight", "qzeros", "scales")
        if not all(k in parts for k in need):
            return None, False
        g_idx = parts.get("g_idx")
        used_identity = g_idx is None
        if used_identity:
            g_idx = _identity_g_idx(parts["qweight"], parts["scales"])
        w = dequantize_gptq_torch(
            parts["qweight"], parts["qzeros"], parts["scales"], g_idx,
            bits=bits, group_size=group_size)
        rep["dequantized"] += 1
        return w.T.contiguous(), used_identity

    def _identity_g_idx(qweight, scales):
        import torch

        in_features = qweight.shape[0] * (32 // bits)
        if group_size == -1:
            return torch.zeros(in_features, dtype=torch.int32)
        return torch.arange(in_features, dtype=torch.int32) // group_size

    arch_moe_inter = getattr(arch, "moe_intermediate_size", None)
    arch_hidden = int(getattr(arch, "hidden_size", 0) or 0)

    def _is_expert_base(base):
        m = _re.match(
            r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$",
            base)
        if not m:
            return None
        return m.group(1), int(m.group(2)), m.group(3)

    def _is_concat_shard(base):
        for _jax_suffix, (shards, _axis) in _CONCAT_GROUPS.items():
            for shard in shards:
                if base.endswith("." + shard):
                    return True
        return False

    def _concat_target_for(base):
        for jax_suffix, (shards, _axis) in _CONCAT_GROUPS.items():
            for shard in shards:
                if base.endswith("." + shard):
                    layer_base = base[: -len(shard) - 1]
                    return layer_base + "." + jax_suffix
        return None

    def _store_expert_piece(layer_base, expert_idx, proj, torch_w):
        """Accumulate one expert's torch-layout (out, in) linear.

        Returns [(jax_name, tensor), ...] when the whole layer is complete
        (fused exp_gate_up [H,E,2I] + exp_down [E,I,H]), else None.
        """
        import torch

        if expert_idx < 0 or expert_idx >= num_experts:
            raise ValueError(
                f"LOAD-FAIL {layer_base}: expert index {expert_idx} "
                f"outside [0, {num_experts})")
        acc = experts.setdefault(layer_base, {"down": {}})
        if proj in ("gate_proj", "up_proj"):
            i, h = int(torch_w.shape[0]), int(torch_w.shape[1])
            if arch_hidden and h != arch_hidden:
                raise ValueError(
                    f"LOAD-FAIL {layer_base}: expert {proj} cols {h} != "
                    f"arch hidden {arch_hidden}")
            if arch_moe_inter and i != int(arch_moe_inter):
                raise ValueError(
                    f"LOAD-FAIL {layer_base}: expert {proj} rows {i} != "
                    f"arch moe_intermediate {arch_moe_inter}")
            if "i" not in acc:
                acc["i"], acc["h"] = i, h
            elif (acc["i"], acc["h"]) != (i, h):
                raise ValueError(
                    f"LOAD-FAIL {layer_base}: expert {proj} shape "
                    f"{(i, h)} != first piece {(acc['i'], acc['h'])}")
            slot = acc.setdefault(proj, {})
            slot[expert_idx] = torch_w.T.contiguous().to(torch.float32)
        else:
            slot = acc.setdefault("down", {})
            slot[expert_idx] = torch_w.T.contiguous().to(torch.float32)
        if len(acc.get("gate_proj", {})) == num_experts and \
                len(acc.get("up_proj", {})) == num_experts and \
                len(acc.get("down", {})) == num_experts and \
                not acc.get("done"):
            acc["done"] = True
            i, h, e = acc["i"], acc["h"], num_experts
            gu = torch.empty(h, e, 2 * i, dtype=torch.float32)
            for ei in range(e):
                gu[:, ei, :i] = acc["gate_proj"][ei]
                gu[:, ei, i:] = acc["up_proj"][ei]
                del acc["gate_proj"][ei], acc["up_proj"][ei]
            dn = torch.empty(e, i, h, dtype=torch.float32)
            for ei in range(e):
                dn[ei] = acc["down"][ei]
                del acc["down"][ei]
            del experts[layer_base]
            rep["assembled"] += 1
            return [(layer_base + ".exp_gate_up", gu),
                    (layer_base + ".exp_down", dn)]
        return None

    def _emit(jax_name, tensor):
        tensor = _maybe_pretranspose(jax_name, tensor)
        filled.add(jax_name)
        return jax_name, _maybe_bf16(jax_name, tensor)

    def _complete_dense(base, parts):
        """Flush one dense linear group.

        Returns (items, flushed): items is a list of (jax_name, tensor) to
        yield; flushed False means incomplete (keep buffering).
        """
        out = []
        if "weight" in parts and "qweight" in parts:
            _log(f"LOAD-WARN {base}: has both .weight and GPTQ shards; "
                 f"preferring .weight")
        if "weight" in parts:
            piece, used_identity = parts["weight"], False
        elif all(k in parts for k in ("qweight", "qzeros", "scales")):
            piece, used_identity = _linear_piece(base, parts)
        else:
            return out, False  # incomplete: keep buffering
        exp = _is_expert_base(base)
        if exp is not None:
            layer_base, expert_idx, proj = exp
            done = _store_expert_piece(layer_base, expert_idx, proj, piece)
            if used_identity:
                assumed_identity.add(base)
            if done:
                for jax_name, tensor in done:
                    filled.add(jax_name)
                    out.append((jax_name, tensor))
            return out, True
        if _is_concat_shard(base):
            target = _concat_target_for(base)
            acc = concat_accum.setdefault(target, {})
            acc[base] = piece
            if used_identity:
                assumed_identity.add(base)
            if _concat_ready(target, acc):
                merged = _concat_shards(target, acc)
                del concat_accum[target]
                jax_name = _jax_name_for(target + ".weight", jax_set)
                if jax_name is None:
                    raise ValueError(
                        f"LOAD-FAIL concat target {target}.weight matches no "
                        f"JAX param")
                rep["assembled"] += 1
                out.append(_emit(jax_name, merged))
            return out, True
        if base.endswith(".mlp.shared_expert.gate_up_proj"):
            lyr = base[: -len(".mlp.shared_expert.gate_up_proj")]
            half = piece.shape[0] // 2
            for suffix, part in ((".mlp.shared_expert.gate_proj.weight",
                                  piece[:half]),
                                 (".mlp.shared_expert.up_proj.weight",
                                  piece[half:])):
                jax_name = _jax_name_for(lyr + suffix, jax_set)
                if jax_name is None:
                    raise ValueError(
                        f"LOAD-FAIL shared split {lyr + suffix} matches no "
                        f"JAX param")
                out.append(_emit(jax_name, part))
            if used_identity:
                assumed_identity.add(base)
            return out, True
        jax_name = _jax_name_for(base + ".weight", jax_set)
        if jax_name is None:
            return out, False  # hold for end-of-stream report
        if used_identity:
            assumed_identity.add(base)
        out.append(_emit(jax_name, piece))
        return out, True

    # -- streaming pass --------------------------------------------------
    import re as _re

    jax_set = set(jax_names)
    assumed_identity = set()
    concat_accum = {}
    rep["warnings"] = []
    import torch as _torch

    for raw_name, tensor in weights:
        if not isinstance(tensor, _torch.Tensor):
            tensor = _torch.as_tensor(tensor)
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        mapped = map_checkpoint_name(raw_name, frozenset())
        if is_ignored_missing(mapped):
            rep["dropped"][mapped] = rep["dropped"].get(mapped, 0) + 1
            continue
        if mapped.endswith(".bias"):
            rep["dropped"][mapped] = rep["dropped"].get(mapped, 0) + 1
            continue
        if is_gptq_aux(mapped):
            base, _, kind = mapped.rpartition(".")
            if base in assumed_identity and kind == "g_idx":
                if not _is_identity_g_idx(tensor, group_size):
                    raise ValueError(
                        f"LOAD-FAIL {base}: g_idx arrived after identity "
                        f"assumption and is NOT identity (desc_act hybrid?)")
                rep["dropped"][mapped] = rep["dropped"].get(mapped, 0) + 1
                continue
            g = groups.setdefault(base, {})
            g[kind] = tensor
            items, flushed = _complete_dense(base, g)
            for item in items:
                yield item
            if flushed:
                del groups[base]
            continue
        if mapped.endswith(".weight"):
            base = mapped[: -len(".weight")]
            if _is_expert_base(base) is not None or _is_concat_shard(base) \
                    or base.endswith(".mlp.shared_expert.gate_up_proj"):
                g = groups.setdefault(base, {})
                g["weight"] = tensor
                items, flushed = _complete_dense(base, g)
                for item in items:
                    yield item
                if flushed:
                    del groups[base]
                # else: concat shard waiting for siblings: keep buffering.
                continue
            jax_name = _jax_name_for(mapped, jax_set)
            if jax_name is not None:
                yield _emit(jax_name, tensor)
            else:
                singles[mapped] = tensor
            continue
        # Single 1D / table / other tensor: buffer for rename resolution.
        singles[mapped] = tensor

    # -- end of stream: flush leftovers -----------------------------------
    for base in sorted(groups):
        parts = groups[base]
        if "weight" not in parts and all(
                k in parts for k in ("qweight", "qzeros", "scales")):
            rep["warnings"].append(
                f"{base}: flushed without g_idx (identity assumed)")
            items, flushed = _complete_dense(base, parts)
            for item in items:
                yield item
            if flushed:
                continue
        rep["unconsumed"].append(
            f"GROUP:{base}=" + ",".join(sorted(parts)))
    for target in sorted(concat_accum):
        rep["unconsumed"].append(
            f"CONCAT:{target}=" + ",".join(sorted(concat_accum[target])))
    for layer_base in sorted(experts):
        acc = experts[layer_base]
        have = {k: len(v) for k, v in acc.items()
                if isinstance(v, dict)}
        rep["unconsumed"].append(
            f"EXPERTS:{layer_base}={have}")
    jax_set = set(jax_names)
    for ckpt_name in sorted(singles):
        jax_name = _jax_name_for(ckpt_name, jax_set)
        if jax_name is not None and jax_name not in filled:
            yield _emit(jax_name, singles[ckpt_name])
            del singles[ckpt_name]
    for ckpt_name in sorted(singles):
        rep["unconsumed"].append(ckpt_name)
    rep["filled"] = sorted(filled)
    rep["missing"] = sorted(set(jax_names) - filled)
    if report is not None:
        report.update(rep)


def _concat_shards_for(target):
    for jax_suffix, (shards, _axis) in _CONCAT_GROUPS.items():
        if target.endswith("." + jax_suffix):
            return shards
    return None


def _concat_ready(target, parts):
    shards = _concat_shards_for(target)
    return shards is not None and all(
        any(b.endswith("." + s) for b in parts) for s in shards)


def _concat_shards(target, parts):
    """Concatenate split shards in _CONCAT_GROUPS order (NOT alphabetical:
    e.g. in_proj_b precedes in_proj_a; down precedes inject)."""
    import torch

    shards = _concat_shards_for(target)
    ordered = []
    for s in shards:
        hit = [b for b in parts if b.endswith("." + s)]
        if len(hit) != 1:
            raise ValueError(
                f"LOAD-FAIL concat {target}: shard {s} has {len(hit)} "
                f"matches in {sorted(parts)}")
        ordered.append(parts[hit[0]])
    return torch.cat(ordered, dim=0).contiguous()


def _is_identity_g_idx(g_idx, group_size):
    import torch

    g = g_idx.reshape(-1)
    n = int(g.numel())
    if group_size == -1:
        return bool((g == 0).all())
    expect = torch.arange(n, dtype=g.dtype) // group_size
    return bool((g == expect).all())


def _maybe_pretranspose(jax_name, tensor):
    # The auto-loader transposes every 2D tensor. Params whose JAX layout
    # already equals the stored layout (PLE n-gram table) are yielded
    # pre-transposed (a view, no copy).
    if jax_name.endswith(".ple.embedding.weight") and tensor.ndim == 2:
        return tensor.T
    return tensor


__all__ = [
    "PREFIX_MAP",
    "STACKED_MAP",
    "TRANSPOSE_SUBSTR",
    "_CONCAT_GROUPS",
    "expected_jax_names",
    "is_ignored_missing",
    "iter_jax_named_weights",
    "map_checkpoint_name",
    "remap_qsa_scale_name",
    "stacked_target",
    "strip_mtp_prefix",
]
