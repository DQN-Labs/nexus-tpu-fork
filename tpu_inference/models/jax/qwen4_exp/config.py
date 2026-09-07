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

"""Qwen4Exp (Qwen3.8-Flash-Next) configuration compatibility.

Canonical references (upstream vLLM):
- ``vllm/models/qwen4_exp/config.py``: Qwen4ExpTextConfig / Qwen4ExpConfig
- ``vllm/transformers_utils/configs/qwen3_next.py``: Qwen3NextConfig base
- ``vllm/transformers_utils/configs/qwen4_exp.py``: re-export shim

This module does NOT re-implement HF config parsing from scratch. It adapts
whatever HF ``config.json`` vLLM already parsed (``hf_config`` /
``hf_text_config``) into a normalized view used by the JAX model, with the
same validation semantics as upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_QSA_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_text_config(hf_config: Any) -> Any:
    """Return the text sub-config (handles nested ``text_config``)."""
    text = _get(hf_config, "text_config", None)
    return text if text is not None else hf_config


@dataclass
class Qwen4ExpArch:
    """Normalized architecture parameters consumed by JAX modules."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.25
    attention_bias: bool = False
    # Qwen3Next linear-attention (GDN) dims
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    # MoE
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    mlp_only_layers: list = field(default_factory=list)
    layer_types: list = field(default_factory=list)
    # Hyper-connection
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE / n-gram
    ple_layer_ids: list = field(default_factory=list)
    ple_embed_dim: int = 0
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    output_gate_type: str = "sigmoid"
    # QSA (None when dense full attention)
    indexer_n_heads: Any = None
    indexer_kv_heads: Any = None
    indexer_head_dim: Any = None
    indexer_budget: Any = None
    indexer_compress_ratio: Any = None

    @property
    def use_qsa(self) -> bool:
        return self.indexer_n_heads is not None

    @property
    def ngram_context_len(self) -> int:
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)

    @property
    def short_conv_layer_ids(self) -> list:
        if not self.ple_layer_ids:
            return []
        return sorted({int(i) - 1 for i in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple | None:
        if not self.short_conv_layer_ids:
            return None
        ple_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        ple_channels = self.hidden_size * self.hc_count
        return (ple_channels, ple_state_len)


def validate_qwen4exp_text_config(text: Any) -> None:
    """Mirror ``Qwen4ExpTextConfig`` validators (raises ValueError)."""
    hc_count = int(_get(text, "hc_count", 4))
    if hc_count <= 1:
        raise ValueError(f"Qwen4Exp requires hc_count > 1, got {hc_count}.")
    hc_lowrank = int(_get(text, "hc_lowrank", 320))
    if hc_lowrank <= 0:
        raise ValueError(f"hc_lowrank must be positive, got {hc_lowrank}")
    ngram_size = int(_get(text, "ngram_size", 3))
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
    heads_per_ngram = int(_get(text, "heads_per_ngram", 8))
    if heads_per_ngram <= 0:
        raise ValueError("heads_per_ngram must be positive")
    ple_embed_dim = _get(text, "ple_embed_dim", None)
    hidden_size = int(_get(text, "hidden_size", 0))
    if ple_embed_dim is None:
        ple_embed_dim = hidden_size
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if int(ple_embed_dim) % ngram_heads:
        raise ValueError(
            f"ple_embed_dim must be divisible by total ngram heads: "
            f"{ple_embed_dim} % {ngram_heads} != 0"
        )
    ple_layer_ids = _get(text, "ple_layer_ids", []) or []
    num_layers = int(_get(text, "num_hidden_layers", 0))
    invalid = [i for i in ple_layer_ids if not 1 <= int(i) <= num_layers]
    if invalid:
        raise ValueError(
            f"ple_layer_ids are 1-based and must refer to an existing layer; "
            f"got {invalid} for {num_layers} layers"
        )
    configured = {k: _get(text, k, None) for k in _QSA_FIELDS}
    if all(v is None for v in configured.values()):
        return
    missing = [k for k, v in configured.items() if v is None]
    if missing:
        raise ValueError(f"QSA config is missing required fields: {missing}")
    vals = {k: int(v) for k, v in configured.items()}
    if any(v <= 0 for v in vals.values()):
        raise ValueError(f"QSA config values must be positive: {vals}")
    if vals["indexer_kv_heads"] != 1:
        raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
    if vals["indexer_budget"] % vals["indexer_compress_ratio"] != 0:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    block_topk = vals["indexer_budget"] // vals["indexer_compress_ratio"]
    if block_topk not in (512, 2048):
        raise ValueError(
            "QSA requires indexer_budget / indexer_compress_ratio to be "
            f"512 or 2048, got {block_topk}"
        )
    head_dim = int(_get(text, "head_dim", 256))
    prf = float(_get(text, "partial_rotary_factor", 0.25))
    rotary_dim = int(head_dim * prf)
    if rotary_dim > vals["indexer_head_dim"]:
        raise ValueError(
            f"QSA indexer_head_dim must cover the attention rotary dimension, "
            f"got {vals['indexer_head_dim']} < {rotary_dim}"
        )


def arch_from_hf_config(hf_config: Any, vocab_size: int | None = None) -> Qwen4ExpArch:
    """Build normalized arch from a parsed HF config (flat or nested)."""
    text = get_text_config(hf_config)
    validate_qwen4exp_text_config(text)

    def g(name: str, default: Any) -> Any:
        v = _get(text, name, _get(hf_config, name, default))
        return default if v is None else v

    hidden_size = int(g("hidden_size", 2048))
    ple_embed_dim = g("ple_embed_dim", None)
    if ple_embed_dim is None:
        ple_embed_dim = hidden_size
    layer_types = g("layer_types", None)
    num_layers = int(g("num_hidden_layers", 48))
    if layer_types is None:
        layer_types = [
            "linear_attention" if (i + 1) % 4 else "full_attention"
            for i in range(num_layers)
        ]
    # Rope theta may live under rope_parameters / rope_theta / text sub-dict.
    rope_theta = float(g("rope_theta", 10000.0))
    rope_params = g("rope_parameters", None) or g("rope_scaling", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        try:
            rope_theta = float(rope_params["rope_theta"])
        except (TypeError, ValueError):
            pass
    if vocab_size is None:
        vocab_size = int(g("vocab_size", 151936))

    return Qwen4ExpArch(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=int(g("num_attention_heads", 16)),
        num_key_value_heads=int(g("num_key_value_heads", 2)),
        head_dim=int(g("head_dim", 256)),
        intermediate_size=int(g("intermediate_size", 5632)),
        vocab_size=int(vocab_size),
        rms_norm_eps=float(g("rms_norm_eps", 1e-6)),
        hidden_act=str(g("hidden_act", "silu")),
        max_position_embeddings=int(g("max_position_embeddings", 32768)),
        rope_theta=rope_theta,
        partial_rotary_factor=float(g("partial_rotary_factor", 0.25)),
        attention_bias=bool(g("attention_bias", False)),
        linear_conv_kernel_dim=int(g("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(g("linear_key_head_dim", 128)),
        linear_value_head_dim=int(g("linear_value_head_dim", 128)),
        linear_num_key_heads=int(g("linear_num_key_heads", 16)),
        linear_num_value_heads=int(g("linear_num_value_heads", 32)),
        decoder_sparse_step=int(g("decoder_sparse_step", 1)),
        moe_intermediate_size=int(g("moe_intermediate_size", 512)),
        shared_expert_intermediate_size=int(g("shared_expert_intermediate_size", 512)),
        num_experts_per_tok=int(g("num_experts_per_tok", 10)),
        num_experts=int(g("num_experts", 512)),
        norm_topk_prob=bool(g("norm_topk_prob", True)),
        mlp_only_layers=list(g("mlp_only_layers", []) or []),
        layer_types=list(layer_types),
        hc_count=int(g("hc_count", 4)),
        hc_lowrank=int(g("hc_lowrank", 320)),
        ple_layer_ids=list(g("ple_layer_ids", []) or []),
        ple_embed_dim=int(ple_embed_dim),
        ple_conv_kernel_size=int(g("ple_conv_kernel_size", 4)),
        ngram_size=int(g("ngram_size", 3)),
        heads_per_ngram=int(g("heads_per_ngram", 8)),
        ngram_vocab_size_base=int(g("ngram_vocab_size_base", 20_000_000)),
        make_ngram_vocab_size_divisible_by=int(
            g("make_ngram_vocab_size_divisible_by", 128)
        ),
        output_gate_type=str(g("output_gate_type", "sigmoid")),
        indexer_n_heads=g("indexer_n_heads", None),
        indexer_kv_heads=g("indexer_kv_heads", None),
        indexer_head_dim=g("indexer_head_dim", None),
        indexer_budget=g("indexer_budget", None),
        indexer_compress_ratio=g("indexer_compress_ratio", None),
    )


def is_moe_layer(arch: Qwen4ExpArch, layer_idx: int) -> bool:
    """Mirror ``Qwen4ExpDecoderLayer`` MoE selection logic."""
    if layer_idx in set(arch.mlp_only_layers or []):
        return False
    if (arch.num_experts or 0) <= 0:
        return False
    return (layer_idx + 1) % arch.decoder_sparse_step == 0


__all__ = [
    "Qwen4ExpArch",
    "arch_from_hf_config",
    "get_text_config",
    "is_moe_layer",
    "validate_qwen4exp_text_config",
]
