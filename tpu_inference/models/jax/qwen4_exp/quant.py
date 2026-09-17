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

"""Q4 quantization support for Qwen4Exp on TPU.

Target (Phase 1): **Q4 weights stored in a straightforward packed
representation + JAX-side dequantization**, per the task spec §7. Do not
attempt NVIDIA NVFP4/FP8 reproduction initially.

Supported checkpoint layouts (auto-detected per tensor):
1. Plain BF16/FP32 safetensors (reference / FP8 checkpoints dequantized by
   the vLLM loader before reaching JAX).
2. Packed int4 ``weight`` (uint8, ``[out, in//2]``) + ``weight_scale``
   (fp32/bf16, per-output-channel or per-group) + optional ``weight_zero``
   — the common GPTQ/AWQ-style Q4 export. Dequant:
       w_f = (q & 0xF / q >> 4 - zero) * scale   (order configurable)
3. ``compressed-tensors`` W4A16 (``weight_packed`` + ``weight_scale_and_zero``).
   Detected and routed to the same dequant path.

All dequant math runs in float32 then casts to the model dtype (bf16), so
generation matches the reference within BF16/Q4 tolerance (same greedy
token objective, task §5).

For MXFP4/ModelOpt-FP4 checkpoints: QSA ``qkv_proj`` is explicitly excluded
from ModelOpt-FP4 upstream (``without_modelopt_fp4``); HC linears never use
quantization. This module enforces the same skips via ``QUANT_SKIP_SUBSTR``.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

QUANT_SKIP_SUBSTR = (
    "attn_hyper_connection",
    "mlp_hyper_connection",
    "hyper_connection_mixer",
    "self_attn._k_scale",
    "self_attn._v_scale",
)

IGNORED_MISSING_SUFFIXES = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
    # PLE n-gram table metadata: derived at load (splitmix64 multipliers,
    # per-head sizes/offsets computed in ngram.py), not loaded params.
    # Diagnosed 2026-09-16 (v44 inspect: 3 of the 5 GAPS).
    ".layer_multipliers",
    ".ngram_heads_offsets",
    ".ngram_heads_vocab_sizes",
)


def should_skip_quant(prefix: str) -> bool:
    return any(s in prefix for s in QUANT_SKIP_SUBSTR)


def dequantize_q4_packed(
    packed: jnp.ndarray,  # uint8 [O, in//2]
    scale: jnp.ndarray,  # [O] or [O, G] fp
    zero: Optional[jnp.ndarray] = None,
    *,
    in_features: Optional[int] = None,
    group_size: int = 128,
    order: str = "low_first",
    dtype=jnp.bfloat16,
) -> jnp.ndarray:
    """Unpack nibbles → float32 → scale/zero → ``dtype``.

    Args:
        packed: uint8 packed int4; last dim holds 2 elems per byte.
        scale: per-channel [O] or per-group [O, G].
        zero: optional zero-points, same shape as scale (default 8 for
            unsigned GPTQ-style).
        in_features: trim unpacked columns to this width (odd ``in`` leaves
            one padding nibble in the last byte).
        group_size: columns per scale group (informational: the per-group
            path derives the group width from ``scale.shape``).
        order: "low_first" (GPTQ) or "high_first".
    """
    if order == "low_first":
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
    else:
        hi = packed & 0xF
        lo = (packed >> 4) & 0xF
    q = jnp.concatenate([lo[..., None], hi[..., None]], axis=-1)
    q = q.reshape(*packed.shape[:-1], packed.shape[-1] * 2).astype(jnp.float32)
    if in_features is not None and q.shape[-1] > in_features:
        q = q[..., :in_features]
    if scale.ndim == 1:
        s = scale.astype(jnp.float32)[:, None]
        z = (
            jnp.full_like(s, 8.0)
            if zero is None
            else zero.astype(jnp.float32)[:, None]
        )
        w = (q - z) * s
    else:
        # Per-group: scale [O, G].
        o, total_in = q.shape
        ng = scale.shape[1]
        gs = total_in // ng
        qg = q.reshape(o, ng, gs)
        s = scale.astype(jnp.float32)[:, :, None]
        z = (
            jnp.full_like(s, 8.0)
            if zero is None
            else zero.astype(jnp.float32)[:, :, None]
        )
        w = ((qg - z) * s).reshape(o, total_in)
    _ = group_size
    return w.astype(dtype)


__all__ = [
    "IGNORED_MISSING_SUFFIXES",
    "QUANT_SKIP_SUBSTR",
    "dequantize_q4_packed",
    "should_skip_quant",
]
