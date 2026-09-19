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


def dequantize_gptq_torch(qweight, qzeros, scales, g_idx, *, bits=4,
                          group_size=128):
    """CPU dequant of one GPTQModel-v2 linear to float32 ``(in, out)``.

    Exact inverse of GPTQModel ``pack_block`` (low-nibble-first int32
    packing, zeros stored directly, ``W = scales[g] * (codes - zeros[g])``;
    verified against ``gptqmodel/nn_modules/qlinear/{__init__.py,torch.py}``).
    Pinned contract: INT4, group 128, asymmetric (zeros kept),
    ``DescAct=False`` — but ``g_idx`` is honored as stored, so any order
    works. Raises ``ValueError`` with shapes on any layout mismatch (fail
    fast in the server log rather than silently mis-loading).
    """
    import torch

    if bits not in (2, 4, 8):
        raise ValueError(f"GPTQ dequant supports bits 2/4/8, got {bits}")
    pack = 32 // bits
    maxq = (1 << bits) - 1
    qw = qweight.to(torch.int32)
    if qw.ndim != 2:
        raise ValueError(f"qweight must be 2D, got {tuple(qw.shape)}")
    in_packed, out = qw.shape
    shifts = (torch.arange(pack, dtype=torch.int32) * bits).view(1, pack, 1)
    codes = (torch.bitwise_right_shift(
        qw.unsqueeze(1).expand(-1, pack, -1), shifts) & maxq).to(torch.float32)
    codes = codes.reshape(in_packed * pack, out)
    if scales.ndim != 2:
        raise ValueError(f"scales must be 2D, got {tuple(scales.shape)}")
    num_groups, scales_out = scales.shape
    if scales_out != out:
        raise ValueError(
            f"scales out dim {scales_out} != qweight out dim {out}")
    qz = qzeros.to(torch.int32)
    if qz.ndim != 2 or qz.shape[0] != num_groups:
        raise ValueError(
            f"qzeros shape {tuple(qz.shape)} incompatible with "
            f"{num_groups} groups x {out} out")
    zshifts = (torch.arange(pack, dtype=torch.int32) * bits).view(1, 1, pack)
    zeros = (torch.bitwise_right_shift(
        qz.unsqueeze(2).expand(-1, -1, pack), zshifts) & maxq).to(torch.float32)
    zeros = zeros.reshape(num_groups, -1)[:, :out]
    g = g_idx.reshape(-1).to(torch.long)
    in_features = int(g.numel())
    if codes.shape[0] < in_features:
        raise ValueError(
            f"unpacked rows {codes.shape[0]} < g_idx len {in_features}")
    codes = codes[:in_features]
    if group_size != -1 and in_features // group_size != num_groups:
        raise ValueError(
            f"in_features {in_features} // group_size {group_size} != "
            f"{num_groups} scale groups")
    if int(g.min()) < 0 or int(g.max()) >= num_groups:
        raise ValueError(
            f"g_idx range [{int(g.min())}, {int(g.max())}] outside "
            f"{num_groups} groups")
    s = scales.to(torch.float32)
    return s[g] * (codes - zeros[g])


def dequantize_gptq_jax(qweight, qzeros, scales, g_idx, *, bits=4,
                        group_size=128):
    """JAX twin of :func:`dequantize_gptq_torch` (identical integer math).

    Used for dequant-in-forward under INT4 HBM residency: weights stay
    packed int32 + bf16 scales on device and are expanded per matmul in
    XLA (elementwise shifts/masks/gathers + bf16 dot), so v5e-8 HBM holds
    the ~4-bit checkpoint instead of a full-precision copy. Returns
    float32 ``(in, out)``; callers transpose/cast for their layout.
    """
    pack = 32 // bits
    maxq = (1 << bits) - 1
    qw = qweight.astype(jnp.int32)
    in_packed, out = qw.shape
    shifts = (jnp.arange(pack, dtype=jnp.int32) * bits).reshape(1, pack, 1)
    codes = jnp.broadcast_to(qw[:, None, :], (in_packed, pack, out))
    codes = ((codes >> shifts) & maxq).astype(jnp.float32)
    codes = codes.reshape(in_packed * pack, out)
    num_groups = scales.shape[0]
    qz = qzeros.astype(jnp.int32)
    zshifts = (jnp.arange(pack, dtype=jnp.int32) * bits).reshape(1, 1, pack)
    zeros = jnp.broadcast_to(qz[:, :, None],
                             (num_groups, qz.shape[1], pack))
    zeros = ((zeros >> zshifts) & maxq).astype(jnp.float32)
    zeros = zeros.reshape(num_groups, -1)[:, :out]
    g = g_idx.reshape(-1).astype(jnp.int32)
    in_features = int(g.shape[0])
    codes = codes[:in_features]
    s = scales.astype(jnp.float32)
    return s[g] * (codes - zeros[g])


__all__ = [
    "IGNORED_MISSING_SUFFIXES",
    "QUANT_SKIP_SUBSTR",
    "dequantize_gptq_jax",
    "dequantize_gptq_torch",
    "dequantize_q4_packed",
    "should_skip_quant",
]
