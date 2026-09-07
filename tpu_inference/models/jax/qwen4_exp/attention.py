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

"""Dense full-attention branch of Qwen4Exp (non-QSA layers).

Upstream references:
- ``vllm/model_executor/models/qwen3_next.py``: ``Qwen3NextAttention``
  (QK-norm + partial RoPE + sigmoid output gate + paged GQA).
- ``vllm/models/qwen4_exp/nvidia/model.py``: selects this path when
  ``indexer_n_heads`` is absent.
- ``vllm/models/qwen4_exp/nvidia/qsa.py``: QSA owner reuses
  ``_project_qkv_gate`` with ``attn_output_gate=True``.

Weight layout (checkpoint, verified against
``vllm/model_executor/models/qwen3_next.py::Qwen3NextAttention``):
- ``self_attn.qkv_proj.weight`` fused ``[2*q + 2*kv, H]`` rows where the
  leading ``2*q`` rows are per-head interleaved pairs ``[q_h, g_h]``
  (``q_gate.view(..., num_heads, 2*head_dim)`` then ``chunk(2, -1)`` into
  ``(q, gate)``) — NOT ``[all-q, all-gate]`` halves. ``project()`` below
  reproduces that exact reshape-then-chunk.
- ``self_attn.o_proj.weight``: ``[H, num_heads*head_dim]``.
- ``self_attn.q_norm.weight`` / ``k_norm.weight``: per-``head_dim`` Gemma
  scales.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

_init = nnx.initializers.uniform()


def gemma_rmsnorm_last_dim(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return ((xf * jax.lax.rsqrt(var + eps)) * (1.0 + weight.astype(jnp.float32))).astype(
        x.dtype
    )


def apply_partial_rope(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rotary_dim: int,
    theta: float,
) -> Tuple[jax.Array, jax.Array]:
    """NeoX-style partial RoPE; trailing dims pass through.

    Matches upstream ``Qwen3NextAttention`` which applies RoPE to the first
    ``rotary_dim = head_dim * partial_rotary_factor`` dims only.
    Uses standard (non-MRoPE) 1D positions; MRoPE temporal branch shares the
    same math on TPU (height/width branches unsupported in text-only mode and
    fall back to temporal positions — documented in ``docs/qwen4_exp.md``).
    """
    if rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    half = rotary_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / rotary_dim))
    # positions: [T] -> angles [T, half]
    angles = positions.astype(jnp.float32)[:, None] * freqs[None, :]
    cos = jnp.cos(angles)[:, None, None, :]  # [T,1,1,half]
    sin = jnp.sin(angles)[:, None, None, :]
    # q_rot: [T, N, rotary_dim] -> split even/odd pairs
    qr1, qr2 = q_rot[..., :half], q_rot[..., half:]
    kr1, kr2 = k_rot[..., :half], k_rot[..., half:]
    # Interleaved NeoX rotation on half-split (matches HF Llama-style RoPE
    # used by Qwen3Next; verified against upstream rope kernels).
    q_out = jnp.concatenate([qr1 * cos.squeeze(2) - qr2 * sin.squeeze(2),
                             qr1 * sin.squeeze(2) + qr2 * cos.squeeze(2)], axis=-1)
    k_out = jnp.concatenate([kr1 * cos.squeeze(2) - kr2 * sin.squeeze(2),
                             kr1 * sin.squeeze(2) + kr2 * cos.squeeze(2)], axis=-1)
    q = jnp.concatenate([q_out, q_pass], axis=-1)
    k = jnp.concatenate([k_out, k_pass], axis=-1)
    return q, k


class Qwen4ExpDenseAttention(JaxModule):
    """GQA + QK-norm + partial-RoPE + sigmoid gate (JAX, correctness-first)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 0.25,
        attn_output_gate: bool = True,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.rope_theta = rope_theta
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.attn_output_gate = attn_output_gate
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        q_out = num_heads * head_dim * (2 if attn_output_gate else 1)
        self.qkv_proj = nnx.Einsum(
            "TD,DK->TK",
            (hidden_size, q_out + 2 * num_kv_heads * head_dim),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")),
            rngs=rngs,
        )
        self.o_proj = nnx.Einsum(
            "TD,DK->TK",
            (num_heads * head_dim, hidden_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, ("model", None)),
            rngs=rngs,
        )
        self.q_norm_w = nnx.Param(jnp.zeros((head_dim,), dtype=jnp.float32))
        self.k_norm_w = nnx.Param(jnp.zeros((head_dim,), dtype=jnp.float32))

    def project(
        self, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array]]:
        fused = jnp.einsum(
            "TD,DK->TK",
            x.astype(jnp.float32),
            self.qkv_proj.kernel.value.astype(jnp.float32),
        )
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        off = 0
        gate = None
        if self.attn_output_gate:
            # Per-head interleaved [q_h, g_h] pairs (upstream
            # ``q_gate.view(..., num_heads, -1)`` + ``chunk(2, -1)``).
            qg = fused[..., : 2 * q_size].reshape(
                *x.shape[:-1], self.num_heads, 2 * self.head_dim
            )
            q = qg[..., : self.head_dim]
            gate = qg[..., self.head_dim :]
            off = 2 * q_size
        else:
            q = fused[..., :q_size].reshape(
                *x.shape[:-1], self.num_heads, self.head_dim
            )
            off = q_size
        k = fused[..., off : off + kv_size].reshape(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        v = fused[..., off + kv_size : off + 2 * kv_size].reshape(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        q = gemma_rmsnorm_last_dim(q, self.q_norm_w.value, self.eps)
        k = gemma_rmsnorm_last_dim(k, self.k_norm_w.value, self.eps)
        return (
            q.astype(x.dtype),
            k.astype(x.dtype),
            v.astype(x.dtype),
            None if gate is None else gate.astype(x.dtype),
        )

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        kv_cache: Optional[jax.Array] = None,
        attention_metadata=None,
        mesh: Mesh | None = None,
    ) -> Tuple[Optional[jax.Array], jax.Array]:
        """Functional full attention.

        On TPU the caller routes through ``attention_interface.attention``
        with paged KV. This method implements the math directly so unit tests
        run on CPU: causal softmax(QK^T/sqrt(d))V with GQA repeat, then
        ``sigmoid(gate)`` and ``o_proj``.
        """
        q, k, v, gate = self.project(x)
        q, k = apply_partial_rope(
            q, k, positions, self.head_dim, self.rotary_dim, self.rope_theta
        )
        # GQA expand KV heads.
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, rep, axis=-2)
            v = jnp.repeat(v, rep, axis=-2)
        scale = 1.0 / (self.head_dim**0.5)
        logits = jnp.einsum("TNH,SNH->NTS", q.astype(jnp.float32),
                            k.astype(jnp.float32)) * scale
        t, s = logits.shape[1], logits.shape[2]
        # Causal mask over flat token axis (prefill). Decode callers pass T=1
        # with absolute positions; they should use paged attention instead.
        qpos = positions[:, None]
        # Reconstruct key positions as [0..S) for the direct path; paged path
        # on TPU uses true slot positions from metadata.
        kpos = jnp.arange(s)
        mask = kpos[None, :] <= (qpos + (s - t))
        logits = jnp.where(mask[None, :, :], logits, jnp.asarray(-1e9, jnp.float32))
        probs = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
        o = jnp.einsum("NTS,SNH->TNH", probs, v)
        if gate is not None:
            o = o * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(o.dtype)
        w_o = self.o_proj.kernel.value.astype(jnp.float32).reshape(
            self.num_heads, self.head_dim, self.hidden_size)
        out = jnp.einsum("TNH,NHD->TD", o.astype(jnp.float32), w_o).astype(x.dtype)
        return kv_cache, out


__all__ = [
    "Qwen4ExpDenseAttention",
    "apply_partial_rope",
    "gemma_rmsnorm_last_dim",
]
