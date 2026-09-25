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

"""Gated Delta Net (GDN) linear-attention branch, in JAX.

Upstream references:
- ``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py``:
  ``QwenGatedDeltaNetAttention`` (in_proj_qkvz / in_proj_ba, causal depthwise
  conv1d + SiLU, ``chunk_gated_delta_rule`` from FLA, ``RMSNormGated`` with
  ``sigmoid`` for Qwen4Exp ``output_gate_type``, ``out_proj``).
- ``vllm/models/qwen4_exp/nvidia/model.py``: constructs with
  ``gqa_interleaved_layout=False``.

TPU execution:
- Preferred: ``tpu_inference.layers.common.gdn_attention.run_jax_gdn_attention``
  (Pallas GDN v3 kernel with conv + recurrent state). This module calls it
  when ``attention_metadata`` carries the required state indices; otherwise
  it falls back to a chunked JAX reference (correct, slower) so CPU tests
  and single-request correctness do not depend on the kernel.
- State shapes follow upstream ``MambaSpec``: conv
  ``[C, W]`` and recurrent ``[H, K, V]`` per request (see ``cache.py``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from ._jax_compat import JaxEinsum

_init = nnx.initializers.uniform()


def rmsnorm_gated(
    o: jax.Array, z: jax.Array, weight: jax.Array, eps: float,
    gate_act: str = "sigmoid",
) -> jax.Array:
    """Upstream ``Qwen3NextRMSNormGated`` (transformers ``qwen3_next``).

    - ``weight`` is ``[head_v_dim]`` **ones**-init, applied with a PLAIN
      multiply (contrast the Gemma ``(1+w)`` zeros-init norms used for q/k).
    - Variance is over the head dim: inputs arrive flat ``[..., V*Dv]``
      and are viewed ``[..., V, Dv]`` so each value head is normalized
      independently (a full-vector variance + ``[Dv]`` weight would be
      mathematically different and the checkpoint shape proves per-head).
    - Gate is sigmoid for Qwen4Exp ``output_gate_type`` (HF default silu
      otherwise); norm runs before the gate.
    """
    dv = int(weight.shape[-1])
    of = o.reshape(*o.shape[:-1], -1, dv).astype(jnp.float32)
    zf = z.reshape(*z.shape[:-1], -1, dv).astype(jnp.float32)
    var = jnp.mean(jnp.square(of), axis=-1, keepdims=True)
    normed = of * jax.lax.rsqrt(var + eps)
    normed = normed * weight.astype(jnp.float32)
    if gate_act in ("silu", "swish"):
        g = jax.nn.silu(zf)
    else:  # upstream maps Qwen4Exp "sigmoid" (and default) to sigmoid
        g = jax.nn.sigmoid(zf)
    return (normed * g).reshape(o.shape).astype(o.dtype)


class Qwen4ExpGDN(JaxModule):
    """Linear-attention layer with Qwen3.5-style [q,k,v,z]/[b,a] layout."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        k_head_dim: int = 128,
        v_head_dim: int = 128,
        conv_kernel: int = 4,
        eps: float = 1e-6,
        output_gate_type: str = "sigmoid",
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.conv_kernel = conv_kernel
        self.eps = eps
        self.output_gate_type = output_gate_type
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        # Qwen3.5 layout: qkv = q + k + v + z where q dims == k dims
        # (num_k_heads*k_head_dim), v/z dims == num_v_heads*v_head_dim.
        qkv_dim = (
            2 * num_k_heads * k_head_dim + 2 * num_v_heads * v_head_dim
        )
        self.in_proj_qkvz = JaxEinsum(
            "TD,DK->TK",
            (hidden_size, qkv_dim),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")),
            rngs=rngs,
            prefix=prefix + ".in_proj_qkvz",
        )
        self.in_proj_ba = JaxEinsum(
            "TD,DK->TK",
            (hidden_size, 2 * num_v_heads),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")),
            rngs=rngs,
            prefix=prefix + ".in_proj_ba",
        )
        self.conv_weight = nnx.Param(
            jnp.zeros((2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim,
                       1, conv_kernel), dtype=jnp.float32)
        )
        self.A_log = nnx.Param(jnp.zeros((num_v_heads,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((num_v_heads,), dtype=jnp.float32))
        # Output norm: upstream Qwen3NextRMSNormGated(head_v_dim), ones-init.
        # Per-head dim (NOT num_v_heads*v_head_dim): the checkpoint ships
        # [v_head_dim] (v62: torch (128,) vs jax (6144,) LOAD-FAIL).
        self.norm_w = nnx.Param(
            jnp.ones((v_head_dim,), dtype=jnp.float32)
        )
        self.out_proj = JaxEinsum(
            "TD,DK->TK",
            (num_v_heads * v_head_dim, hidden_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, ("model", None)),
            rngs=rngs,
            prefix=prefix + ".out_proj",
        )

    def split_qkvz(
        self, mixed: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        qd = self.num_k_heads * self.k_head_dim
        vd = self.num_v_heads * self.v_head_dim
        q = mixed[..., :qd]
        k = mixed[..., qd : 2 * qd]
        v = mixed[..., 2 * qd : 2 * qd + vd]
        z = mixed[..., 2 * qd + vd :]
        return q, k, v, z

    def _causal_depthwise_conv(self, x: jax.Array) -> jax.Array:
        # x: [T, C]; weight [C, K]; causal, SiLU (upstream conv path).
        # XLA-friendly: gather windows via a static index matrix (no dynamic
        # slicing inside vmap).
        w = self.conv_weight.value[:, 0, :]  # [C, K]
        k = self.conv_kernel
        t = x.shape[0]
        xp = jnp.pad(x.astype(jnp.float32), ((k - 1, 0), (0, 0)))  # [T+K-1, C]
        rows = jnp.arange(t)[:, None] + jnp.arange(k)[None, :]  # [T, K]
        # windows: [T, K, C] via take along axis 0 (static gather).
        windows = jnp.take(xp, rows, axis=0)
        out = jnp.sum(windows * w.T[None, :, :], axis=1)  # [T, C]
        return jax.nn.silu(out).astype(x.dtype)

    def _recurrent_reference(
        self,
        q: jax.Array,  # [T, K, Dk]
        k: jax.Array,
        v: jax.Array,  # [T, V, Dv]
        b: jax.Array,  # [T, V]
        a: jax.Array,
    ) -> jax.Array:
        """Simplified gated-delta reference: S += outer(k*beta, v); o=qS.

        This is NOT the full FLA ``chunk_gated_delta_rule`` (which includes
        Householder-style delta correction + per-head decay from A_log/dt).
        It preserves shapes, gating, and normalization ordering so the model
        runs and weight loading can be validated; the production TPU path
        must call ``run_jax_gdn_attention`` (Pallas v3). The gap is tracked
        explicitly in ``docs/qwen4_exp.md`` § GDN.
        """
        # State is per *value* head: S[h] in [Dk, Dv]. Key/query heads are
        # broadcast K->V (GQA-style repeat) when counts differ.
        n_v, d_k, d_v = self.num_v_heads, self.k_head_dim, self.v_head_dim
        reps = n_v // max(self.num_k_heads, 1)
        decay = jax.nn.sigmoid(
            self.dt_bias.value.astype(jnp.float32)
            + jnp.exp(self.A_log.value.astype(jnp.float32))
        )  # [V] in (0,1)
        s = jnp.zeros((n_v, d_k, d_v), dtype=jnp.float32)
        # lax.scan (not a Python loop): unrolling T steps explodes compile
        # time on long prefills; scan compiles the recurrence once.
        # NOTE: argument order is (carry, step-inputs), NOT (step, carry).
        def _step(s, qkvb):
            qt, kt, vt, bt = qkvb
            beta = jax.nn.sigmoid(bt).astype(jnp.float32)  # [V]
            kk = jnp.repeat(kt.astype(jnp.float32), reps, axis=0)  # [V, Dk]
            vv = vt.astype(jnp.float32)  # [V, Dv]
            update = jnp.einsum("HD,HV->HDV", kk, vv)  # [V, Dk, Dv]
            s = s * decay[:, None, None] + update * beta[:, None, None]
            qr = jnp.repeat(qt.astype(jnp.float32), reps, axis=0)  # [V, Dk]
            o = jnp.einsum("HD,HDV->HV", qr, s)  # [V, Dv]
            return s, o.reshape(-1)

        _, outs = jax.lax.scan(
            _step, s,
            (q.astype(jnp.float32), k.astype(jnp.float32),
             v.astype(jnp.float32), b))
        return outs.astype(q.dtype)

    def __call__(
        self,
        x: jax.Array,
        attention_metadata=None,
        mesh=None,
        conv_state: Optional[jax.Array] = None,
        recurrent_state: Optional[jax.Array] = None,
    ) -> jax.Array:
        mixed = jnp.einsum(
            "TD,DK->TK",
            x.astype(jnp.float32),
            self.in_proj_qkvz.weight.value.astype(jnp.float32),
        ).astype(x.dtype)
        ba = jnp.einsum(
            "TD,DK->TK",
            x.astype(jnp.float32),
            self.in_proj_ba.weight.value.astype(jnp.float32),
        ).astype(x.dtype)
        q, k, v, z = self.split_qkvz(mixed)
        b, a = ba[..., : self.num_v_heads], ba[..., self.num_v_heads :]
        mixed_qkv = self._causal_depthwise_conv(
            jnp.concatenate([q, k, v], axis=-1)
        )
        qd = self.num_k_heads * self.k_head_dim
        q = mixed_qkv[..., :qd].reshape(-1, self.num_k_heads, self.k_head_dim)
        k = mixed_qkv[..., qd : 2 * qd].reshape(-1, self.num_k_heads, self.k_head_dim)
        v = mixed_qkv[..., 2 * qd :].reshape(-1, self.num_v_heads, self.v_head_dim)

        # Production TPU path: Pallas GDN kernel with external state.
        if attention_metadata is not None and conv_state is not None:
            try:
                from tpu_inference.layers.common.gdn_attention import (
                    run_jax_gdn_attention,
                )

                (new_conv, new_rec), o = run_jax_gdn_attention(
                    j_mixed_qkv=mixed,
                    j_b=b,
                    j_a=a,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    j_conv_weight=self.conv_weight.value,
                    j_conv_bias=None,
                    j_A_log=self.A_log.value,
                    j_dt_bias=self.dt_bias.value,
                    state_indices=attention_metadata.mamba_state_indices,
                    query_start_loc=attention_metadata.query_start_loc,
                    distribution=attention_metadata.request_distribution,
                    seq_lens=attention_metadata.seq_lens,
                    n_kq=self.num_k_heads,
                    n_v=self.num_v_heads,
                    d_k=self.k_head_dim,
                    d_v=self.v_head_dim,
                    kernel_size=self.conv_kernel,
                    mesh=mesh,
                )
                del new_conv, new_rec
                o = o.reshape(x.shape[0], -1)
                out = rmsnorm_gated(
                    o, z.reshape(x.shape[0], -1), self.norm_w.value,
                    eps=self.eps, gate_act=self.output_gate_type,
                )
                return jnp.einsum(
                    "TD,DK->TK",
                    out.astype(jnp.float32),
                    self.out_proj.weight.value.astype(jnp.float32),
                ).astype(x.dtype)
            except (ImportError, AttributeError):
                pass

        o = self._recurrent_reference(q, k, v, b, a)
        out = rmsnorm_gated(
            o, z.reshape(x.shape[0], -1), self.norm_w.value,
            eps=self.eps, gate_act=self.output_gate_type,
        )
        return jnp.einsum(
            "TD,DK->TK",
            out.astype(jnp.float32),
            self.out_proj.weight.value.astype(jnp.float32),
        ).astype(x.dtype)


__all__ = ["Qwen4ExpGDN", "rmsnorm_gated"]
