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

"""Mixture-of-Experts + dense MLP for Qwen4Exp, in JAX.

Upstream references:
- ``vllm/model_executor/models/qwen3_next.py``: ``Qwen3NextSparseMoeBlock``
  (gate → top-k → ``norm_topk_prob`` → ``FusedMoE`` with
  ``moe_intermediate_size``, plus shared-expert MLP) and ``Qwen3NextMLP``.
- ``vllm/models/qwen4_exp/nvidia/model.py``: ``Qwen4ExpSparseMoeBlock``
  (bans sequence-parallel MoE, exposes ``n_shared_experts``) and the
  ``is_moe_layer`` selection rule.

TPU strategy: reuse ``tpu_inference`` MoE backends when available
(``layers/jax/moe`` fused/GMM paths); fall back to a dense-mat reference
(``DENSE_MAT`` semantics: per-token top-k einsums in float32) so numerics
can be validated on CPU. Shared expert is a plain SwiGLU MLP added to the
routed sum.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


def silu(x: jax.Array) -> jax.Array:
    return jax.nn.silu(x)


class Qwen4ExpMLP(JaxModule):
    """Dense SwiGLU MLP (``Qwen3NextMLP`` equivalent)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act: str = "silu",
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        rngs = rngs or nnx.Rngs(0)
        self.gate_proj = JaxEinsum(
            "TD,DK->TK", (hidden_size, intermediate_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")), rngs=rngs,
            prefix=prefix + ".gate_proj")
        self.up_proj = JaxEinsum(
            "TD,DK->TK", (hidden_size, intermediate_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")), rngs=rngs,
            prefix=prefix + ".up_proj")
        self.down_proj = JaxEinsum(
            "TD,DK->TK", (intermediate_size, hidden_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, ("model", None)), rngs=rngs,
            prefix=prefix + ".down_proj")

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(jnp.float32)
        g = jnp.einsum("TD,DK->TK", xf, self.gate_proj.weight.value.astype(jnp.float32))
        u = jnp.einsum("TD,DK->TK", xf, self.up_proj.weight.value.astype(jnp.float32))
        h = silu(g) * u
        return jnp.einsum(
            "TK,KD->TD", h, self.down_proj.weight.value.astype(jnp.float32)
        ).astype(x.dtype)


class Qwen4ExpMoE(JaxModule):
    """Routed MoE + shared expert.

    Checkpoint (per MoE layer ``mlp.``):
    - ``gate.weight``: [num_experts, H]
    - ``experts.gate_up_proj`` fused [E, 2*moe_inter, H] or split
      ``gate_proj``/``up_proj`` + ``down_proj`` [E, H, moe_inter]
    - ``shared_expert.gate_proj/up_proj/down_proj`` (+ optional
      ``shared_expert_gate`` when fused — see ``maybe_fuse_shared_experts``).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.moe_inter = moe_intermediate_size
        self.shared_inter = shared_intermediate_size
        self.num_experts = num_experts
        self.topk = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        self.gate = JaxEinsum(
            "TD,DK->TK", (hidden_size, num_experts),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")), rngs=rngs,
            prefix=prefix + ".gate")
        # Fused gate_up: [E, 2*I, H] stored as [2*I, E*H]-ish einsum kernel
        # [H, E, 2I] for XLA-friendly gather.
        self.exp_gate_up = nnx.Param(
            jnp.zeros((hidden_size, num_experts, 2 * moe_intermediate_size),
                      dtype=jnp.float32))
        self.exp_down = nnx.Param(
            jnp.zeros((num_experts, moe_intermediate_size, hidden_size),
                      dtype=jnp.float32))
        self.n_shared_experts = int(shared_intermediate_size > 0)
        if self.n_shared_experts:
            self.shared = Qwen4ExpMLP(hidden_size, shared_intermediate_size,
                                      rngs=rngs, prefix=prefix + ".shared_expert")

    def route(
        self, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        logits = jnp.einsum(
            "TD,DK->TK", x.astype(jnp.float32),
            self.gate.weight.value.astype(jnp.float32))
        weights, idx = jax.lax.top_k(logits, self.topk)
        if self.norm_topk_prob:
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        return logits, weights.astype(x.dtype), idx.astype(jnp.int32)

    def __call__(
        self, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Returns (output, router_logits) — logits for EPLB/aux-loss hooks."""
        logits, weights, idx = self.route(x)  # [T,K]
        xf = x.astype(jnp.float32)
        gu = self.exp_gate_up.value.astype(jnp.float32)  # [H, E, 2I]
        dn = self.exp_down.value.astype(jnp.float32)  # [E, I, H]
        # Gather per-token expert weights: [T, K, H, 2I] is too big for large
        # E; gather only selected experts via take (dynamic gather, XLA-safe).
        def token_moe(xt, wt, idt):
            g = jnp.take(gu, idt, axis=1)  # [H, K, 2I]
            w = jnp.einsum("H,HKI->KI", xt, g)  # [K, 2I]
            i = self.moe_inter
            h = silu(w[:, :i]) * w[:, i:]
            d = jnp.take(dn, idt, axis=0)  # [K, I, H]
            o = jnp.einsum("KI,KIH->KH", h, d)  # [K, H]
            return jnp.sum(o * wt[:, None], axis=0)

        out = jax.vmap(token_moe)(xf, weights.astype(jnp.float32), idx)
        out = out.astype(x.dtype)
        if self.n_shared_experts:
            out = out + self.shared(x)
        return out, logits


__all__ = ["Qwen4ExpMLP", "Qwen4ExpMoE", "silu"]
