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

    INT4 HBM residency (v5e-8 cannot hold this MoE dequantized): experts
    stay packed (ModelOpt NVFP4 layout) and only the top-k selected experts
    per token are dequantized in-forward. Checkpoint (per MoE layer
    ``mlp.``): ``gate.weight`` [E, H] plus per expert
    ``experts.{e}.{gate,up,down}_proj`` x ``{weight u8, weight_scale fp8,
    weight_scale_2 fp32 scalar}`` (``input_scale`` unused, activations stay
    bf16). Shared expert is plain BF16 (``shared_expert.*_proj``).
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
        # NVFP4 triplets per projection (torch layout per expert row-block):
        # gate/up: weight [E, I, H//2] u8, scales [E, I, H//16] fp8,
        #   global [E] fp32; down: weight [E, H, I//2] u8,
        #   scales [E, H, I//16] fp8, global [E] fp32.
        i, h, e = moe_intermediate_size, hidden_size, num_experts
        # Narrow-linears guard: degenerate (in < 16) projections carry a
        # single scale column (real exports always satisfy in >= 16).
        self.exp_gate_w = nnx.Param(jnp.zeros((e, i, h // 2), jnp.uint8))
        self.exp_gate_sc = nnx.Param(
            jnp.zeros((e, i, max(1, h // 16)), jnp.float8_e4m3fn))
        self.exp_gate_g = nnx.Param(jnp.zeros((e,), jnp.float32))
        self.exp_up_w = nnx.Param(jnp.zeros((e, i, h // 2), jnp.uint8))
        self.exp_up_sc = nnx.Param(
            jnp.zeros((e, i, max(1, h // 16)), jnp.float8_e4m3fn))
        self.exp_up_g = nnx.Param(jnp.zeros((e,), jnp.float32))
        self.exp_down_w = nnx.Param(jnp.zeros((e, h, i // 2), jnp.uint8))
        self.exp_down_sc = nnx.Param(
            jnp.zeros((e, h, max(1, i // 16)), jnp.float8_e4m3fn))
        self.exp_down_g = nnx.Param(jnp.zeros((e,), jnp.float32))
        self.n_shared_experts = int(shared_intermediate_size > 0)
        # NOTE: attribute name IS the load contract (nnx paths derive from
        # it): ``shared_expert`` matches the checkpoint + loader
        # (v58 LOAD-FAIL class: abbreviated ``shared`` live name).
        if self.n_shared_experts:
            self.shared_expert = Qwen4ExpMLP(
                hidden_size, shared_intermediate_size,
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
        from .quant import dequantize_nvfp4_jax

        logits, weights, idx = self.route(x)  # [T,K]
        xf = x.astype(jnp.float32)
        gw, gs, gg = (self.exp_gate_w.value, self.exp_gate_sc.value,
                      self.exp_gate_g.value)
        uw, us, ug = (self.exp_up_w.value, self.exp_up_sc.value,
                      self.exp_up_g.value)
        dw, ds, dg = (self.exp_down_w.value, self.exp_down_sc.value,
                      self.exp_down_g.value)

        def _deq(w, s, g):
            return jax.vmap(dequantize_nvfp4_jax, in_axes=(0, 0, 0))(w, s, g)

        # Gather + dequantize ONLY the selected experts (dynamic gather,
        # XLA-safe; full dequant of all E experts per step would be ~250x
        # the traffic on decode).
        def token_moe(xt, wt, idt):
            g = _deq(jnp.take(gw, idt, axis=0), jnp.take(gs, idt, axis=0),
                     jnp.take(gg, idt, axis=0))  # [K, I, H]
            u = _deq(jnp.take(uw, idt, axis=0), jnp.take(us, idt, axis=0),
                     jnp.take(ug, idt, axis=0))  # [K, I, H]
            w = jnp.einsum("H,KIH->KI", xt, g)
            h = silu(w) * jnp.einsum("H,KIH->KI", xt, u)
            d = _deq(jnp.take(dw, idt, axis=0), jnp.take(ds, idt, axis=0),
                     jnp.take(dg, idt, axis=0))  # [K, H, I]
            o = jnp.einsum("KI,KIH->KH", h, d.transpose(0, 2, 1))  # [K, H]
            return jnp.sum(o * wt[:, None], axis=0)

        out = jax.vmap(token_moe)(xf, weights.astype(jnp.float32), idx)
        out = out.astype(x.dtype)
        if self.n_shared_experts:
            out = out + self.shared_expert(x)
        return out, logits


__all__ = ["Qwen4ExpMLP", "Qwen4ExpMoE", "silu"]
