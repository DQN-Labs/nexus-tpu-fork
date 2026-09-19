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

"""Hyper-connection (GatedResidual) for Qwen4Exp, in JAX.

Upstream references:
- ``vllm/models/qwen4_exp/common/hyperconnection.py``: portable
  ``GatedResidual`` + ``GroupedGemmaRMSNorm`` (math reference).
- ``vllm/models/qwen4_exp/nvidia/hyperconnection.py``: fused
  mix / combine_and_mix used in the hot path (delayed combine).
- ``vllm/models/qwen4_exp/nvidia/ops/hc.py``: Triton kernels
  ``qwen4_exp_grouped_gemma_rmsnorm``, ``hc_silu``, ``hc_gate_mix``,
  ``hc_combine``, ``hc_combine_norm``.

Math (must match upstream exactly):
    xn = GroupedRMS(hyper)                      # [T, HC*H], per-H streams
    down, inj_logits = split(Down(xn))          # [T, lowrank], [T, HC]
    lora = silu(down / HC)
    gate = Up(lora)                             # [T, HC*H]
    block_in = mean_hc(sigmoid(gate) * xn)      # [T, H]
    combine: w = 2*sigmoid(inj/HC); hyper' = hyper + block_out[:,None,:]*w
    combine_and_mix fuses combine + next mix's GroupedRMS.

TPU strategy: express the same math with plain JAX ops (no Pallas
initially). Correctness first; fuse later if profiling demands it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover - allows CPU-only unit tests

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from ._jax_compat import JaxEinsum

_init = nnx.initializers.uniform()


def grouped_gemma_rmsnorm(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
    hc_count: int,
) -> jax.Array:
    """Per-H-stream Gemma RMSNorm: ``y = x/sqrt(mean(x^2)+eps) * (1+w)``.

    Args:
        x: [..., HC*H] multi-stream state.
        weight: [HC*H] ``(1+w)`` scale (checkpoint ``hc_norm.weight``,
            zero-initialized).
        hc_count: number of streams HC.
    """
    h = x.shape[-1] // hc_count
    xf = x.astype(jnp.float32)
    grouped = xf.reshape(*xf.shape[:-1], hc_count, h)
    var = jnp.mean(jnp.square(grouped), axis=-1, keepdims=True)
    normed = grouped * jax.lax.rsqrt(var + eps)
    normed = normed.reshape(*xf.shape[:-1], hc_count * h)
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


class GroupedGemmaRMSNorm(JaxModule):
    """Learnable grouped RMSNorm over ``[T, HC*H]``."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        eps: float = 1e-6,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.eps = eps
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        # Upstream inits hc_norm.weight to zeros (so scale starts at 1.0).
        self.weight = nnx.Param(
            jnp.zeros((hidden_size * hc_count,), dtype=jnp.float32)
        )
        del rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        return grouped_gemma_rmsnorm(x, self.weight.value, self.eps, self.hc_count)


class GatedResidual(JaxModule):
    """JAX equivalent of upstream ``GatedResidual``.

    Checkpoint layout (see ``nvidia/model.py::_EXTRA_WEIGHTS_MAPPER``):
    - ``input_mix_weight_down_block_inject.weight``: [(lowrank+HC), HC*H]
      merged from ``input_mix_weight_down`` (rows [0:lowrank]) and
      ``block_inject_weight`` (rows [lowrank:lowrank+HC]).
    - ``input_mix_weight_up.weight``: [HC*H, lowrank].
    - ``hc_norm.weight``: [HC*H].

    When ``use_combine=False`` (final mixer) there is no injection path;
    ``combine*`` must not be called.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float = 1e-6,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        use_combine: bool = True,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.use_combine = use_combine
        self.prefix = prefix
        self.dtype = dtype
        rngs = rngs or nnx.Rngs(0)
        wide = hidden_size * hc_count
        self.hc_norm = GroupedGemmaRMSNorm(
            hidden_size, hc_count, eps, dtype, rngs, prefix + ".hc_norm"
        )
        # Merged down projection: [HC*H] -> [lowrank + HC].
        # JaxEinsum (not raw nnx.Einsum): the ``.weight`` alias is what the
        # loader matches on (raw Einsum kernels are invisible to it).
        self.down_block_inject = JaxEinsum(
            "TD,DK->TK",
            (wide, hc_lowrank + hc_count),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")),
            rngs=rngs,
            prefix=prefix + ".down_block_inject",
        )
        self.up = JaxEinsum(
            "TD,DK->TK",
            (hc_lowrank, wide),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, ("model", None)),
            rngs=rngs,
            prefix=prefix + ".up",
        )

    # -- core ops -----------------------------------------------------
    def mix(
        self, hyper: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Split multi-stream state into block input + injection logits."""
        xn = self.hc_norm(hyper)
        wide = self.hidden_size * self.hc_count
        # Keep matmuls in float32 then cast back (matches bf16 numerics).
        proj = jnp.einsum(
            "TD,DK->TK",
            xn.astype(jnp.float32),
            self.down_block_inject.weight.value.astype(jnp.float32),
        )
        lora = proj[..., : self.hc_lowrank]
        inj = proj[..., self.hc_lowrank : self.hc_lowrank + self.hc_count]
        lora = jax.nn.silu(lora / float(self.hc_count))
        gate = jnp.einsum(
            "TD,DK->TK",
            lora,
            self.up.weight.value.astype(jnp.float32),
        )
        gate = gate.reshape(*hyper.shape[:-1], self.hc_count, self.hidden_size)
        xn_r = xn.reshape(*hyper.shape[:-1], self.hc_count, self.hidden_size).astype(
            jnp.float32
        )
        block_in = jnp.mean(
            jax.nn.sigmoid(gate) * xn_r, axis=-2
        ).astype(hyper.dtype)
        injection: Optional[jax.Array] = inj.astype(hyper.dtype)
        if not self.use_combine:
            injection = None
        return hyper, block_in, injection

    def combine(
        self,
        hyper: jax.Array,
        block_out: jax.Array,
        injection: Optional[jax.Array],
    ) -> jax.Array:
        """Materialize pending residual: hyper + block_out * w(injection)."""
        if injection is None:
            return hyper + jnp.concatenate(
                [block_out] * self.hc_count, axis=-1
            ).astype(hyper.dtype)
        w = (2.0 * jax.nn.sigmoid(injection.astype(jnp.float32) / float(self.hc_count)))
        w = w[..., None]  # [T, HC, 1]
        b = block_out.astype(jnp.float32)[:, None, :]
        combined = hyper.astype(jnp.float32).reshape(
            *hyper.shape[:-1], self.hc_count, self.hidden_size
        )
        return (combined + b * w).reshape(hyper.shape).astype(hyper.dtype)

    def combine_and_mix(
        self,
        hyper: jax.Array,
        block_out: jax.Array,
        injection: Optional[jax.Array],
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Fused combine + next mix (saves one grouped norm vs separate)."""
        materialized = self.combine(hyper, block_out, injection)
        _, block_in, new_inj = self.mix(materialized)
        return materialized, block_in, new_inj


__all__ = ["GatedResidual", "GroupedGemmaRMSNorm", "grouped_gemma_rmsnorm"]
