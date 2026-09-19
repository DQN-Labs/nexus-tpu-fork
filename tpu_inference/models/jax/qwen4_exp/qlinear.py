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

"""INT4-residency linear for serving GPTQ weights on TPU.

vLLM's GPTQ path is CUDA-only (rejected on TPU), and a full-precision
materialization of this MoE (~100-200B params + 102 GB n-gram table) does
not fit v5e-8's 128 GB HBM. So weights stay packed (GPTQModel-v2 layout)
in HBM at ~0.53 bytes/param and are expanded per matmul in XLA by
``quant.dequantize_gptq_jax`` (proven bit-exact vs the CPU reference in
``test_gptq_jax_matches_torch_bit_exact``), followed by a bf16 dot.

Layout per linear (torch convention, ``in``/``out`` as in nn.Linear):
- ``qweight`` int32 ``(in//pack, out)``, pack = 8 for 4-bit
- ``qzeros`` int32 ``(groups, out//pack)``
- ``scales`` bf16 ``(groups, out)``
- ``g_idx`` int32 ``(in,)`` group index per input row

All four are ``nnx.Param`` so the loader fills them by exact JAX name;
the loader assigns raw shards directly (no transpose heuristics apply to
these names).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from .quant import dequantize_gptq_jax


class Qwen4ExpQLinear(JaxModule):
    """INT4 GPTQ linear with dequant-in-forward (see module docstring)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int = 4,
        group_size: int = 128,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(f"QLinear supports bits 2/4/8, got {bits}")
        pack = 32 // bits
        if in_features % pack:
            raise ValueError(
                f"in_features {in_features} must be divisible by pack "
                f"{pack} for bits={bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.dtype = dtype
        self.prefix = prefix
        del rngs  # quantized params come from the checkpoint, never random
        num_groups = (in_features // group_size if group_size != -1 else 1)
        self.qweight = nnx.Param(
            jnp.zeros((in_features // pack, out_features), dtype=jnp.int32))
        self.qzeros = nnx.Param(
            jnp.zeros((num_groups, out_features // pack), dtype=jnp.int32))
        self.scales = nnx.Param(
            jnp.zeros((num_groups, out_features), dtype=jnp.bfloat16))
        self.g_idx = nnx.Param(
            jnp.zeros((in_features,), dtype=jnp.int32))

    def dequantized(self) -> jax.Array:
        """Full float32 ``(in, out)`` weight (primarily for tests)."""
        return dequantize_gptq_jax(
            self.qweight.value, self.qzeros.value, self.scales.value,
            self.g_idx.value, bits=self.bits, group_size=self.group_size)

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.dequantized()
        return jnp.einsum(
            "TD,DK->TK", x.astype(jnp.float32),
            w).astype(self.dtype)


__all__ = ["Qwen4ExpQLinear"]
