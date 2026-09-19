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

"""Framework layer aliases with CPU-test fallbacks.

On any env with ``tpu-inference`` installed (notably the Kaggle TPU VM),
``JaxEinsum`` / ``JaxEmbed`` are the real framework classes. They alias
``kernel``/``embedding`` to ``weight`` and derive from ``JaxModule``, which
is what makes parameters visible to ``named_parameters()`` and loadable by
name — raw ``nnx.Einsum``/``nnx.Embed`` are invisible to the loader walk.

In minimal CPU test envs (no tpu-inference) thin ``nnx`` shims with
identical numerics and the same ``.weight`` naming are used instead, so
unit tests exercise the same code paths.
"""

from __future__ import annotations

from flax import nnx

try:
    from tpu_inference.layers.jax.embed import JaxEmbed as _RealEmbed
    from tpu_inference.layers.jax.linear import JaxEinsum as _RealEinsum

    JaxEinsum = _RealEinsum
    JaxEmbed = _RealEmbed
    _USING_REAL = True
except ImportError:  # pragma: no cover - CPU-only test fallback

    class JaxEinsum(nnx.Einsum):  # type: ignore[no-redef]
        """Fallback: nnx.Einsum + ``kernel`` -> ``weight`` alias."""

        def __init__(self, *args, prefix: str = "",
                     quant_config=None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.weight = self.kernel
            delattr(self, "kernel")

    class JaxEmbed(nnx.Embed):  # type: ignore[no-redef]
        """Fallback: nnx.Embed + ``embedding`` -> ``weight`` alias."""

        def __init__(self, *args, prefix: str = "",
                     quant_config=None, dtype=None, **kwargs) -> None:
            if dtype is not None and "param_dtype" not in kwargs:
                kwargs["param_dtype"] = dtype
            super().__init__(*args, **kwargs)
            self.weight = self.embedding
            delattr(self, "embedding")

        def __call__(self, x):
            return self.weight.value[x]

    _USING_REAL = False


__all__ = ["JaxEmbed", "JaxEinsum", "_USING_REAL"]
