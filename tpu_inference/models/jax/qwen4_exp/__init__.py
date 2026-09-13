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

"""Qwen4Exp JAX package: Qwen3.8-Flash-Next TPU support.

Upstream canonical reference: ``vllm/models/qwen4_exp/`` (nvidia + amd +
common), ``vllm/model_executor/models/qwen3_next.py``,
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py``.
TPU skeleton: ``tpu_inference/models/jax/qwen3.py``.
Full mapping: ``docs/qwen4_exp.md``.
"""

from .cache import (
    Qwen4ExpCacheSpec,
    circular_capacity,
    circular_slot,
    compressed_slot,
    short_conv_state_len,
)
from .config import Qwen4ExpArch, arch_from_hf_config, is_moe_layer
from .model import Qwen4ExpForCausalLM, Qwen4ExpMTP, Qwen4ExpModel

_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
)


def register(register_model=None):
    """Out-of-tree registration (keeps the fork clean).

    Usage::
        from tpu_inference.models.jax.qwen4_exp import register
        register()  # registers Qwen4ExpForCausalLM -> JAX impl

    Falls back to a clear error when ``tpu_inference`` is not installed
    (never silently fall back to CPU — task §12).
    """
    if register_model is None:
        try:
            from tpu_inference.models.common.model_loader import (
                register_model as _reg,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "tpu_inference is required to register Qwen4Exp. Install the "
                "fork first (see scripts/install_clean.sh)."
            ) from e
        register_model = _reg
    register_model("Qwen4ExpForCausalLM", Qwen4ExpForCausalLM)
    register_model("Qwen4ExpForConditionalGeneration", Qwen4ExpForCausalLM)
    # MTP is a draft-model arch only; the validator on newer tpu-inference
    # requires a runner-conformant __call__, so skip at publish time if it
    # would reject the stub. The inspect path (config + weight-loader) does
    # not need serving.
    try:
        register_model("Qwen4ExpMTP", Qwen4ExpMTP)
    except TypeError as e:
        import warnings

        warnings.warn(f"Skipping Qwen4ExpMTP registration: {e}")
    return {
        "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
        "Qwen4ExpForConditionalGeneration": Qwen4ExpForCausalLM,
        "Qwen4ExpMTP": Qwen4ExpMTP,
    }


__all__ = [
    "Qwen4ExpArch",
    "Qwen4ExpCacheSpec",
    "_ARCHITECTURES",
    "arch_from_hf_config",
    "circular_capacity",
    "circular_slot",
    "compressed_slot",
    "is_moe_layer",
    "register",
    "short_conv_state_len",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpMTP",
    "Qwen4ExpModel",
]
