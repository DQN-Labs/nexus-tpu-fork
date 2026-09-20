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

"""Fresh-interpreter startup hook for Qwen4Exp serving.

Imported two ways:
1. Directly by the fork notebook cell (in-process install + proof).
2. Automatically by EVERY new Python process via the
   ``qwen4exp_tpu_startup.pth`` file the fork cell drops into site-packages
   (``.pth`` ``import`` lines run at interpreter startup). This is what
   reaches the ``vllm serve`` subprocess, which otherwise would neither
   parse the checkpoint (transformers lacks ``qwen4_exp``) nor resolve the
   arch to our JAX implementation (registries are per-process).

Importing this module performs the install. Registration of the JAX model
is best-effort-guarded (warn, don't crash third-party processes); the HF
config registration must succeed and raises loudly if it cannot.
"""

from __future__ import annotations


def _guard_qwix_null_quantization_config() -> None:
    """Guard tpu-inference's qwix default-config probe against a nulled
    quantization_config (v52 died here).

    We null the checkpoint's quantization_config (via --hf-overrides, since
    vLLM re-attaches it post-parse from the raw dict / hf_quant_config.json)
    so vLLM never builds its CUDA-only quant path. But
    ``get_default_qwix_quantization_config`` does
    ``hf_config.quantization_config["quant_method"]`` guarded only by
    ``hasattr`` — None passes hasattr and explodes on subscript. Patched
    copy treats non-dict as absent (exact 0.28.0 semantics otherwise).
    Best-effort: never break interpreter startup.
    """
    try:
        from tpu_inference.models.jax.utils.qwix import (
            qwix_utils as _qwix,
        )
    except Exception:
        return
    try:
        _orig = _qwix.get_default_qwix_quantization_config

        def _patched(hf_config, skip_quantization):
            if skip_quantization:
                return None
            qc = getattr(hf_config, "quantization_config", None)
            if not isinstance(qc, dict):
                qc = None
            model_type = getattr(hf_config, "model_type", None)
            model_type = model_type.lower() \
                if isinstance(model_type, str) else None
            quant_method = qc.get("quant_method") \
                if isinstance(qc, dict) else None
            if model_type == "llama4" \
                    and quant_method == "compressed-tensors":
                return _qwix.DEFAULT_LLAMA4_FP8_CONFIG
            if model_type == "gpt_oss" and quant_method == "mxfp4":
                return _qwix.DEFAULT_GPT_OSS_FP4_CONFIG
            return None

        _qwix.get_default_qwix_quantization_config = _patched
    except Exception:
        pass


def install() -> dict:
    """Install HF config mapping + JAX/vLLM model registration."""
    from .hf_config import install_hf_config

    install_hf_config()
    _guard_qwix_null_quantization_config()
    try:
        from . import register

        return register()
    except Exception as e:  # noqa: BLE001 - startup must not break interpreters
        import warnings

        warnings.warn(f"qwen4_exp startup: model register() skipped: {e}")
        return {}


install()

__all__ = ["install"]
