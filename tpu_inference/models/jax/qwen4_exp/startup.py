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


def install() -> dict:
    """Install HF config mapping + JAX/vLLM model registration."""
    from .hf_config import install_hf_config

    install_hf_config()
    try:
        from . import register

        return register()
    except Exception as e:  # noqa: BLE001 - startup must not break interpreters
        import warnings

        warnings.warn(f"qwen4_exp startup: model register() skipped: {e}")
        return {}


install()

__all__ = ["install"]
