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

"""Minimal Qwen4Exp example: register + offline inference.

Serving (normal vLLM path, task §9)::
    TPU_BACKEND_TYPE=jax python -m vllm.entrypoints.cli.main serve \\
        <QWEN_MODEL> --tensor-parallel-size 8 --max-model-len <context>

Out-of-tree registration (keeps fork clean): ``register()`` below calls
``tpu_inference.models.common.model_loader.register_model``.
"""

from tpu_inference.models.jax.qwen4_exp import register


def main(model: str = "Qwen/Qwen3.8-Flash-Next"):
    register()
    print(f"Registered Qwen4Exp JAX model. Serve with:\n"
          f"  TPU_BACKEND_TYPE=jax python -m vllm.entrypoints.cli.main serve "
          f"{model} --tensor-parallel-size 8")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-Flash-Next")
