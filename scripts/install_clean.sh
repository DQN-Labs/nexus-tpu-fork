#!/bin/bash
# Clean-environment install for the Qwen4Exp fork (deliverable 8).
# Fails loudly on errors (task §12).
set -euxo pipefail
python3 --version
pip install --upgrade pip

# Base backend first (CPU jax here; TPU runners install jax[tpu] instead).
pip install "jax[cpu]"

# Base tpu-inference checkout (text model registry + runner).
if [ ! -d tpu-inference ]; then
  git clone --depth 1 https://github.com/vllm-project/tpu-inference
fi
pip install -e ./tpu-inference

# Overlay the fork's JAX model + register it in-tree.
FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp -r "${FORK_ROOT}/tpu_inference/models/jax/qwen4_exp" tpu-inference/tpu_inference/models/jax/
patch -p1 -d tpu-inference < "${FORK_ROOT}/patches/model_loader.patch"

# Fork test deps + CPU unit tests (no TPU needed).
pip install -e "${FORK_ROOT}[test]"
python -m pytest "${FORK_ROOT}/tests/models/jax/test_qwen4_exp.py" -q
