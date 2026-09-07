#!/bin/bash
# Kaggle TPU v5e-8 setup for Qwen4Exp fork (deliverable 9).
# Fails loudly on errors (task §12: never silently fall back).
set -euxo pipefail

# 1. System deps
pip install --upgrade pip

# 2. Base tpu-inference at the pinned commit (see README for hash)
if [ ! -d tpu-inference ]; then
  git clone --depth 1 https://github.com/vllm-project/tpu-inference
fi

# 3. Overlay this fork (assumes this script runs from the fork root)
FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp -r "${FORK_ROOT}/tpu_inference/models/jax/qwen4_exp" tpu-inference/tpu_inference/models/jax/
# In-tree registration: apply the patch, or fail loudly if it drifts.
if patch -p1 --dry-run -d tpu-inference < "${FORK_ROOT}/patches/model_loader.patch"; then
  patch -p1 -d tpu-inference < "${FORK_ROOT}/patches/model_loader.patch"
else
  echo "ERROR: patches/model_loader.patch does not apply to this tpu-inference checkout." >&2
  echo "Update the patch to match upstream, or register out-of-tree via qwen4_exp.register()." >&2
  exit 1
fi

# 4. Install (TPU extras)
pip install "jax[tpu]==0.11.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ./tpu-inference

# 5. Smoke test (CPU parts; TPU e2e needs the v5e-8 runtime)
python -m pytest "${FORK_ROOT}/tests/models/jax/test_qwen4_exp.py" -q

echo "Setup complete. Serve with:"
echo "  TPU_BACKEND_TYPE=jax python -m vllm.entrypoints.cli.main serve <QWEN_MODEL> --tensor-parallel-size 8 --max-model-len <context>"
