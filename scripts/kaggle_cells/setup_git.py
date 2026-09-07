import subprocess, sys
from pathlib import Path

# Code delivery via git (replaces the old base64-embedded zip):
# https://github.com/DQN-Labs/nexus-tpu-fork (public, Apache-2.0).
REPO_URL = "https://github.com/DQN-Labs/nexus-tpu-fork.git"
# NOTE: checkout dir intentionally matches the old embedded-blob layout so
# downstream cells (pytest path, imports) work unchanged.
ROOT = Path("/kaggle/working/qwen4exp-fork")
if not (ROOT / "pyproject.toml").exists():
    subprocess.check_call(
        ["git", "clone", "--depth", "1", REPO_URL, str(ROOT)])
else:
    print("repo already present, refreshing to origin/master")
    subprocess.check_call(["git", "-C", str(ROOT), "fetch", "--depth", "1",
                           "origin", "master"])
    subprocess.check_call(["git", "-C", str(ROOT), "reset", "--hard",
                           "origin/master"])
sha = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
print("fork commit:", sha)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pytest", "flax"])
import jax
print("jax", jax.__version__, "| default_backend:", jax.default_backend())
print("devices:", jax.devices())
print("extracted:", sorted(p.name for p in
      (ROOT / "tpu_inference" / "models" / "jax" / "qwen4_exp").glob("*.py")))

# Fork modules shard eagerly (flax with_partitioning) and require an active
# device mesh at construction - same rule as tpu-inference's own models.
import numpy as _np
from jax.sharding import Mesh
def mesh_ctx():
    m = Mesh(_np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1)),
             axis_names=("data", "attn_dp", "expert", "model"))
    return jax.set_mesh(m) if hasattr(jax, "set_mesh") else m
