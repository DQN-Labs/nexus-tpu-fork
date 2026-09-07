import base64, io, sys, zipfile
from pathlib import Path
BLOB = """__BLOB__"""
ROOT = Path("/kaggle/working/qwen4exp-fork")
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))) as z:
    z.extractall(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytest", "flax"])
import jax
print("jax", jax.__version__, "| default_backend:", jax.default_backend())
print("devices:", jax.devices())
print("extracted:", sorted(p.name for p in (ROOT / "tpu_inference" / "models" / "jax" / "qwen4_exp").glob("*.py")))

# Fork modules shard eagerly (flax with_partitioning) and require an active
# device mesh at construction - same rule as tpu-inference's own models.
import numpy as _np
from jax.sharding import Mesh
def mesh_ctx():
    m = Mesh(_np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1)),
             axis_names=("data", "attn_dp", "expert", "model"))
    return jax.set_mesh(m) if hasattr(jax, "set_mesh") else m
