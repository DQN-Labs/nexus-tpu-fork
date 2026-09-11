import subprocess, sys
from pathlib import Path

# Apply the fork we built: clone + register its Qwen4Exp model into the
# tpu-inference loader, so vLLM resolves Qwen4ExpForCausalLM to OUR JAX
# implementation instead of failing on unknown architecture.
FORK_URL = "https://github.com/DQN-Labs/nexus-tpu-fork.git"
DST = Path("/kaggle/working/nexus-tpu-fork")
if not (DST / "pyproject.toml").exists():
    print(f"cloning {FORK_URL} ...", flush=True)
    subprocess.check_call(["git", "clone", "--depth", "1", FORK_URL, str(DST)])
else:
    print("fork present, refreshing to origin/master", flush=True)
    subprocess.check_call(["git", "-C", str(DST), "fetch", "--depth", "1",
                           "origin", "master"])
    subprocess.check_call(["git", "-C", str(DST), "reset", "--hard",
                           "origin/master"])
sha = subprocess.check_output(
    ["git", "-C", str(DST), "rev-parse", "HEAD"], text=True).strip()
print("fork commit:", sha)
if str(DST) not in sys.path:
    sys.path.insert(0, str(DST))
from tpu_inference.models.jax.qwen4_exp import register
reg = register()
print("registered architectures:", sorted(reg))
Path("/kaggle/working/fork_applied.json").write_text(__import__("json").dumps(
    {"commit": sha, "registered": sorted(reg)}))
print("FORK APPLIED")
