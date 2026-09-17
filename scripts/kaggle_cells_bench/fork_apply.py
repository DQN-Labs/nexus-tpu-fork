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
# Overlay install: our tree ships only the leaf package
# (tpu_inference/models/jax/qwen4_exp) with no intermediate __init__.py, so
# it is invisible next to the regular tpu_inference already installed in
# site-packages (namespace portions do not merge into it). Copy the leaf
# into the installed tree instead - same approach as
# scripts/setup_kaggle_tpu_v5e.sh. Diagnosed 2026-09-09 from
# "ModuleNotFoundError: No module named 'tpu_inference.models.jax.qwen4_exp'".
import shutil
import tpu_inference
base = Path(tpu_inference.__file__).parent
src = DST / "tpu_inference" / "models" / "jax" / "qwen4_exp"
dst = base / "models" / "jax" / "qwen4_exp"
print(f"overlay: {src} -> {dst}")
shutil.copytree(src, dst, dirs_exist_ok=True)
import importlib
importlib.invalidate_caches()
if str(DST) not in sys.path:
    sys.path.insert(0, str(DST))
from tpu_inference.models.jax.qwen4_exp import register
reg = register()
print("registered architectures:", sorted(reg))
# Server-subprocess reach: vLLM's ModelConfig parses the checkpoint with
# transformers BEFORE consulting any model registry, and stock transformers
# (even 5.12.1) does not know model_type qwen4_exp (v44 died here). Install
# our AutoConfig shim in-process, then drop a .pth so EVERY fresh interpreter
# - notably the `vllm serve` subprocess - self-installs config + registry
# before ModelConfig runs.
from tpu_inference.models.jax.qwen4_exp.startup import install as _srv_install
_reg2 = _srv_install()
print("startup install ok:", sorted(_reg2) if _reg2 else "registry deferred")
import sysconfig
_purelib = Path(sysconfig.get_paths()["purelib"])
_pth = _purelib / "qwen4exp_tpu_startup.pth"
_pth.write_text("import tpu_inference.models.jax.qwen4_exp.startup\n")
print("wrote", _pth)
# In-process proof that transformers now parses a qwen4_exp checkpoint
# (synthetic config.json; the real one is inspected in the next cells).
import json as _json, tempfile as _tf
with _tf.TemporaryDirectory() as _d:
    Path(_d, "config.json").write_text(_json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560, "num_hidden_layers": 48,
        "text_config": {"model_type": "qwen4_exp", "hidden_size": 2560,
                        "num_hidden_layers": 48, "hc_count": 4}}))
    from transformers import AutoConfig as _AC
    _c = _AC.from_pretrained(_d)
    assert type(_c).__name__ == "Qwen4ExpConfig", type(_c).__name__
    assert _c.architectures == ["Qwen4ExpForCausalLM"]
    assert _c.text_config.hidden_size == 2560
    print("AutoConfig parses qwen4_exp: OK")
Path("/kaggle/working/fork_applied.json").write_text(__import__("json").dumps(
    {"commit": sha, "registered": sorted(reg)}))
print("FORK APPLIED")
