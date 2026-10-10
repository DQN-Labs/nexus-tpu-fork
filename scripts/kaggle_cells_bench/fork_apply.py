import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Apply the fork we built: clone + register its Qwen4Exp model into the
# tpu-inference loader, so vLLM resolves Qwen4ExpForCausalLM to OUR JAX
# implementation instead of failing on unknown architecture.
#
# Hardened 2026-10-0?: a run died here with no artifact and no visible log
# (only setup stage files survived), so every failure mode below now
# (a) retries transient git/network flakes, (b) falls back to a fresh
# clone when a persisted tree is corrupt, (c) writes fork_error.json with
# the full traceback + environment fingerprints before re-raising, so the
# cause is diagnosable from session outputs alone.
FORK_URL = "https://github.com/DQN-Labs/nexus-tpu-fork.git"
DST = Path("/kaggle/working/nexus-tpu-fork")
W = Path("/kaggle/working")


def _run(cmd, timeout_s, attempts=3):
    last = None
    for a in range(attempts):
        try:
            print(f"$ {' '.join(cmd)} (try {a + 1}/{attempts})", flush=True)
            subprocess.check_call(cmd, timeout=timeout_s)
            return
        except Exception as e:  # noqa: BLE001 - retry, then raise
            last = e
            print(f"  failed: {type(e).__name__}: {str(e)[:300]}",
                  flush=True)
            time.sleep(10 * (a + 1))
    raise last


def _fresh_clone():
    if DST.exists():
        print(f"removing stale tree {DST}", flush=True)
        shutil.rmtree(DST, ignore_errors=True)
    print(f"cloning {FORK_URL} ...", flush=True)
    _run(["git", "clone", "--depth", "1", FORK_URL, str(DST)],
         timeout_s=900)


try:
    # Preflight fingerprints (diagnose Kaggle runtime drift: proxy env,
    # git version, disk, and a cheap ls-remote canary before the clone).
    print("git:", subprocess.check_output(
        ["git", "--version"], text=True, timeout=60).strip(), flush=True)
    print("proxy env:", {k: v for k, v in os.environ.items()
                         if "proxy" in k.lower()}, flush=True)
    try:
        du = shutil.disk_usage(W)
        print(f"disk free: {du.free / 1e9:.1f} GB", flush=True)
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"disk probe skipped: {e}", flush=True)
    try:
        head = subprocess.check_output(
            ["git", "ls-remote", FORK_URL, "HEAD"], text=True,
            timeout=120).strip()
        print(f"ls-remote HEAD: {head[:80]}", flush=True)
    except Exception as e:  # noqa: BLE001 - clone attempt still follows
        print(f"ls-remote canary failed (clone may still work): "
              f"{type(e).__name__}: {str(e)[:300]}", flush=True)
    if not (DST / "pyproject.toml").exists():
        _fresh_clone()
    else:
        print("fork present, refreshing to origin/master", flush=True)
        try:
            _run(["git", "-C", str(DST), "fetch", "--depth", "1",
                  "origin", "master"], timeout_s=600)
            _run(["git", "-C", str(DST), "reset", "--hard",
                  "origin/master"], timeout_s=300)
        except Exception:
            print("refresh failed; falling back to fresh clone", flush=True)
            _fresh_clone()
    sha = subprocess.check_output(
        ["git", "-C", str(DST), "rev-parse", "HEAD"], text=True,
        timeout=60).strip()
    print("fork commit:", sha)
    if not (DST / "tpu_inference" / "models" / "jax" / "qwen4_exp").is_dir():
        raise RuntimeError(
            f"clone missing qwen4_exp leaf (tree corrupt?): "
            f"{sorted(p.name for p in DST.iterdir())}")
except Exception:
    (W / "fork_error.json").write_text(__import__("json").dumps(
        {"stage": "git", "error": traceback.format_exc()[-6000:]}, indent=1))
    print("FORK-APPLY git stage failed; wrote fork_error.json", flush=True)
    raise
# Overlay install: our tree ships only the leaf package
# (tpu_inference/models/jax/qwen4_exp) with no intermediate __init__.py, so
# it is invisible next to the regular tpu_inference already installed in
# site-packages (namespace portions do not merge into it). Copy the leaf
# into the installed tree instead - same approach as
# scripts/setup_kaggle_tpu_v5e.sh. Diagnosed 2026-09-09 from
# "ModuleNotFoundError: No module named 'tpu_inference.models.jax.qwen4_exp'".
import tpu_inference
try:
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
    # transformers BEFORE consulting any model registry, and stock
    # transformers (even 5.12.1) does not know model_type qwen4_exp (v44
    # died here). Install our AutoConfig shim in-process, then drop a .pth
    # so EVERY fresh interpreter - notably the `vllm serve` subprocess -
    # self-installs config + registry before ModelConfig runs.
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
except Exception:
    (W / "fork_error.json").write_text(__import__("json").dumps(
        {"stage": "overlay-register", "commit": sha,
         "error": traceback.format_exc()[-6000:]}, indent=1))
    print("FORK-APPLY overlay/register stage failed; wrote fork_error.json",
          flush=True)
    raise
