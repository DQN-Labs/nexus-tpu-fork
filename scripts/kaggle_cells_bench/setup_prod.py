import subprocess, sys, time
from pathlib import Path

# Idempotent setup: probe first, install only what is missing or
# version-wrong. Safe on both cold VMs (installs everything) and warm VMs
# with Files persistence (near-instant no-op).
W = Path("/kaggle/working")

def _dist_version(name):
    import importlib.metadata
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def _importable(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None

PINNED = {"tpu-inference": "0.29.0", "vllm": "0.28.0", "torchaudio": "2.10.0",
          "lark": "1.2.2"}
VLLM_RUNTIME_DEPS = [
    "fastapi", "uvicorn", "openai", "pydantic", "tiktoken", "sentencepiece",
    "safetensors", "tokenizers", "einops", "cloudpickle", "msgspec", "pyzmq",
    "setproctitle", "psutil", "pillow", "protobuf", "python-json-logger",
    "prometheus_client", "blake3", "py-cpuinfo", "partial-json-parser",
    "jsonschema", "filelock", "pyyaml",
    "openai-harmony", "anthropic", "model-hosting-container-standards", "mcp",
    "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp",
    "opentelemetry-semantic-conventions-ai", "ninja", "cachetools", "cbor2",
    "ijson", "pybase64", "compressed-tensors==0.17.0", "fastsafetensors",
    "outlines_core==0.2.14", "lm-format-enforcer==0.11.3", "xgrammar",
    "llguidance", "mistral_common", "depyf",
    "prometheus-fastapi-instrumentator",
    "lark==1.2.2", "huggingface_hub>=1.27.0",
]
MODULE_DEPS = ["fastapi", "uvicorn", "openai", "pydantic", "tiktoken",
               "sentencepiece", "safetensors", "tokenizers", "einops",
               "requests", "huggingface_hub",
               "openai_harmony", "anthropic", "mcp", "ninja", "cachetools",
               "cbor2", "ijson", "pybase64", "xgrammar", "llguidance",
               "mistral_common", "depyf", "lark", "torchaudio"]

PY = sys.executable

def stage(name, cmd, timeout_s=1500):
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}", flush=True)
    subprocess.check_call(cmd, timeout=timeout_s)
    dt = time.time() - t0
    (W / f"stage_{name}.txt").write_text(f"ok in {dt:.0f}s")
    print(f"[{time.strftime('%H:%M:%S')}] DONE {name} in {dt:.0f}s", flush=True)

problems = []
for dist, want in PINNED.items():
    have = _dist_version(dist)
    if have != want:
        problems.append(f"{dist}: have {have}, want {want}")
for mod in MODULE_DEPS:
    if not _importable(mod):
        problems.append(f"module missing: {mod}")
if problems:
    print(f"{len(problems)} env gaps -> installing (vllm with --no-deps, "
          f"see comments for why):", flush=True)
    for p in problems[:20]:
        print("  -", p, flush=True)
    stage("tpu_inference", [PY, "-m", "pip", "install", "tpu-inference",
                            "huggingface_hub", "requests"])
    stage("vllm_nodeps", [PY, "-m", "pip", "install", "--no-deps",
                          "vllm==0.28.0"])
    stage("vllm_runtime_deps", [PY, "-m", "pip", "install",
                                *VLLM_RUNTIME_DEPS, "torchaudio==2.10.0"])
else:
    print("env already complete (persistence hit) - skipping pip installs")
    (W / "stage_setup_skipped.txt").write_text("env complete, nothing to do")

import jax, importlib.metadata
print("jax", jax.__version__)
# NOTE: never jax.devices() in-process (exclusive TPU ownership); gate in a
# subprocess that releases the chips on exit.
gate = subprocess.run(
    [PY, "-c",
     "import jax; ds = jax.devices(); print('devices:', ds); "
     "assert any(d.platform == 'tpu' for d in ds), 'FAIL: no TPU visible'"],
    capture_output=True, text=True, timeout=600)
print(gate.stdout[-1500:])
print(gate.stderr[-500:])
assert gate.returncode == 0, "FAIL: TPU gate subprocess failed"
print("tpu_inference", importlib.metadata.version("tpu_inference"))
import vllm
print("vllm", vllm.__version__)
import vllm.entrypoints.cli.main
import torchaudio
print("vllm CLI + torchaudio imports OK")
(W / "stage_setup_done.txt").write_text("ok")
print("SETUP OK")
