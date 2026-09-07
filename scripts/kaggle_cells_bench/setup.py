import subprocess, sys, time
from pathlib import Path

# DO NOT CANCEL: this cell downloads ~3GB (torch/JAX/vLLM wheels) and takes
# 15-30 min on a fresh VM. Progress streams below; stage marker files land in
# /kaggle/working/ as each step completes. Silence != stuck.
W = Path("/kaggle/working")

def stage(name, cmd, timeout_s=1500):
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}: {' '.join(cmd)}", flush=True)
    try:
        subprocess.check_call(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FAIL: stage '{name}' exceeded {timeout_s}s")
    dt = time.time() - t0
    (W / f"stage_{name}.txt").write_text(f"ok in {dt:.0f}s")
    print(f"[{time.strftime('%H:%M:%S')}] DONE {name} in {dt:.0f}s", flush=True)

PY = sys.executable
# NOTE: tpu-inference does not pull vLLM itself; vllm==0.28.0 installs with
# --no-deps from its cp38-abi3 wheel because a full resolve backtracks into
# a CUDA-only source build (numba/transformers pin conflicts). The JAX
# serving path only needs vllm's orchestration plus small pure-Python deps.
stage("tpu_inference", [PY, "-m", "pip", "install", "tpu-inference",
                        "huggingface_hub", "requests"])
stage("vllm_nodeps", [PY, "-m", "pip", "install", "--no-deps", "vllm==0.28.0"])
stage("vllm_runtime_deps", [PY, "-m", "pip", "install",
    # server + sampling orchestration
    "fastapi", "uvicorn", "openai", "pydantic", "tiktoken", "sentencepiece",
    "safetensors", "tokenizers", "einops", "cloudpickle", "msgspec", "pyzmq",
    "setproctitle", "psutil", "pillow", "protobuf", "python-json-logger",
    "prometheus_client", "blake3", "py-cpuinfo", "partial-json-parser",
    "jsonschema", "filelock", "pyyaml",
    # vllm hard imports (ModuleNotFoundError at CLI startup without these;
    # diagnosed 2026-09-05 from vllm_server.log: openai_harmony first)
    "openai-harmony", "anthropic", "model-hosting-container-standards", "mcp",
    "opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp",
    "opentelemetry-semantic-conventions-ai", "ninja", "cachetools", "cbor2",
    "ijson", "pybase64", "compressed-tensors==0.17.0", "fastsafetensors",
    "outlines_core==0.2.14", "lm-format-enforcer==0.11.3", "xgrammar",
    "llguidance", "mistral_common", "depyf",
    "prometheus-fastapi-instrumentator",
    # version adjustments vllm 0.28.0 metadata demands (image ships others)
    "lark==1.2.2", "huggingface_hub>=1.27.0",
    # torch-ecosystem skew (diagnosed 2026-09-05 from vllm_server.log):
    # tpu-inference upgrades torch 2.8->2.10, which breaks the image's
    # torchaudio 2.8 native lib (undefined Symbol sym_ne), and transformers
    # 5.x imports torchaudio unconditionally (loss_rnnt). Pin the trio to
    # the mutually consistent 2.10 set.
    "torchaudio==2.10.0"])
    # NOTE: deliberately NOT installed: torch==2.13.0 (image torch works),
    # numba==0.65.0 / setuptools<81 (metadata-only so far), CUDA-only
    # kernels (flashinfer/tilelang/cutlass/quack/tokenspeed/humming/
    # torchcodec/PyNvVideoCodec/tvm-ffi). Added on demand if a traceback
    # names them; the JAX/TPU path never executes them.

import jax, importlib.metadata
print("jax", jax.__version__)
# NOTE: never call jax.devices()/jax.distributed here: initializing the TPU
# client in the notebook process takes EXCLUSIVE ownership of the chips, and
# the vLLM server subprocess then dies with "Device or resource busy"
# (diagnosed 2026-09-05 from vllm_server.log). The TPU gate runs in a
# short-lived subprocess instead, which releases the TPU on exit.
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
import torch
print("torch", torch.__version__)
# Smoke-test the exact CLI import chain that failed before (openai_harmony,
# then torchaudio via transformers loss_rnnt).
import vllm.entrypoints.cli.main
import torchaudio
print("vllm CLI + torchaudio imports OK")
(W / "stage_setup_done.txt").write_text("ok")
print("SETUP OK")
