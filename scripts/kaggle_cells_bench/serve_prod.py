import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

# Pinned quant contract for the production export: GPTQ INT4, group_size
# 128, asymmetric (zeros kept), DescAct=False.
# NOTE (v48): do NOT pass --quantization gptq. vLLM's GPTQ path is CUDA-only
# and its ModelConfig gate rejects GPTQ on TPU outright ("auto_gptq
# quantization is currently not supported in tpu"). The fork serves GPTQ
# weights via JAX-side CPU dequant at load (weight_loader dequantizes
# qweight/qzeros/scales into float params); the checkpoint's
# quantization_config is neutralized in hf_config.py so vLLM never builds
# its GPTQ config from either trigger.
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 16

info = json.loads(Path("/kaggle/working/model_path.json").read_text())
if not info.get("path"):
    msg = "NO PATH FOUND - skipping serve (provide the GPTQ export, then rerun)."
    print(msg)
    Path("/kaggle/working/server_info.json").write_text(json.dumps(
        {"healthy": False, "reason": "NO PATH FOUND"}))
else:
    model_path = info["path"]
    log = open("/kaggle/working/vllm_server.log", "w")
    env = dict(os.environ, TPU_BACKEND_TYPE="jax")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
           model_path,
           "--tensor-parallel-size", "8", "--max-model-len", str(MAX_MODEL_LEN),
           "--max-num-seqs", str(MAX_NUM_SEQS), "--port", "8000",
           # Quantization bypass (v51): vLLM's get_config RE-ATTACHES
           # quantization_config from the raw config_dict (or the export's
           # hf_quant_config.json) AFTER AutoConfig parsing, defeating any
           # class-level strip, and then resolves its CUDA-only quant path
           # (auto_gptq gate / modelopt_fp4 JAX-universe gate). Our
           # hf_config shim strips it at parse, and this override NULLs it
           # after the re-attach (config.update runs last) — belt and
           # suspenders. Weights are served via JAX-side load-time dequant.
           "--hf-overrides", '{"quantization_config": null}']
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    healthy = False
    for _ in range(240):
        time.sleep(10)
        try:
            with urllib.request.urlopen("http://localhost:8000/health",
                                        timeout=5) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        if proc.poll() is not None:
            break
    startup_s = time.time() - t0
    print(f"healthy={healthy} startup_s={startup_s:.0f} "
          f"returncode={proc.poll()}")
    try:
        log.flush()
        tail = Path("/kaggle/working/vllm_server.log").read_text().splitlines()
        print(f"--- vllm_server.log tail ({len(tail)} lines) ---")
        for line in tail[-60:]:
            print(line[:500])
    except Exception as e:
        tail = [f"<log unreadable: {e}>"]
        print(tail[0])
    Path("/kaggle/working/server_info.json").write_text(json.dumps(
        {"startup_s": startup_s, "path": model_path, "quant": "gptq-jax-dequant",
         "max_model_len": MAX_MODEL_LEN, "healthy": healthy,
         "returncode": proc.poll(), "server_log_tail": tail[-60:]}))
    assert healthy, "FAIL: server did not become healthy (see tail above)"
    print("SERVER UP")
