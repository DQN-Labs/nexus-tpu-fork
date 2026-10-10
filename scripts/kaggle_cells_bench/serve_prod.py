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
#
# KV ladder (v91): the full model loads (1306/1306) but max_model_len=32768
# needs 12.0 GiB KV vs 10.21 GiB available (v90 died in
# _check_enough_kv_cache_memory; engine's own estimate: 27856 max). Try
# 24576 first (under the estimate), then fall back to 16384 / 8192 on the
# KV ValueError. Each attempt costs one ~7-min load; worst case ~25 min,
# well inside the 12h session.
MAX_MODEL_LEN_LADDER = [24576, 16384, 8192]
MAX_NUM_SEQS = 16
KV_SHORTAGE_MARK = "larger than the available KV cache memory"


def try_serve(model_path, max_model_len):
    log = open("/kaggle/working/vllm_server.log", "w")
    env = dict(os.environ, TPU_BACKEND_TYPE="jax")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
           model_path,
           "--tensor-parallel-size", "8", "--max-model-len", str(max_model_len),
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
    print(f"max_model_len={max_model_len} healthy={healthy} "
          f"startup_s={startup_s:.0f} returncode={proc.poll()}")
    try:
        log.flush()
        tail = Path("/kaggle/working/vllm_server.log").read_text().splitlines()
        print(f"--- vllm_server.log tail ({len(tail)} lines) ---")
        for line in tail[-60:]:
            print(line[:500])
    except Exception as e:
        tail = [f"<log unreadable: {e}>"]
        print(tail[0])
    if not healthy and proc.poll() is None:
        # Dead engine on a failed attempt: reap it before the next rung.
        # On success the server MUST stay up for the prompts cell.
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except Exception:
            proc.kill()
    if not healthy:
        # Best-effort: orphaned engine children can hold :8000 / TPU
        # state across rungs; this VM runs nothing else.
        try:
            subprocess.run(["pkill", "-f", "vllm.entrypoints"],
                           timeout=60)
            time.sleep(15)
        except Exception as e:  # noqa: BLE001 - informational only
            print(f"pkill skipped: {e}")
    kv_shortage = any(KV_SHORTAGE_MARK in line for line in tail)
    return healthy, startup_s, proc.poll(), tail, kv_shortage


info = json.loads(Path("/kaggle/working/model_path.json").read_text())
if not info.get("path"):
    msg = "NO PATH FOUND - skipping serve (provide the GPTQ export, then rerun)."
    print(msg)
    Path("/kaggle/working/server_info.json").write_text(json.dumps(
        {"healthy": False, "reason": "NO PATH FOUND"}))
else:
    model_path = info["path"]
    attempts = []
    healthy = False
    final_len = None
    for mml in MAX_MODEL_LEN_LADDER:
        ok, startup_s, rc, tail, kv_shortage = try_serve(model_path, mml)
        attempts.append({"max_model_len": mml, "healthy": ok,
                         "startup_s": startup_s, "returncode": rc,
                         "kv_shortage": kv_shortage})
        if ok:
            healthy = True
            final_len = mml
            break
        if not kv_shortage:
            print(f"max_model_len={mml} failed WITHOUT kv-shortage marker; "
                  f"not retrying ladder (see tail above)")
            break
        print(f"max_model_len={mml} hit KV shortage; trying next rung")
    Path("/kaggle/working/server_info.json").write_text(json.dumps(
        {"startup_s": attempts[-1]["startup_s"] if attempts else 0,
         "path": model_path, "quant": "jax-side-dequant",
         "max_model_len": final_len, "healthy": healthy,
         "returncode": attempts[-1]["returncode"] if attempts else None,
         "attempts": [{k: v for k, v in a.items() if k != "tail"}
                      for a in attempts],
         "server_log_tail": tail[-60:] if attempts else []}))
    assert healthy, "FAIL: server did not become healthy (see tail above)"
    print(f"SERVER UP at max_model_len={final_len}")
