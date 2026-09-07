import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

info = json.loads(Path("/kaggle/working/weight_choice.json").read_text())
model_path = info["path"]
log = open("/kaggle/working/vllm_server.log", "w")
env = dict(os.environ, TPU_BACKEND_TYPE="jax")
cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model_path,
       "--tensor-parallel-size", "8", "--max-model-len", "20480",
       "--max-num-seqs", "8", "--port", "8000"]
print(" ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
t0 = time.time()
healthy = False
for _ in range(240):  # up to ~40 min for install-cold startup
    time.sleep(10)
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
            if r.status == 200:
                healthy = True
                break
    except Exception:
        pass
    if proc.poll() is not None:
        break
startup_s = time.time() - t0
print(f"healthy={healthy} startup_s={startup_s:.0f} returncode={proc.poll()}")
# The server log is the only record of a startup crash: always surface its
# tail to cell stdout AND persist it (outputs survive even when cancelled).
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
    {"startup_s": startup_s, "model": info["model"], "path": model_path,
     "healthy": healthy, "returncode": proc.poll(),
     "server_log_tail": tail[-60:]}))
assert healthy, "FAIL: vLLM server did not become healthy (see tail above)"
print("SERVER UP")
