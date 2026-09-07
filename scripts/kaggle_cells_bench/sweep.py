import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

BASE = "http://localhost:8000"
MODEL = requests.get(f"{BASE}/v1/models", timeout=60).json()["data"][0]["id"]
print("serving model id:", MODEL)

UNIT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "

def make_prompt(n_tokens):
    return UNIT * max(1, n_tokens // 9)  # actual length read back from usage

def run_once(prompt_text, max_tokens, timeout=3600):
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/completions",
                      json={"model": MODEL, "prompt": prompt_text,
                            "max_tokens": max_tokens, "temperature": 0.0,
                            "stream": True,
                            "stream_options": {"include_usage": True}},
                      timeout=timeout, stream=True)
    r.raise_for_status()
    t_first, usage = None, {}
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        elif t_first is None and obj.get("choices"):
            t_first = time.time()
    t_end = time.time()
    pt = usage.get("prompt_tokens", -1)
    ct = usage.get("completion_tokens", -1)
    ttft = (t_first or t_end) - t0
    dec_s = max(t_end - (t_first or t_end), 1e-9)
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "ttft_s": round(ttft, 3), "total_s": round(t_end - t0, 3),
            "prefill_tok_per_s": round(pt / max(ttft, 1e-9), 1),
            "decode_tok_per_s": round(ct / dec_s, 1)}

print("warmup (absorbs XLA compile, excluded)...", flush=True)
w = run_once(make_prompt(512), 16)
print("warmup done:", w)

results = {"model_id": MODEL, "points": []}
for n in (1024, 4096, 16384):
    print(f"prefill ~{n} x decode 128 ...", flush=True)
    m = run_once(make_prompt(n), 128)
    m["label"] = f"prefill~{n}_decode128_batch1"
    print(m)
    results["points"].append(m)

print("batch-4 x prefill ~2048 x decode 64 ...", flush=True)
prompts = [make_prompt(2048) for _ in range(4)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    outs = list(ex.map(lambda p: run_once(p, 64), prompts))
wall = time.time() - t0
tot_ct = sum(o["completion_tokens"] for o in outs)
batch = {"label": "prefill~2048_decode64_batch4",
         "wall_s": round(wall, 3),
         "aggregate_decode_tok_per_s": round(tot_ct / max(wall, 1e-9), 1),
         "per_request": outs}
print(json.dumps(batch, indent=1)[:1500])
results["points"].append(batch)

try:
    tail = Path("/kaggle/working/vllm_server.log").read_text().splitlines()[-40:]
    results["server_log_tail"] = tail
except Exception as e:
    results["server_log_tail"] = [f"<unavailable: {e}>"]

Path("/kaggle/working/sweep_results.json").write_text(json.dumps(results, indent=1))
print("SWEEP COMPLETE")
print(json.dumps([{k: p[k] for k in ("label", "prefill_tok_per_s", "decode_tok_per_s") if k in p}
                  for p in results["points"] if "label" in p and "prefill_tok_per_s" in p], indent=1))
