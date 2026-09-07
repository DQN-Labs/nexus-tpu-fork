import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

tok = os.environ.get("HF_TOKEN")  # Kaggle secret, if the owner attached one
cands = ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
choice, notes = None, []
api = HfApi(token=tok)
for repo in cands:
    try:
        info = api.model_info(repo)
        choice = repo
        notes.append(f"{repo}: accessible ({info.safetensors.total if info.safetensors else '?'} params)")
        print("selected:", notes[-1])
        break
    except Exception as e:
        notes.append(f"{repo}: skipped ({type(e).__name__}: {str(e)[:150]})")
        print(notes[-1])
assert choice, f"FAIL: no servable weights accessible. Probed: {notes}"
w = Path("/kaggle/working/weights") / choice.split("/")[-1]
w.mkdir(parents=True, exist_ok=True)
print(f"downloading ~16GB to {w} (10-20 min, do not cancel)...", flush=True)
snapshot_download(choice, local_dir=str(w), token=tok, max_workers=4,
                  ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"])
n_gb = sum(p.stat().st_size for p in w.rglob("*") if p.is_file()) / 1e9
print(f"weights: {choice} -> {w} ({n_gb:.1f} GB)")
Path("/kaggle/working/weight_choice.json").write_text(
    json.dumps({"model": choice, "path": str(w), "gb": n_gb, "probes": notes}))
