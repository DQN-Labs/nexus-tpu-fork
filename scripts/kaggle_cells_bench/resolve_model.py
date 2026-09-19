import json, os
from pathlib import Path

# Model-path slot. Precedence: NEXUS_MODEL_PATH env -> MODEL_PATHS constants
# below (NVFP4 primary, GPTQ fallback) -> scan /kaggle/input for a directory
# containing *.safetensors.
# Primary export (NVFP4, ModelOpt FP4): keithtyser/qwen3-8-flash-next-nvfp4
# (pytorch/radixark-modelopt-fp4/1), plus tokenizer + config.json.
# Fallback export: GPTQ INT4, group_size 128, asymmetric (zeros kept),
# DescAct=False (ram2121/...-gptq-4bit).
MODEL_PATHS = [
    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",  # NVFP4 (primary)
    "/kaggle/input/models/ram2121/qwen3-8-flash-next-gptq-4bit/transformers/4bit/1",  # GPTQ (fallback)
]

found, checked = None, []
env_path = os.environ.get("NEXUS_MODEL_PATH", "").strip()
if env_path:
    checked.append(f"env:{env_path}")
    p = Path(env_path)
    if p.is_dir() and list(p.glob("*.safetensors")):
        found = str(p)
for const in MODEL_PATHS:
    if found is not None:
        break
    if not const.strip():
        continue
    checked.append(f"const:{const.strip()}")
    p = Path(const.strip())
    if p.is_dir() and list(p.glob("*.safetensors")):
        found = str(p)
if found is None:
    for cand in sorted(Path("/kaggle/input").glob("*")):
        if cand.is_dir() and list(cand.glob("*.safetensors")):
            checked.append(f"scan:{cand}")
            found = str(cand)
            break
        checked.append(f"scan:{cand} (no safetensors)")

if found is None:
    msg = ("NO PATH FOUND - no model provided yet. Checked: "
           + (", ".join(checked) if checked else "nothing (no env, no const, "
              "/kaggle/input empty or missing)"))
    print(msg)
    Path("/kaggle/working/model_path.json").write_text(json.dumps(
        {"path": None, "status": "NO PATH FOUND", "checked": checked}))
else:
    print(f"MODEL PATH: {found}")
    n = len(list(Path(found).glob("*.safetensors")))
    gb = sum(p.stat().st_size for p in Path(found).glob("*.safetensors")) / 1e9
    print(f"shards: {n}, ~{gb:.1f} GB safetensors")
    Path("/kaggle/working/model_path.json").write_text(json.dumps(
        {"path": found, "status": "FOUND", "shards": n, "gb": round(gb, 1)}))
