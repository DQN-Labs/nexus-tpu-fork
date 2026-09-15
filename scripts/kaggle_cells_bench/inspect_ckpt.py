import json
from pathlib import Path

# Inspect the REAL checkpoint with the FORK's tooling (no weight values
# loaded - index/headers only): parse config.json through the fork's arch
# adapter, then check every checkpoint tensor name against the fork's
# weight-name mapping. This is the weight-compatibility proof.
info = json.loads(Path("/kaggle/working/model_path.json").read_text())
rep = {"path": info.get("path")}
if not info.get("path"):
    rep["status"] = "NO PATH FOUND - skipping inspection"
else:
    import sys
    sys.path.insert(0, "/kaggle/working/nexus-tpu-fork")
    from tpu_inference.models.jax.qwen4_exp import config as C
    from tpu_inference.models.jax.qwen4_exp import weight_loader as W

    mp = Path(info["path"])
    cfg = json.loads((mp / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    arch = C.arch_from_hf_config(text, vocab_size=text.get("vocab_size"))
    rep["arch"] = {
        "hidden_size": arch.hidden_size, "layers": arch.num_hidden_layers,
        "heads": arch.num_attention_heads, "kv_heads": arch.num_key_value_heads,
        "head_dim": arch.head_dim, "experts": arch.num_experts,
        "experts_per_tok": arch.num_experts_per_tok,
        "hc_count": arch.hc_count, "layer_types": sorted(set(arch.layer_types)),
        "ple_layers": arch.ple_layer_ids, "use_qsa": arch.use_qsa,
        "vocab_size": arch.vocab_size,
    }
    print("arch parsed:", json.dumps(rep["arch"], indent=1)[:1200])

    # tensor names: prefer the index json (no file opens at all)
    names = []
    idx = mp / "model.safetensors.index.json"
    if idx.exists():
        names = sorted(json.loads(idx.read_text())["weight_map"].keys())
        rep["name_source"] = "index"
    else:
        from safetensors import safe_open
        rep["name_source"] = "headers"
        for shard in sorted(mp.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                names.extend(f.keys())
        names = sorted(set(names))
    rep["checkpoint_tensors"] = len(names)

    qsa_ids = frozenset(
        i for i, lt in enumerate(arch.layer_types)
        if lt == "full_attention" and arch.use_qsa)
    mapped, stacked, ignored, unknown = [], [], [], []
    for n in names:
        m = W.map_checkpoint_name(n, qsa_ids)
        st = W.stacked_target(m)
        if st:
            stacked.append([n, st[0]])
        elif W.is_ignored_missing(m):
            ignored.append(n)
        elif W.is_gptq_aux(m) or W.is_gdn_param(m):
            # GPTQ shards + GDN A_log/dt_bias: real params without ".weight"
            mapped.append([n, m])
        elif ".weight" in m or "weight" in n:
            mapped.append([n, m])
        else:
            unknown.append(n)
    rep["mapped"] = len(mapped)
    rep["stacked"] = len(stacked)
    rep["ignored_missing"] = len(ignored)
    rep["unknown"] = len(unknown)
    rep["unknown_sample"] = unknown[:20]
    rep["status"] = ("FULL COVERAGE" if not unknown
                     else f"GAPS: {len(unknown)} unmapped")
    print(f"tensors={len(names)} mapped={len(mapped)} stacked={len(stacked)} "
          f"ignored={len(ignored)} unknown={len(unknown)}")
    print("status:", rep["status"])

Path("/kaggle/working/inspect_results.json").write_text(json.dumps(rep, indent=1))
print("INSPECTION COMPLETE")
