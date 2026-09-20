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
    # Raw expert-dim keys (the JAX model is built from arch_from_hf_config;
    # a missing/renamed moe_intermediate_size would silently build wrong
    # expert shapes, so echo every candidate key verbatim).
    rep["moe_keys"] = {k: text.get(k) for k in text
                       if "inter" in k.lower() or "expert" in k.lower()
                       or k in ("hidden_size", "num_hidden_layers")}
    print("moe_keys:", json.dumps(rep["moe_keys"])[:600])

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

    # Structural histogram: per-submodule tensor inventory (names only) so
    # the JAX loader's rename/assembly table can be audited without moving
    # 187 GB. Key: "<layer-scope> | <submodule> | <kind>" where kind is the
    # last component (weight/qweight/...). Layer indices collapsed to #.
    import re as _re
    hist = {}
    for n in names:
        parts = n.split(".")
        scope = "top"
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                scope = "layers.#"
                rest = parts[i + 2:]
                break
        else:
            rest = parts[2:] if parts[:1] == ["model"] else parts
            if parts[:2] == ["model", "mtp"] or parts[:1] == ["mtp"]:
                scope = "mtp"
        kind = rest[-1] if rest else "?"
        mod = ".".join(rest[:-1]) if len(rest) > 1 else "(root)"
        mod = _re.sub(r"\.\d+\.", ".#.", "." + mod + ".").strip(".")
        key = f"{scope} | {mod} | {kind}"
        hist[key] = hist.get(key, 0) + 1
    rep["histogram"] = dict(sorted(hist.items()))
    print(f"histogram groups: {len(hist)}")
    # Per-group shape/dtype (first occurrence; headers only, no values).
    # This is what sizes the loader's fusions (qkv gate rows, expert dims,
    # table shards, mixer rows) without moving 100+ GB.
    try:
        import struct as _st
        if idx.exists():
            _wmap = json.loads(idx.read_text())["weight_map"]
            _paths = sorted({mp / _f for _f in _wmap.values()})
        else:
            _paths = sorted(mp.glob("*.safetensors"))
        shapes = {}
        for _shard in _paths:
            try:
                with open(str(_shard), "rb") as _fh:
                    _n = _st.unpack("<Q", _fh.read(8))[0]
                    _header = json.loads(_fh.read(_n))
                for _k, _info in _header.items():
                    if _k == "__metadata__":
                        continue
                    # RAW names (same keying as histogram above, no mapping).
                    _parts = _k.split(".")
                    _scope, _rest = "top", _parts
                    for _i, _p in enumerate(_parts):
                        if _p == "layers" and _i + 1 < len(_parts):
                            _scope = "layers.#"
                            _rest = _parts[_i + 2:]
                            break
                    _kind = _rest[-1] if _rest else "?"
                    _mod = ".".join(_rest[:-1]) if len(_rest) > 1 else "(root)"
                    _mod = _re.sub(r"\.\d+\.", ".#.", "." + _mod + ".").strip(".")
                    _gk = f"{_scope} | {_mod} | {_kind}"
                    if _gk not in shapes:
                        shapes[_gk] = [_info.get("shape"), _info.get("dtype")]
            except Exception as _e:
                print(f"shape scan skipped {_shard.name}: {_e}")
                continue
        rep["hist_shapes"] = dict(sorted(shapes.items()))
        print(f"hist_shapes groups: {len(shapes)}")
    except Exception as _e:
        rep["hist_shapes"] = {"error": str(_e)[:300]}
        print(f"hist_shapes failed: {_e}")
    # quantization_config echo (what the loader bypass neutralizes).
    rep["quantization_config"] = cfg.get("quantization_config",
                                         text.get("quantization_config", None))

Path("/kaggle/working/inspect_results.json").write_text(json.dumps(rep, indent=1))
print("INSPECTION COMPLETE")
