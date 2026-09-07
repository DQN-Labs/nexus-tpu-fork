import jax, json
devs = jax.devices()
tpus = [d for d in devs if d.platform == "tpu"]
print(json.dumps({"devices": [str(d) for d in devs], "tpu_count": len(tpus)}, indent=1))
assert tpus, "FAIL: no TPU devices visible - refusing to silently fall back to CPU (fork task rule)"
if len(tpus) != 8:
    print(f"WARNING: v5e-8 should expose 8 TPU devices, saw {len(tpus)}")
else:
    print("TPU v5e-8 detected (8 devices).")
