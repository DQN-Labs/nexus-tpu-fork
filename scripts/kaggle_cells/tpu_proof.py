import time
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from tpu_inference.models.jax.qwen4_exp.config import Qwen4ExpArch
from tpu_inference.models.jax.qwen4_exp.layers import Qwen4ExpDecoderLayer

tpu_devs = [d for d in jax.devices() if d.platform == "tpu"]
cpu = jax.devices("cpu")[0]
arch = Qwen4ExpArch(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, intermediate_size=64, vocab_size=128,
    partial_rotary_factor=0.5, layer_types=["full_attention"], hc_count=2, hc_lowrank=8,
    ple_embed_dim=32, decoder_sparse_step=100, num_experts=0,
    indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=16,
    indexer_budget=16, indexer_compress_ratio=4)

def make_hist():
    return {"k": jnp.zeros((0, 2, 16), jnp.float32),
            "v": jnp.zeros((0, 2, 16), jnp.float32),
            "k_raw": jnp.zeros((0, 16), jnp.float32),
            "pos": jnp.zeros((0,), jnp.int32)}

layer = None
with mesh_ctx():
    layer = Qwen4ExpDecoderLayer(arch=arch, layer_idx=0, dtype=jnp.float32,
                                rngs=nnx.Rngs(0), prefix="model.layers.0")
    hidden = (jnp.arange(8 * 64, dtype=jnp.float32).reshape(8, 64) - 256.0) / 64.0
    pos = jnp.arange(8, dtype=jnp.int32)

    t0 = time.time()  # includes XLA compilation for these shapes
    h1, o1, _ = layer.forward(hidden, None, None, pos, kv_history=make_hist())
    jax.block_until_ready((h1, o1))
    t_compile = time.time() - t0
    t0 = time.time()
    h2, o2, _ = layer.forward(hidden, None, None, pos, kv_history=make_hist())
    jax.block_until_ready((h2, o2))
    t_steady = time.time() - t0
    with jax.default_device(cpu):
        hc, oc, _ = layer.forward(hidden, None, None, pos, kv_history=make_hist())
        jax.block_until_ready((hc, oc))
print(f"QSA-layer prefill: compile_s={t_compile:.2f} steady_s={t_steady:.3f} "
      f"tok_per_s={8 / max(t_steady, 1e-9):.1f}")
print("output devices:", o2.devices())
assert all(d.platform == "tpu" for d in o2.devices()), "FAIL: output not on TPU"
np.testing.assert_allclose(np.asarray(jax.device_get(o2)),
                           np.asarray(jax.device_get(oc)), rtol=1e-4, atol=1e-4)
print("CPU-TPU numeric parity OK (QSA layer)")
TPU_PROOF_RES = {"compile_s": t_compile, "steady_s": t_steady,
                 "prefill_tok_per_s": 8 / max(t_steady, 1e-9),
                 "output_on_tpu": True, "cpu_tpu_parity": True}
