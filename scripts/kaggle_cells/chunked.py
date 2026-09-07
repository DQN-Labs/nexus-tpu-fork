import json, time
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from pathlib import Path
from tpu_inference.models.jax.qwen4_exp.config import Qwen4ExpArch
from tpu_inference.models.jax.qwen4_exp.layers import Qwen4ExpDecoderLayer

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
                                rngs=nnx.Rngs(7), prefix="model.layers.0")
    T, SPLIT = 12, 7
    hidden = (jnp.arange(T * 64, dtype=jnp.float32).reshape(T, 64) - T * 32.0) / 64.0
    pos = jnp.arange(T, dtype=jnp.int32)
    _, o_full, _ = layer.forward(hidden, None, None, pos, kv_history=make_hist())
    jax.block_until_ready(o_full)
    hist = make_hist()
    t0 = time.time()
    _, o_a, _ = layer.forward(hidden[:SPLIT], None, None, pos[:SPLIT], kv_history=hist)
    jax.block_until_ready(o_a)
    t_prefill = time.time() - t0
    t0 = time.time()
    _, o_b, _ = layer.forward(hidden[SPLIT:], None, None, pos[SPLIT:], kv_history=hist)
    jax.block_until_ready(o_b)
    t_decode = time.time() - t0
np.testing.assert_allclose(np.asarray(jax.device_get(o_a)),
                           np.asarray(jax.device_get(o_full[:SPLIT])), rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(np.asarray(jax.device_get(o_b)),
                           np.asarray(jax.device_get(o_full[SPLIT:])), rtol=1e-4, atol=1e-4)
res = {"pytest_exit_code": globals().get("PYTEST_CODE"),
       "tpu_proof": globals().get("TPU_PROOF_RES"),
       "prefill_tokens": SPLIT, "prefill_s": t_prefill,
       "prefill_tok_per_s": SPLIT / max(t_prefill, 1e-9),
       "decode_tokens": T - SPLIT, "decode_s": t_decode,
       "decode_tok_per_s": (T - SPLIT) / max(t_decode, 1e-9),
       "chunked_matches_full": True,
       "devices": [str(d) for d in jax.devices()]}
Path("/kaggle/working/qwen4exp_tpu_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print("PREFILL->DECODE CONSISTENCY OK on TPU")
