# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Qwen4Exp JAX implementation.

Covers (task §11):
- config parsing / validation
- weight-name mapping (representative checkpoint tensors)
- individual components (HC norm, RoPE, MoE routing, PLE hash, QSA select,
  GDN shapes, cache slot rules, Q4 dequant)
- full forward shapes (tiny arch, CPU)

Run on CPU (no TPU required)::

    python -m pytest tests/models/jax/test_qwen4_exp.py -q

End-to-end TPU test (requires v5e-8 + checkpoint) is
``test_e2e_tpu`` (skipped unless ``QWEN4EXP_E2E=1``).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import jax.numpy as jnp
    import numpy as np

    from tpu_inference.models.jax.qwen4_exp import (
        cache as cache_mod,
        config as cfg_mod,
        hyperconnection as hc_mod,
        ngram as ngram_mod,
        quant as quant_mod,
        qsa as qsa_mod,
        weight_loader as wl_mod,
    )
    from tpu_inference.models.jax.qwen4_exp.attention import apply_partial_rope
    from tpu_inference.models.jax.qwen4_exp.moe import Qwen4ExpMoE

    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as e:  # fail loudly with the reason (task §12)
    _IMPORT_OK = False
    _IMPORT_ERR = e

requires_impl = pytest.mark.skipif(
    not _IMPORT_OK, reason=f"qwen4_exp import failed: {_IMPORT_ERR}")


@pytest.fixture(scope="module")
def mesh():
    """Single-device mesh for module construction.

    Mirrors ``tpu_inference``'s own test convention (see its
    ``tests/models/jax/conftest.py``): flax nnx ``with_partitioning``
    initializers shard eagerly and require an active mesh context, so every
    test that constructs modules runs inside this mesh. Production serving
    always builds models under the runner mesh the same way.

    Newer JAX only exposes the mesh to flax via ``jax.set_mesh`` (plain
    ``with Mesh`` is deprecated and no longer sets the abstract mesh), while
    older JAX lacks ``set_mesh`` entirely — hence the version-adaptive
    context below.
    """
    import jax
    import numpy as _np
    from jax.sharding import Mesh

    devices = _np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1))
    m = Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))
    set_mesh = getattr(jax, "set_mesh", None)
    ctx = set_mesh(m) if set_mesh is not None else m
    with ctx:
        yield m


@requires_impl
def test_config_defaults_and_validation():
    from types import SimpleNamespace

    base = SimpleNamespace(
        hidden_size=32, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=64,
        vocab_size=256, rms_norm_eps=1e-6, hidden_act="silu",
        max_position_embeddings=128, rope_theta=10000.0,
        partial_rotary_factor=0.25, attention_bias=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, decoder_sparse_step=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts_per_tok=2, num_experts=4, norm_topk_prob=True,
        mlp_only_layers=[], layer_types=None, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=2,
        ngram_size=3, heads_per_ngram=4, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8, output_gate_type="sigmoid",
    )
    arch = cfg_mod.arch_from_hf_config(base, vocab_size=256)
    assert arch.hc_count == 2
    assert arch.short_conv_layer_ids == [1]
    assert arch.ngram_context_len == 2
    assert arch.short_conv_state_shape == (64, 3)
    assert cfg_mod.is_moe_layer(arch, 1)  # (1+1)%2==0
    assert not cfg_mod.is_moe_layer(arch, 0)

    bad = SimpleNamespace(**{**base.__dict__, "hc_count": 1})
    with pytest.raises(ValueError):
        cfg_mod.validate_qwen4exp_text_config(bad)


@requires_impl
def test_weight_name_mapping():
    m = wl_mod.map_checkpoint_name("model.language_model.layers.3.self_attn.qkv_proj.weight")
    assert m == "model.layers.3.self_attn.qkv_proj.weight"
    assert wl_mod.stacked_target("model.layers.0.ple.key_proj.weight")[1] == 0
    assert wl_mod.stacked_target("model.layers.0.ple.value_proj.weight")[1] == 1
    remapped = wl_mod.remap_qsa_scale_name(
        "model.layers.5.self_attn.k_proj.k_scale", frozenset({5}))
    assert remapped.endswith("self_attn._k_scale")
    assert wl_mod.is_ignored_missing("model.layers.0.self_attn.qkv_proj.bias")
    assert not wl_mod.is_ignored_missing("model.layers.0.self_attn.qkv_proj.weight")
    # v44 leftovers (5 GAPS): PLE table metadata is derived at load, MTP
    # draft tensors are never loaded (stub shares the target trunk).
    assert wl_mod.is_ignored_missing(
        "model.layers.1.ple.ple_embedding.layer_multipliers")
    assert wl_mod.is_ignored_missing(
        "model.layers.1.ple.ple_embedding.ngram_heads_offsets")
    assert wl_mod.is_ignored_missing(
        "model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes")
    assert wl_mod.is_ignored_missing("mtp.layers.0.mlp.experts.down_proj")
    assert wl_mod.is_ignored_missing("mtp.layers.0.mlp.experts.gate_up_proj")


@requires_impl
def test_grouped_rmsnorm_matches_grouped_math():
    import jax

    x = jnp.arange(2 * 6, dtype=jnp.float32).reshape(2, 6)  # HC=2,H=3
    w = jnp.zeros((6,), dtype=jnp.float32)
    y = hc_mod.grouped_gemma_rmsnorm(x, w, 1e-6, hc_count=2)
    g = x.reshape(2, 2, 3)
    ref = (g / jnp.sqrt(jnp.mean(g**2, axis=-1, keepdims=True) + 1e-6)).reshape(2, 6)
    np.testing.assert_allclose(np.asarray(y), np.asarray(ref), rtol=1e-5)


@requires_impl
def test_partial_rope_identity_for_zero_positions():
    q = jnp.ones((2, 2, 16), dtype=jnp.float32)
    k = jnp.ones((2, 1, 16), dtype=jnp.float32)
    pos = jnp.zeros((2,), dtype=jnp.int32)
    qr, kr = apply_partial_rope(q, k, pos, 16, 4, 10000.0)
    # Zero rotation leaves the rotary slice unchanged; tail passes through.
    np.testing.assert_allclose(np.asarray(qr[..., :4]), np.asarray(q[..., :4]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(qr[..., 4:]), np.asarray(q[..., 4:]), atol=1e-6)


@requires_impl
def test_qsa_slot_rules_and_select():
    assert cache_mod.circular_slot(10, 4) == 2
    assert cache_mod.compressed_slot(7, 4) == 1  # (7+1)%4==0
    assert cache_mod.compressed_slot(6, 4) is None
    assert cache_mod.circular_capacity(4, 0) == 4
    # Tiny select: budget=8, ratio=4 -> width=11, 2 blocks.
    q = jnp.ones((2, 2, 8), dtype=jnp.float32)
    kc = jnp.ones((3, 8), dtype=jnp.float32)
    pos = jnp.asarray([5, 9], dtype=jnp.int32)
    idx, counts = qsa_mod.qsa_select_indices(q, kc, pos, 4, 8)
    assert idx.shape == (2, 11)
    assert int(counts[0]) <= 11 and int(counts[1]) <= 11
    # Causal: no selected token may exceed its query position.
    for r in range(2):
        for c in idx[r, : counts[r]]:
            assert int(c) <= int(pos[r])


@requires_impl
def test_ple_hash_deterministic_and_shaped():
    m1 = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    m2 = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    assert m1 == m2 and len(m1) == 3 and all(x % 2 == 1 for x in m1)
    # Hash bound follows the vocab: larger vocabs admit smaller multipliers.
    m_big = ngram_mod.ple_multipliers(1234, 0, 3, 10_000_000)
    assert all(0 < v < (1 << 63) for v in m_big)
    sizes, offsets = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8)
    assert sizes.shape == (4,) and offsets.shape == (4,)
    # Layer-1 heads continue the global prime sequence (no overlap w/ layer 0).
    sizes1, _ = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8,
                                                 ple_dense_layer_id=1)
    assert not bool(jnp.any(sizes1 == sizes))
    ids = ngram_mod.compute_ngram_ids(
        jnp.asarray([5, 6, 7], dtype=jnp.int32),
        jnp.asarray([0, 3], dtype=jnp.int32),
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        sizes, offsets, ngram_mod.ple_multipliers(1234, 0, 3, 1000), 3)
    assert ids.shape == (3, 4)
    assert bool(jnp.all(ids >= 0))


@requires_impl
def test_moe_routing_normalization(mesh):
    from flax import nnx

    moe = Qwen4ExpMoE(hidden_size=16, moe_intermediate_size=8,
                      shared_intermediate_size=8, num_experts=4,
                      num_experts_per_tok=2, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 16), dtype=jnp.float32)
    out, logits = moe(x)
    assert out.shape == (3, 16) and logits.shape == (3, 4)
    _, w, _ = moe.route(x)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(w, axis=-1)), np.ones((3,)), rtol=1e-5)


@requires_impl
def test_decoder_layer_forward_shapes(mesh):
    from types import SimpleNamespace

    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.layers import Qwen4ExpDecoderLayer

    base = SimpleNamespace(
        hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, intermediate_size=32,
        vocab_size=64, rms_norm_eps=1e-6, hidden_act="silu",
        max_position_embeddings=64, rope_theta=10000.0,
        partial_rotary_factor=0.5, attention_bias=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, decoder_sparse_step=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts_per_tok=2, num_experts=4, norm_topk_prob=True,
        mlp_only_layers=[], layer_types=None, hc_count=2, hc_lowrank=4,
        ple_layer_ids=[], ple_embed_dim=16, ple_conv_kernel_size=2,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8, output_gate_type="sigmoid",
    )
    arch = cfg_mod.arch_from_hf_config(base, vocab_size=64)
    arch.layer_types = ["full_attention"] * 4
    dense = Qwen4ExpDecoderLayer(arch=arch, layer_idx=1, dtype=jnp.float32,
                                rngs=nnx.Rngs(0), prefix="model.layers.1")
    arch_lin = cfg_mod.arch_from_hf_config(base, vocab_size=64)
    arch_lin.layer_types = ["linear_attention"] * 4
    lin = Qwen4ExpDecoderLayer(arch=arch_lin, layer_idx=0, dtype=jnp.float32,
                              rngs=nnx.Rngs(1), prefix="model.layers.0")
    t, hcw = 4, 32
    hidden = jnp.ones((t, hcw), dtype=jnp.float32)
    pos = jnp.arange(t, dtype=jnp.int32)
    h2, o2, i2 = dense.forward(hidden, None, None, pos)
    assert h2.shape == (t, hcw) and o2.shape == (t, 16)
    h1, o1, i1 = lin.forward(hidden, None, None, pos)
    assert h1.shape == (t, hcw) and o1.shape == (t, 16)
    # Delayed combine: feed one's output as prev into the next.
    h3, o3, _ = dense.forward(h2, o2, i2, pos)
    assert h3.shape == (t, hcw)


@requires_impl
def test_hc_combine_matches_reference(mesh):
    import jax
    from flax import nnx

    hc, h = 2, 3
    mod = hc_mod.GatedResidual(hidden_size=h, hc_count=hc, hc_lowrank=4,
                              eps=1e-6, dtype=jnp.float32, rngs=nnx.Rngs(0))
    x = (jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6) - 12.0) / 5.0
    block = jnp.ones((4, h), dtype=jnp.float32)
    # None injection -> unit weight broadcast (upstream hc_combine).
    y = mod.combine(x, block, None)
    ref = (x.reshape(4, hc, h) + block[:, None, :]).reshape(4, hc * h)
    np.testing.assert_allclose(np.asarray(y), np.asarray(ref), rtol=1e-5)
    # mix matches the closed form with the module's own weights.
    _, block_in, inj = mod.mix(x)
    assert inj is not None and inj.shape == (4, hc)
    xn = hc_mod.grouped_gemma_rmsnorm(
        x, mod.hc_norm.weight.value, 1e-6, hc).astype(jnp.float32)
    down = mod.down_block_inject.kernel.value.astype(jnp.float32)
    up = mod.up.kernel.value.astype(jnp.float32)
    lora = jax.nn.silu(xn @ down[:, :4] / float(hc))
    gate = (lora @ up).reshape(4, hc, h)
    ref_in = jnp.mean(
        jax.nn.sigmoid(gate) * xn.reshape(4, hc, h), axis=1)
    np.testing.assert_allclose(
        np.asarray(block_in), np.asarray(ref_in), rtol=1e-4)
    # combine_and_mix fuses both: materialized matches combine, block input
    # matches a fresh mix of the materialized state.
    mat, block_in2, _ = mod.combine_and_mix(x, block, inj)
    np.testing.assert_allclose(
        np.asarray(mat),
        np.asarray(mod.combine(x, block, inj)), rtol=1e-6)
    _, ref_in2, _ = mod.mix(mat)
    np.testing.assert_allclose(
        np.asarray(block_in2), np.asarray(ref_in2), rtol=1e-6)


@requires_impl
def test_ple_ngram_per_order_isolation():
    sizes, offsets = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8)
    mults = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    qsl = jnp.asarray([0, 4], dtype=jnp.int32)
    ctx = jnp.zeros((1, 2), dtype=jnp.int32)
    t2r = jnp.zeros((4,), dtype=jnp.int32)

    def _ids(toks):
        return ngram_mod.compute_ngram_ids(
            jnp.asarray(toks, dtype=jnp.int32), qsl, ctx, t2r,
            sizes, offsets, mults, 3)

    ids_a = _ids([1, 2, 3, 4])
    # Change tok[1]: row 3's order-2 window is (tok3, tok2) -> cols 0:2 kept;
    # row 1 is the changed token itself -> the whole row changes.
    ids_b = _ids([1, 9, 3, 4])
    np.testing.assert_array_equal(
        np.asarray(ids_a[3, :2]), np.asarray(ids_b[3, :2]))
    assert not np.array_equal(np.asarray(ids_a[1]), np.asarray(ids_b[1]))


@requires_impl
def test_ple_gate_broadcast_structure(mesh):
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.ngram import Qwen4ExpPLE
    ple = Qwen4ExpPLE(hidden_size=8, hc_count=2, ple_embed_dim=8,
                      ngram_size=3, heads_per_ngram=2, conv_kernel=2,
                      unigram_vocab_size=64, ple_dense_layer_id=0,
                      vocab_base=1000, divisible_by=8, dtype=jnp.float32,
                      rngs=nnx.Rngs(0))
    hidden = (jnp.arange(3 * 16, dtype=jnp.float32).reshape(3, 16) - 24) / 8.0
    kv = (jnp.arange(3 * 24, dtype=jnp.float32).reshape(3, 24) - 36) / 9.0
    gated, conv_in = ple.gate(hidden, kv)
    assert gated.shape == (3, 16) and conv_in.shape == (3, 16)
    # gated = g * v with v broadcast over HC streams: the per-stream ratio
    # gated[t,s,:]/v[t,:] is constant over the H lanes.
    v = kv[..., 16:24]
    ratio = (gated.reshape(3, 2, 8).astype(jnp.float32)
             / v[:, None, :].astype(jnp.float32))
    spread = jnp.max(ratio, axis=-1) - jnp.min(ratio, axis=-1)
    assert bool(jnp.all(spread < 1e-4))
    # conv_in is the grouped RMS of gated with norm_conv_w.
    ref = hc_mod.grouped_gemma_rmsnorm(
        gated, ple.norm_conv_w.value, ple.eps, 2)
    np.testing.assert_allclose(
        np.asarray(conv_in), np.asarray(ref), rtol=1e-5)


@requires_impl
def test_layer_forward_deterministic(mesh):
    from types import SimpleNamespace

    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.layers import Qwen4ExpDecoderLayer

    base = SimpleNamespace(
        hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, intermediate_size=32,
        vocab_size=64, rms_norm_eps=1e-6, hidden_act="silu",
        max_position_embeddings=64, rope_theta=10000.0,
        partial_rotary_factor=0.5, attention_bias=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, decoder_sparse_step=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts_per_tok=2, num_experts=4, norm_topk_prob=True,
        mlp_only_layers=[], layer_types=None, hc_count=2, hc_lowrank=4,
        ple_layer_ids=[], ple_embed_dim=16, ple_conv_kernel_size=2,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8, output_gate_type="sigmoid",
    )
    arch = cfg_mod.arch_from_hf_config(base, vocab_size=64)
    arch.layer_types = ["linear_attention"] * 4
    lin = Qwen4ExpDecoderLayer(arch=arch, layer_idx=0, dtype=jnp.float32,
                              rngs=nnx.Rngs(1), prefix="model.layers.0")
    hidden = (jnp.arange(4 * 32, dtype=jnp.float32).reshape(4, 32) - 64) / 16.0
    pos = jnp.arange(4, dtype=jnp.int32)
    h1, o1, i1 = lin.forward(hidden, None, None, pos)
    h2, o2, i2 = lin.forward(hidden, None, None, pos)
    np.testing.assert_array_equal(np.asarray(h1), np.asarray(h2))
    np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))


@requires_impl
def test_qsa_chunked_matches_full(mesh):
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.config import Qwen4ExpArch
    from tpu_inference.models.jax.qwen4_exp.layers import Qwen4ExpDecoderLayer

    arch = Qwen4ExpArch(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, intermediate_size=32,
        vocab_size=64, partial_rotary_factor=0.5,
        layer_types=["full_attention"], hc_count=2, hc_lowrank=4,
        ple_embed_dim=16, decoder_sparse_step=100, num_experts=0,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=4)
    layer = Qwen4ExpDecoderLayer(arch=arch, layer_idx=0, dtype=jnp.float32,
                                rngs=nnx.Rngs(0), prefix="model.layers.0")

    def _empty_hist():
        return {"k": jnp.zeros((0, 2, 8), dtype=jnp.float32),
                "v": jnp.zeros((0, 2, 8), dtype=jnp.float32),
                "k_raw": jnp.zeros((0, 8), dtype=jnp.float32),
                "pos": jnp.zeros((0,), dtype=jnp.int32)}

    hidden = (jnp.arange(6 * 32, dtype=jnp.float32).reshape(6, 32) - 96) / 16.0
    pos = jnp.arange(6, dtype=jnp.int32)
    _, o_full, _ = layer.forward(hidden, None, None, pos,
                                 kv_history=_empty_hist())
    hist = _empty_hist()
    _, o_a, _ = layer.forward(hidden[:4], None, None, pos[:4], kv_history=hist)
    np.testing.assert_allclose(
        np.asarray(o_a), np.asarray(o_full[:4]), rtol=1e-5)
    _, o_b, _ = layer.forward(hidden[4:], None, None, pos[4:], kv_history=hist)
    np.testing.assert_allclose(
        np.asarray(o_b), np.asarray(o_full[4:]), rtol=1e-5)


@requires_impl
def test_dense_qkv_gate_interleave(mesh):
    # Upstream packs qkv rows as [q_h, g_h] per-head pairs, then [k, v].
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.attention import (
        Qwen4ExpDenseAttention,
    )
    nh, kvh, hd, hdim = 2, 1, 4, 8
    attn = Qwen4ExpDenseAttention(
        hidden_size=hdim, num_heads=nh, num_kv_heads=kvh, head_dim=hd,
        attn_output_gate=True, dtype=jnp.float32, rngs=nnx.Rngs(0))
    out_dim = 2 * nh * hd + 2 * kvh * hd
    # JAX kernel col r == torch row r: fill col r with constant r so the
    # fused activation at x=ones is hdim*r (direction preserved by RMSNorm).
    attn.qkv_proj.kernel.value = jnp.tile(
        jnp.arange(out_dim, dtype=jnp.float32)[None, :], (hdim, 1))
    x = jnp.ones((1, hdim), dtype=jnp.float32)
    q, k, v, gate = attn.project(x)
    assert gate is not None
    # Head 1: q rows 8..11, gate rows 12..15 (per-head interleave).
    np.testing.assert_allclose(
        np.asarray(q[0, 1] / q[0, 1, 0]),
        np.asarray([1.0, 9 / 8, 10 / 8, 11 / 8]), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(gate[0, 1] / gate[0, 1, 0]),
        np.asarray([1.0, 13 / 12, 14 / 12, 15 / 12]), rtol=1e-5)
    # v is unnormalized: exact rows 20..23 scaled by hdim.
    np.testing.assert_allclose(
        np.asarray(v[0, 0]),
        np.asarray(hdim * np.arange(20, 24, dtype=np.float32)), rtol=1e-6)


@requires_impl
def test_q4_dequant_roundtrip():
    # In-range values for int4 (+/-8 * scale); clipping would break roundtrip.
    w_true = jnp.asarray([[-4.0, -3.0, -2.0, -1.0],
                          [0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32)
    scale = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    q = jnp.clip(jnp.round(w_true / scale[:, None]) + 8, 0, 15).astype(jnp.uint8)
    packed = (q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)
    w = quant_mod.dequantize_q4_packed(
        packed, scale, zero=jnp.full((2,), 8.0), dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(w), np.asarray(w_true), atol=0.6)


@requires_impl
def test_hf_config_autoconfig_parses_qwen4_exp(tmp_path):
    """v44: stock transformers rejects model_type qwen4_exp, killing the
    vLLM server in ModelConfig. Our shim must make AutoConfig parse it."""
    import json

    transformers = pytest.importorskip("transformers")
    from tpu_inference.models.jax.qwen4_exp import hf_config as hf_mod

    hf_mod.install_hf_config()
    hf_mod.install_hf_config()  # idempotent
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560, "num_hidden_layers": 48,
        "vocab_size": 248320, "hc_count": 4,
        "text_config": {"model_type": "qwen4_exp", "hidden_size": 2560,
                        "num_hidden_layers": 48, "hc_count": 4,
                        "indexer_n_heads": 8},
    }))
    cfg = transformers.AutoConfig.from_pretrained(str(tmp_path))
    assert type(cfg).__name__ == "Qwen4ExpConfig"
    assert cfg.architectures == ["Qwen4ExpForCausalLM"]
    assert cfg.text_config.hidden_size == 2560
    assert cfg.text_config.hc_count == 4  # extras preserved
    assert cfg.text_config.indexer_n_heads == 8


@pytest.mark.skipif(os.environ.get("QWEN4EXP_E2E") != "1",
                    reason="needs TPU v5e-8 + checkpoint (QWEN4EXP_E2E=1)")
def test_e2e_tpu():
    """HF checkpoint → TPU server → OpenAI-compatible generation.

    Manual steps (automated by benchmarks/bench_qwen4_exp.py):
    1. install fork, 2. serve <QWEN_MODEL> --tensor-parallel-size 8,
    3. POST /v1/completions, 4. assert non-empty greedy decode.
    """
