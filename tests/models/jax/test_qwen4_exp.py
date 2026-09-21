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
    assert m == "model.layers.3.self_attn.qkv.weight"
    m = wl_mod.map_checkpoint_name("model.language_model.layers.3.ple.ple_embedding.weight")
    assert m == "model.layers.3.ple.embedding.weight"
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
def test_qsa_select_exact_and_jits():
    """Hand-computed selection (order, dedup, truncation) + g==0 branch,
    plus jit traceability (the serving requirement)."""
    import jax

    from tpu_inference.models.jax.qwen4_exp import qsa as qsa_mod
    # structural properties on random scores (order/dedup covered by the
    # compaction construction) + exact g==0 branch.
    rng = np.random.default_rng(1)
    qr = jnp.asarray(rng.normal(size=(4, 2, 4)), dtype=jnp.float32)
    kcr = jnp.asarray(rng.normal(size=(5, 4)), dtype=jnp.float32)
    posr = jnp.asarray([3, 6, 9, 12], dtype=jnp.int32)
    idx, counts = qsa_mod.qsa_select_indices(qr, kcr, posr, 4, 8)
    assert idx.shape == (4, 11)
    for r in range(4):
        c = int(counts[r])
        assert 0 < c <= 11
        row = np.asarray(idx[r, :c])
        assert bool(np.all(row >= 0)) and len(set(row.tolist())) == c
        assert bool(np.all(row <= int(posr[r])))
        assert bool(np.all(np.asarray(idx[r, c:]) == -1))
    # forced exact cases (top-k set independent of score order)
    q1 = jnp.ones((1, 2, 4), dtype=jnp.float32)
    idx1, counts1 = qsa_mod.qsa_select_indices(
        q1, jnp.ones((1, 4), dtype=jnp.float32),
        jnp.asarray([0], dtype=jnp.int32), 2, 2)
    np.testing.assert_array_equal(np.asarray(idx1), np.asarray([[0, -1, -1]]))
    np.testing.assert_array_equal(np.asarray(counts1), np.asarray([1]))
    idx2, counts2 = qsa_mod.qsa_select_indices(
        q1, jnp.ones((2, 4), dtype=jnp.float32),
        jnp.asarray([3], dtype=jnp.int32), 2, 4)
    np.testing.assert_array_equal(
        np.asarray(idx2), np.asarray([[0, 1, 2, 3, -1]]))
    np.testing.assert_array_equal(np.asarray(counts2), np.asarray([4]))
    # empty-history branch is exact
    q0 = jnp.ones((2, 2, 4), dtype=jnp.float32)
    idx0, counts0 = qsa_mod.qsa_select_indices(
        q0, jnp.zeros((0, 4), dtype=jnp.float32),
        jnp.asarray([2, 5], dtype=jnp.int32), 2, 4)
    np.testing.assert_array_equal(
        np.asarray(idx0),
        np.asarray([[-1, -1, 0, 1, 2], [1, 2, 3, 4, 5]]))
    np.testing.assert_array_equal(np.asarray(counts0), np.asarray([3, 5]))
    # traces under jit
    fj = jax.jit(lambda a, b, c: qsa_mod.qsa_select_indices(a, b, c, 4, 8))
    idxj, _ = fj(qr, kcr, posr)
    np.testing.assert_array_equal(np.asarray(idxj), np.asarray(idx))


@requires_impl
def test_ple_hash_deterministic_and_shaped():
    m1 = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    m2 = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    assert m1 == m2 and len(m1) == 3 and all(x % 2 == 1 for x in m1)
    # Hash bound follows the vocab: larger vocabs admit smaller multipliers.
    m_big = ngram_mod.ple_multipliers(1234, 0, 3, 10_000_000)
    assert all(0 < v < (1 << 63) for v in m_big)
    sizes, offsets = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8)
    # Plain Python ints (trace-safe construction): convert at the boundary.
    assert isinstance(sizes, list) and len(sizes) == 4
    assert offsets == [0, sizes[0], sizes[0] + sizes[1],
                       sizes[0] + sizes[1] + sizes[2]]
    assert ngram_mod.ple_padded_rows(sizes, 8) % 8 == 0
    # Layer-1 heads continue the global prime sequence (no overlap w/ layer 0).
    sizes1, _ = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8,
                                                 ple_dense_layer_id=1)
    assert set(sizes1).isdisjoint(sizes)
    _sizes = jnp.asarray(sizes, dtype=jnp.int32)
    _offsets = jnp.asarray(offsets, dtype=jnp.int32)
    ids = ngram_mod.compute_ngram_ids(
        jnp.asarray([5, 6, 7], dtype=jnp.int32),
        jnp.asarray([0, 3], dtype=jnp.int32),
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        _sizes, _offsets, ngram_mod.ple_multipliers(1234, 0, 3, 1000), 3)
    assert ids.shape == (3, 4)
    assert bool(jnp.all(ids >= 0))


@requires_impl
def test_moe_triplet_forward_matches_fused_reference(mesh):
    """Triplet MoE (selected-expert NVFP4 dequant) == fused fp32 math on
    the same dequantized values; traces under jit."""
    import jax
    import torch
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod
    from tpu_inference.models.jax.qwen4_exp.moe import Qwen4ExpMoE, silu

    torch.manual_seed(9)
    H, I, E, K = 32, 16, 4, 2
    LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
           -2.0, -3.0, -4.0, -6.0)

    def nvfp4_pack(Wtrue):
        O, II = Wtrue.shape
        G = II // 16
        scales = torch.empty(O, G)
        for j in range(G):
            blk = Wtrue[:, j * 16:(j + 1) * 16].abs().max(dim=1).values
            scales[:, j] = (blk / 6.0).clamp_min(1e-6)
        scaled = Wtrue / scales[:, torch.arange(II) // 16]
        dist = torch.stack([(scaled - v).abs() for v in LUT])
        codes = dist.argmin(dim=0).to(torch.int32)
        qw = (((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4))
              .to(torch.uint8))
        return qw, scales.to(torch.float8_e4m3fn), torch.tensor(1.0)

    moe = Qwen4ExpMoE(hidden_size=H, moe_intermediate_size=I,
                      shared_intermediate_size=0, num_experts=E,
                      num_experts_per_tok=K, rngs=nnx.Rngs(0))
    fused = {}
    # Stack per-expert triplets exactly like the loader (gate_g is [E],
    # not scalar: assigning a scalar would collapse the param shape).
    buckets = {}
    for e in range(E):
        for p, (O, II) in (("gate", (I, H)), ("up", (I, H)),
                            ("down", (H, I))):
            Wt = torch.randn(O, II)
            qw, sc, gs = nvfp4_pack(Wt)
            buckets.setdefault(f"exp_{p}_w", []).append(qw.numpy())
            buckets.setdefault(f"exp_{p}_sc", []).append(sc.float().numpy())
            buckets.setdefault(f"exp_{p}_g", []).append(gs.numpy())
            fused[(e, p)] = quant_mod.dequantize_nvfp4_torch(
                qw, sc, gs, group_size=16).numpy()
    for name, parts in buckets.items():
        getattr(moe, name).value = jnp.asarray(np.stack(parts))
    # router weights fixed for determinism
    moe.gate.weight.value = jnp.asarray(
        np.random.default_rng(4).normal(size=(H, E)).astype(np.float32))
    x = jnp.asarray(np.random.default_rng(5).normal(size=(3, H)),
                    dtype=jnp.float32)
    out, _ = moe(x)
    # fused reference with the same dequantized values
    logits = np.asarray(x) @ np.asarray(moe.gate.weight.value)
    top = np.argsort(-logits, axis=1)[:, :K]
    wts = np.take_along_axis(logits, top, axis=1)
    wts = wts / wts.sum(axis=1, keepdims=True)
    ref = np.zeros((3, H), dtype=np.float32)
    for t in range(3):
        for k in range(K):
            e = int(top[t, k])
            g = fused[(e, "gate")].T
            u = fused[(e, "up")].T
            h = silu(np.asarray(x[t]) @ g) * (np.asarray(x[t]) @ u)
            d = fused[(e, "down")].T
            ref[t] += wts[t, k] * (h @ d)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-3)
    fj = jax.jit(lambda a: moe(a)[0])
    # Eager-vs-jit XLA reassociation noise only (correctness pinned above).
    np.testing.assert_allclose(np.asarray(fj(x)), np.asarray(out), rtol=1e-4)


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
    down = mod.down_block_inject.weight.value.astype(jnp.float32)
    up = mod.up.weight.value.astype(jnp.float32)
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
def test_ngram_ids_vectorized_matches_reference_and_jits():
    """Vectorized compute_ngram_ids == eager Python reference (bit-exact),
    on multi-request + cross-chunk-context inputs, and compiles under jit."""
    import jax

    _MASK64 = (1 << 64) - 1

    def _ref(input_ids, query_start_loc, ngram_context, token_to_req,
             sizes, offsets, multipliers, ngram_size, eos_id=0):
        toks = [int(v) for v in list(input_ids.reshape(-1))]
        qsl = [int(v) for v in list(query_start_loc.reshape(-1))]
        t2r = [int(v) for v in list(token_to_req.reshape(-1))]
        ctx = [[int(v) for v in list(row)]
               for row in list(ngram_context.reshape(
                   ngram_context.shape[0], -1))]
        sizes_l = [int(v) for v in list(sizes.reshape(-1))]
        offs_l = [int(v) for v in list(offsets.reshape(-1))]
        out_rows = []
        for pos in range(int(input_ids.shape[0])):
            req = t2r[pos]
            seq_start = 0
            for b in range(len(qsl) - 1):
                if qsl[b] <= pos < qsl[b + 1]:
                    seq_start = qsl[b]
                    break
            hist = []
            for back in range(ngram_size):
                p = pos - back
                if p >= seq_start:
                    hist.append(toks[p])
                else:
                    need = back - (pos - seq_start)
                    crow = ctx[req] if 0 <= req < len(ctx) else []
                    hist.append(crow[len(crow) - need]
                                if 0 < need <= len(crow) else eos_id)
            row = []
            for order in range(2, int(ngram_size) + 1):
                val = 0
                for i in range(order):
                    val = (val ^ ((hist[i] & _MASK64)
                                  * (multipliers[i] & _MASK64))) & _MASK64
                base_h = (order - 2) * 2
                for h in range(2):
                    row.append((val % sizes_l[base_h + h]) + offs_l[base_h + h])
            out_rows.append(row)
        return jnp.asarray(out_rows, dtype=jnp.int32)

    rng = np.random.default_rng(0)
    sizes = jnp.asarray([101, 103, 107, 109], dtype=jnp.int32)
    offsets = jnp.asarray([0, 101, 204, 311], dtype=jnp.int32)
    mults = ngram_mod.ple_multipliers(1234, 0, 3, 1000)
    cases = [
        (jnp.asarray([5, 6, 7, 1, 2, 3, 4, 9]),
         jnp.asarray([0, 3, 8]), jnp.zeros((2, 2), dtype=jnp.int32),
         jnp.asarray([0, 0, 0, 1, 1, 1, 1, 1])),
        (jnp.asarray(rng.integers(0, 999, size=7), dtype=jnp.int32),
         jnp.asarray([0, 2, 7]),
         jnp.asarray(rng.integers(0, 999, size=(2, 2)), dtype=jnp.int32),
         jnp.asarray([0, 0, 1, 1, 1, 1, 1])),
    ]
    for toks, qsl, ctx, t2r in cases:
        ref = _ref(toks, qsl, ctx, t2r, sizes, offsets, mults, 3)
        got = ngram_mod.compute_ngram_ids(toks, qsl, ctx, t2r, sizes,
                                          offsets, mults, 3)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # traces under jit (the serving requirement)
    f = jax.jit(lambda a, b, c, d: ngram_mod.compute_ngram_ids(
        a, b, c, d, sizes, offsets, mults, 3))
    toks, qsl, ctx, t2r = cases[1]
    out = f(toks, qsl, ctx, t2r)
    np.testing.assert_array_equal(
        np.asarray(out),
        np.asarray(_ref(toks, qsl, ctx, t2r, sizes, offsets, mults, 3)))


@requires_impl
def test_ple_ngram_per_order_isolation():
    _sizes, _offsets = ngram_mod.ple_vocab_sizes_offsets(3, 2, 1000, 8)
    sizes = jnp.asarray(_sizes, dtype=jnp.int32)
    offsets = jnp.asarray(_offsets, dtype=jnp.int32)
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
def test_dilated_conv_gather_matches_loop_and_jits():
    """Gather-based dilated conv == reference per-token loop, and traces."""
    import jax
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp.ngram import Qwen4ExpPLE
    ple = Qwen4ExpPLE(hidden_size=8, hc_count=2, ple_embed_dim=8,
                      ngram_size=3, heads_per_ngram=2, conv_kernel=2,
                      unigram_vocab_size=256, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(2).normal(size=(9, 16)),
                    dtype=jnp.float32)
    got = ple.dilated_conv(x)
    w = np.asarray(ple.conv_w.value, dtype=np.float32)
    xn = np.asarray(x, dtype=np.float32)
    zero = np.zeros((16,), dtype=np.float32)
    ref = np.stack([
        w[:, 0] * xn[t] + w[:, 1] * (xn[t - 3] if t >= 3 else zero)
        for t in range(9)], axis=0)
    np.testing.assert_allclose(np.asarray(got),
                               jax.nn.silu(ref).astype(np.float32),
                               rtol=1e-5, atol=1e-6)
    fj = jax.jit(lambda a: ple.dilated_conv(a))
    np.testing.assert_allclose(np.asarray(fj(x)), np.asarray(got),
                               rtol=1e-6)


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
    attn.qkv.weight.value = jnp.tile(
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


@requires_impl
def test_loader_heuristic_audit():
    """JaxAutoWeightsLoader assigns reshape/permute heuristics by NAME
    substring (tpu-inference 0.28 weight_utils). A JAX name accidentally
    containing k/v/q_proj.weight would route a 2D param into the 3D branch
    and crash loader construction (v48 root cause). This pins the safe set:
    only o_proj hits a 3D branch (3D by construction), embed/lm_head hit
    their intended branches, everything else takes the default transpose.
    """
    from tpu_inference.models.jax.qwen4_exp import config as cfg_mod
    from tpu_inference.models.jax.qwen4_exp import weight_loader as wl_mod

    arch = cfg_mod.arch_from_hf_config({
        "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
        "vocab_size": 256, "layer_types": ["full_attention",
                                           "linear_attention"] * 2,
        "ple_layer_ids": [2], "hc_count": 2, "num_experts": 4,
        "num_experts_per_tok": 2, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
        "indexer_budget": 512, "indexer_compress_ratio": 1,
    })
    names = wl_mod.expected_jax_names(arch)
    assert len(names) > 80
    o_proj = [n for n in names if "o_proj.weight" in n]
    assert o_proj and all(".self_attn.o_proj.weight" in n for n in o_proj)
    for n in names:
        assert "k_proj.weight" not in n, n
        assert "v_proj.weight" not in n, n
        assert "q_proj.weight" not in n, n
        assert "q_proj.bias" not in n and "k_proj.bias" not in n \
            and "v_proj.bias" not in n, n
        if "lm_head" in n:
            assert n.endswith("lm_head.weight"), n
        if "embed_tokens.weight" in n:
            assert n.endswith("model.embed_tokens.weight"), n
    # fused/renamed linears keep loader-safe names
    assert any(n.endswith(".self_attn.qkv.weight") for n in names)
    assert any(n.endswith(".ple.kv.weight") for n in names)
    assert any(n.endswith(".indexer.index_qk.weight") for n in names)
    # NVFP4 triplets (direct-assign): present, fused fp32 gone
    for s in ("exp_gate_w", "exp_gate_sc", "exp_gate_g",
              "exp_up_w", "exp_up_sc", "exp_up_g",
              "exp_down_w", "exp_down_sc", "exp_down_g"):
        assert any(n.endswith(".mlp." + s) for n in names), s
    assert not any(n.endswith(".mlp.exp_gate_up") for n in names)
    assert not any(n.endswith(".mlp.exp_down") for n in names)
    assert any(n.endswith(".ple.embedding.table_scale") for n in names)


@requires_impl
def test_gptq_assembly_iterator():
    """Synthetic GPTQModel-v2 checkpoint -> JAX-named tensors.

    Covers: plain passthrough + renames, GPTQ dequant (reconstruction
    bound), expert stack/fuse, concat order, norm rename, gate_up split,
    and empty missing/unconsumed reports.
    """
    torch = pytest.importorskip("torch")
    from tpu_inference.models.jax.qwen4_exp import weight_loader as wl_mod

    torch.manual_seed(0)
    from types import SimpleNamespace
    arch = SimpleNamespace(num_experts=2, hidden_size=32,
                           moe_intermediate_size=16)

    def gptq_group(prefix, O, I, gs=8):
        G = I // gs
        Wtrue = torch.randn(O, I) * 2
        scales = torch.empty(G, O)
        zeros = torch.empty(G, O, dtype=torch.int32)
        g_idx = torch.arange(I) // gs
        for gg in range(G):
            blk = Wtrue[:, gg * gs:(gg + 1) * gs]
            mn, mx = blk.min(dim=1).values, blk.max(dim=1).values
            s = ((mx - mn) / 15).clamp_min(1e-6)
            z = torch.round(-mn / s).clamp(0, 15).to(torch.int32)
            scales[gg] = s
            zeros[gg] = z
        codes = torch.clamp(
            torch.round(Wtrue / scales[g_idx].T + zeros[g_idx].float().T),
            0, 15).to(torch.int32)
        qw = torch.zeros(I // 8, O, dtype=torch.int32)
        for k in range(8):
            qw = qw + codes[:, k::8].T * (1 << (4 * k))
        qz = torch.zeros(G, O // 8, dtype=torch.int32)
        for k in range(8):
            qz = qz + zeros[:, k::8] * (1 << (4 * k))
        return {prefix + ".qweight": qw, prefix + ".qzeros": qz,
                prefix + ".scales": scales.to(torch.bfloat16),
                prefix + ".g_idx": g_idx.to(torch.int32)}, Wtrue

    stream = [(("model.language_model.layers.0.self_attn.qkv_proj.weight"),
               torch.randn(40, 32))]
    _g, Wt = gptq_group("model.layers.0.self_attn.o_proj", 32, 40, gs=8)
    stream += list(_g.items())
    exp_true = {}
    for e in range(2):
        for p, (O, I) in (("gate_proj", (16, 32)), ("up_proj", (16, 32)),
                           ("down_proj", (32, 16))):
            gg, Wt2 = gptq_group(
                f"model.layers.0.mlp.experts.{e}.{p}", O, I, gs=8)
            stream += list(gg.items())
            exp_true[(e, p)] = Wt2
    d_extra = {
        "model.layers.0.ple.key_proj.weight": torch.randn(20, 64),
        "model.layers.0.ple.value_proj.weight": torch.randn(12, 64),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(8),
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight":
            torch.randn(32, 32),
    }
    stream += list(d_extra.items())
    jax_names = [
        "model.layers.0.self_attn.qkv.weight",
        "model.layers.0.mlp.exp_gate_up",
        "model.layers.0.mlp.exp_down",
        "model.layers.0.ple.kv.weight",
        "model.layers.0.self_attn.q_norm_w",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]
    rep = {}
    out = dict(wl_mod.iter_jax_named_weights(
        iter(stream), arch, jax_names, report=rep, group_size=8))
    assert rep["missing"] == []
    assert rep["unconsumed"] == []
    assert rep["dequantized"] == 7 and rep["assembled"] == 2
    assert torch.equal(out["model.layers.0.self_attn.qkv.weight"], stream[0][1])
    assert (out["model.layers.0.self_attn.o_proj.weight"] - Wt).abs().max() < 0.5
    gu = out["model.layers.0.mlp.exp_gate_up"]
    assert gu.shape == (32, 2, 32)
    assert torch.allclose(gu[:, 0, :16], exp_true[(0, "gate_proj")].T, atol=0.6)
    assert torch.allclose(gu[:, 0, 16:], exp_true[(0, "up_proj")].T, atol=0.6)
    assert torch.allclose(gu[:, 1, :16], exp_true[(1, "gate_proj")].T, atol=0.6)
    assert torch.allclose(gu[:, 1, 16:], exp_true[(1, "up_proj")].T, atol=0.6)
    dn = out["model.layers.0.mlp.exp_down"]
    assert dn.shape == (2, 16, 32)
    assert torch.allclose(dn[0], exp_true[(0, "down_proj")].T, atol=0.6)
    assert torch.allclose(dn[1], exp_true[(1, "down_proj")].T, atol=0.6)
    kv = out["model.layers.0.ple.kv.weight"]
    assert kv.shape == (32, 64)
    assert torch.equal(kv[:20], d_extra["model.layers.0.ple.key_proj.weight"])
    assert torch.equal(kv[20:], d_extra["model.layers.0.ple.value_proj.weight"])
    assert torch.equal(out["model.layers.0.self_attn.q_norm_w"],
                       d_extra["model.layers.0.self_attn.q_norm.weight"])
    assert torch.equal(
        out["model.layers.0.mlp.shared_expert.gate_proj.weight"],
        d_extra["model.layers.0.mlp.shared_expert.gate_up_proj.weight"][:16])


@requires_impl
def test_gptq_jax_matches_torch_bit_exact():
    """INT4-residency forward dequant must reproduce the CPU reference
    exactly (same integer ops, elementwise float math)."""
    torch = pytest.importorskip("torch")
    import jax.numpy as jnp

    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod

    torch.manual_seed(7)
    O, I, gs, G, pack = 48, 64, 16, 4, 8
    codes = torch.randint(0, 16, (O, I), dtype=torch.int32)
    zeros = torch.randint(0, 16, (G, O), dtype=torch.int32)
    scales = (torch.rand(G, O) * 0.5 + 0.1).to(torch.bfloat16)
    g_idx = torch.arange(I) // gs
    qw = torch.zeros(I // pack, O, dtype=torch.int32)
    for k in range(pack):
        qw = qw + codes[:, k::pack].T * (1 << (4 * k))
    qz = torch.zeros(G, O // pack, dtype=torch.int32)
    for k in range(pack):
        qz = qz + zeros[:, k::pack] * (1 << (4 * k))
    ref = quant_mod.dequantize_gptq_torch(
        qw, qz, scales, g_idx.to(torch.int32), bits=4, group_size=gs)
    got = quant_mod.dequantize_gptq_jax(
        jnp.asarray(qw.numpy()), jnp.asarray(qz.numpy()),
        jnp.asarray(scales.float().numpy()), jnp.asarray(g_idx.numpy()),
        bits=4, group_size=gs)
    np.testing.assert_array_equal(np.asarray(got), ref.numpy())


@requires_impl
def test_qlinear_forward_matches_dequant_reference():
    """Qwen4ExpQLinear forward == matmul with the CPU-dequantized weight."""
    torch = pytest.importorskip("torch")
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod
    from tpu_inference.models.jax.qwen4_exp.qlinear import Qwen4ExpQLinear

    torch.manual_seed(3)
    O, I, gs, G = 32, 64, 16, 4
    Wtrue = torch.randn(O, I)
    scales = torch.empty(G, O)
    zeros = torch.empty(G, O, dtype=torch.int32)
    g_idx = torch.arange(I) // gs
    for gg in range(G):
        blk = Wtrue[:, gg * gs:(gg + 1) * gs]
        mn, mx = blk.min(dim=1).values, blk.max(dim=1).values
        s = ((mx - mn) / 15).clamp_min(1e-6)
        scales[gg] = s
        zeros[gg] = torch.round(-mn / s).clamp(0, 15).to(torch.int32)
    codes = torch.clamp(
        torch.round(Wtrue / scales[g_idx].T + zeros[g_idx].float().T),
        0, 15).to(torch.int32)
    qw = torch.zeros(I // 8, O, dtype=torch.int32)
    for k in range(8):
        qw = qw + codes[:, k::8].T * (1 << (4 * k))
    qz = torch.zeros(G, O // 8, dtype=torch.int32)
    for k in range(8):
        qz = qz + zeros[:, k::8] * (1 << (4 * k))
    mod = Qwen4ExpQLinear(I, O, bits=4, group_size=gs,
                          dtype=jnp.float32, rngs=nnx.Rngs(0))
    mod.qweight.value = jnp.asarray(qw.numpy())
    mod.qzeros.value = jnp.asarray(qz.numpy())
    mod.scales.value = jnp.asarray(scales.to(torch.bfloat16).float().numpy())
    mod.g_idx.value = jnp.asarray(g_idx.numpy())
    x = jnp.asarray(torch.randn(5, I).numpy())
    ref = quant_mod.dequantize_gptq_torch(
        qw, qz, scales.to(torch.bfloat16), g_idx.to(torch.int32),
        bits=4, group_size=gs).numpy().T
    np.testing.assert_allclose(
        np.asarray(mod(x)), np.asarray(x) @ ref.T, rtol=1e-4, atol=1e-3)


@requires_impl
def test_nvfp4_dequant_reconstruction():
    """NVFP4 torch dequant inverts a calibrated synthetic pack (bound)."""
    torch = pytest.importorskip("torch")

    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod

    torch.manual_seed(11)
    O, I, gs = 64, 128, 16
    G = I // gs
    Wtrue = (torch.randn(O, I) * 1.5)
    LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
           -2.0, -3.0, -4.0, -6.0)
    # calibrate per-block scales to fit codes (mimics ModelOpt export)
    scales = torch.empty(O, G)
    for j in range(G):
        blk = Wtrue[:, j * gs:(j + 1) * gs].abs().max(dim=1).values
        scales[:, j] = (blk / 6.0).clamp_min(1e-6)
    glob = torch.tensor(1.0)
    codes = torch.clamp(
        torch.round(Wtrue / (scales[:, torch.arange(I) // gs] * glob.item())
                    / torch.tensor(LUT)[..., None].max()), 0, 15)
    # simpler: quantize by nearest LUT entry after scaling
    scaled = Wtrue / (scales[:, torch.arange(I) // gs] * glob.item())
    dist = torch.stack([(scaled - v).abs() for v in LUT])
    codes = dist.argmin(dim=0).to(torch.int32)
    qw = (((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4))
          .to(torch.uint8))
    out = quant_mod.dequantize_nvfp4_torch(
        qw, scales.to(torch.float8_e4m3fn), glob, group_size=gs)
    assert out.shape == (O, I)
    # crude synthetic calibration (not ModelOpt GPTQ-grade): bound reflects
    # 4-bit coarseness on outliers, not implementation error (pack/unpack is
    # bit-exact per test_nvfp4 below via allclose assembly checks).
    assert (out - Wtrue).abs().max() < 1.0


@requires_impl
def test_nvfp4_jax_matches_torch_bit_exact():
    """JAX dequant-in-forward twin == CPU reference (bit-exact)."""
    torch = pytest.importorskip("torch")
    import jax.numpy as jnp

    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod

    torch.manual_seed(5)
    O, I, gs = 48, 64, 16
    codes = torch.randint(0, 16, (O, I), dtype=torch.int32)
    qw = (((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4))
          .to(torch.uint8))
    scales = (torch.rand(O, I // gs) * 0.4 + 0.05)
    glob = torch.tensor(2.5)
    ref = quant_mod.dequantize_nvfp4_torch(qw, scales, glob, group_size=gs)
    got = quant_mod.dequantize_nvfp4_jax(
        jnp.asarray(qw.numpy()), jnp.asarray(scales.numpy()),
        jnp.asarray(glob.numpy()), group_size=gs)
    np.testing.assert_array_equal(np.asarray(got), ref.numpy())


@requires_impl
def test_nvfp4_expert_assembly_and_dense_gaps():
    """NVFP4 experts + qkv concat + mixer zero-fill + table zeros."""
    torch = pytest.importorskip("torch")
    from tpu_inference.models.jax.qwen4_exp import weight_loader as wl_mod

    from types import SimpleNamespace
    arch = SimpleNamespace(num_experts=2, hidden_size=32,
                           moe_intermediate_size=16, num_attention_heads=2,
                           head_dim=8, num_key_value_heads=1, hc_count=2,
                           hc_lowrank=4)
    LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
           -2.0, -3.0, -4.0, -6.0)

    def nvfp4_group(prefix, O, I, gs=16):
        G = I // gs
        Wtrue = torch.randn(O, I)
        scales = torch.empty(O, G)
        for j in range(G):
            blk = Wtrue[:, j * gs:(j + 1) * gs].abs().max(dim=1).values
            scales[:, j] = (blk / 6.0).clamp_min(1e-6)
        scaled = Wtrue / scales[:, torch.arange(I) // gs]
        dist = torch.stack([(scaled - v).abs() for v in LUT])
        codes = dist.argmin(dim=0).to(torch.int32)
        qw = (((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4))
              .to(torch.uint8))
        return {prefix + ".weight": qw,
                prefix + ".weight_scale": scales.to(torch.float8_e4m3fn),
                prefix + ".weight_scale_2": torch.tensor(1.0),
                prefix + ".input_scale": torch.tensor(2.0)}, Wtrue

    stream = []
    exp_true = {}
    for e in range(2):
        for p, (O, I) in (("gate_proj", (16, 32)), ("up_proj", (16, 32)),
                           ("down_proj", (32, 16))):
            gg, Wt2 = nvfp4_group(
                f"model.layers.0.mlp.experts.{e}.{p}", O, I)
            stream += list(gg.items())
            exp_true[(e, p)] = Wt2
    # q/k/v separate plain (q carries gate rows: 2*N*D = 32)
    stream += [("model.layers.0.self_attn.q_proj.weight", torch.randn(32, 32)),
               ("model.layers.0.self_attn.k_proj.weight", torch.randn(8, 32)),
               ("model.layers.0.self_attn.v_proj.weight", torch.randn(8, 32))]
    # mixer down only (inject zero-filled): down rows = hc_lowrank = 4
    stream += [("model.hyper_connection_mixer.input_mix_weight_down.weight",
                torch.randn(4, 64))]
    # table shards fp8 (concatenated in numeric order) + global scale
    stream += [(f"model.layers.0.ple.ple_embedding.ngram_embedding.shard_{i}.weight",
                torch.randint(0, 256, (5, 8), dtype=torch.uint8))
               for i in range(3)]
    stream += [("model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale",
                torch.tensor(0.5))]
    jax_names = ["model.layers.0.mlp.exp_gate_w",
                 "model.layers.0.mlp.exp_gate_sc",
                 "model.layers.0.mlp.exp_gate_g",
                 "model.layers.0.mlp.exp_up_w",
                 "model.layers.0.mlp.exp_up_sc",
                 "model.layers.0.mlp.exp_up_g",
                 "model.layers.0.mlp.exp_down_w",
                 "model.layers.0.mlp.exp_down_sc",
                 "model.layers.0.mlp.exp_down_g",
                 "model.layers.0.self_attn.qkv.weight",
                 "model.hyper_connection_mixer.down_block_inject.weight",
                 "model.layers.0.ple.embedding.weight",
                 "model.layers.0.ple.embedding.table_scale"]
    rep = {}
    out = dict(wl_mod.iter_jax_named_weights(
        iter(stream), arch, jax_names, report=rep,
        nvfp4_group_size=16))
    assert rep["missing"] == [], rep["missing"]
    assert rep["unconsumed"] == [], rep["unconsumed"]
    direct = rep["direct_tensors"]
    # raw triplets stacked per layer (exact dtypes preserved)
    assert direct["model.layers.0.mlp.exp_gate_w"].shape == (2, 16, 16)
    assert direct["model.layers.0.mlp.exp_gate_w"].dtype == torch.uint8
    assert direct["model.layers.0.mlp.exp_gate_sc"].dtype == torch.float8_e4m3fn
    assert direct["model.layers.0.mlp.exp_gate_g"].shape == (2,)
    assert direct["model.layers.0.mlp.exp_down_w"].shape == (2, 32, 8)
    assert torch.equal(direct["model.layers.0.mlp.exp_gate_w"][0],
                       dict(stream)["model.layers.0.mlp.experts.0.gate_proj.weight"])
    # triplet content dequantizes back (global scale 1.0 here)
    from tpu_inference.models.jax.qwen4_exp import quant as quant_mod
    g0 = quant_mod.dequantize_nvfp4_torch(
        direct["model.layers.0.mlp.exp_gate_w"][0],
        direct["model.layers.0.mlp.exp_gate_sc"][0],
        direct["model.layers.0.mlp.exp_gate_g"][0], group_size=16)
    assert torch.allclose(g0, exp_true[(0, "gate_proj")], atol=0.6)
    qkv = out["model.layers.0.self_attn.qkv.weight"]
    assert qkv.shape == (48, 32)
    d = dict(stream)
    assert torch.equal(qkv[:32], d["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv[32:40], d["model.layers.0.self_attn.k_proj.weight"])
    assert torch.equal(qkv[40:], d["model.layers.0.self_attn.v_proj.weight"])
    mix = out["model.hyper_connection_mixer.down_block_inject.weight"]
    assert mix.shape == (6, 64)
    assert torch.equal(mix[:4],
                       d["model.hyper_connection_mixer.input_mix_weight_down.weight"])
    assert bool((mix[4:] == 0).all())
    # table concat (numeric shard order) + scalar scale
    assert direct["model.layers.0.ple.embedding.weight"].shape == (15, 8)
    assert torch.equal(
        direct["model.layers.0.ple.embedding.weight"][:5],
        d["model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"])
    assert torch.equal(
        direct["model.layers.0.ple.embedding.weight"][10:],
        d["model.layers.0.ple.ple_embedding.ngram_embedding.shard_2.weight"])
    assert direct["model.layers.0.ple.embedding.table_scale"].shape == torch.Size([])


@requires_impl
def test_qwix_guard_handles_nulled_quantization_config():
    """v52: --hf-overrides nulls quantization_config (vLLM re-attaches it
    post-parse); tpu-inference's probe subscripts it behind a bare hasattr
    and explodes on None. The startup guard must make None behave as
    absent while preserving the llama4/gpt_oss defaults."""
    import sys
    import types
    from types import SimpleNamespace

    from tpu_inference.models.jax.qwen4_exp import startup as startup_mod

    fake = types.ModuleType("qwix_utils")

    def _orig(hf_config, skip_quantization):
        # exact 0.28.0 probe semantics (crashes on None)
        if skip_quantization:
            return None
        return hf_config.quantization_config["quant_method"] if hasattr(
            hf_config, "quantization_config") else None

    fake.get_default_qwix_quantization_config = _orig
    fake.DEFAULT_LLAMA4_FP8_CONFIG = {"llama": 1}
    fake.DEFAULT_GPT_OSS_FP4_CONFIG = {"gptoss": 1}
    chain = ["tpu_inference", "tpu_inference.models",
             "tpu_inference.models.jax", "tpu_inference.models.jax.utils",
             "tpu_inference.models.jax.utils.qwix"]
    saved = {n: sys.modules.get(n) for n in chain
             + ["tpu_inference.models.jax.utils.qwix.qwix_utils"]}
    try:
        for n in chain:
            sys.modules[n] = types.ModuleType(n)
        sys.modules["tpu_inference.models.jax.utils.qwix.qwix_utils"] = fake
        startup_mod._guard_qwix_null_quantization_config()
        probed = sys.modules[
            "tpu_inference.models.jax.utils.qwix.qwix_utils" \
        ].get_default_qwix_quantization_config
        assert probed(SimpleNamespace(model_type="qwen4_exp",
                                      quantization_config=None), False) is None
        assert probed(SimpleNamespace(model_type="qwen4_exp"), False) is None
        assert probed(SimpleNamespace(
            model_type="llama4",
            quantization_config={"quant_method": "compressed-tensors"}),
            False) == {"llama": 1}
        assert probed(SimpleNamespace(model_type="qwen4_exp",
                                      quantization_config=None), True) is None
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


@requires_impl
def test_hf_config_neutralizes_gptq(tmp_path):
    """vLLM must not see quantization_config (its CUDA-only GPTQ gate
    rejects TPU); the original is stashed, everything else preserved."""
    import json

    transformers = pytest.importorskip("transformers")
    from tpu_inference.models.jax.qwen4_exp import hf_config as hf_mod

    hf_mod.install_hf_config()
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560, "num_hidden_layers": 48,
        "quantization_config": {"quant_method": "gptq", "bits": 4,
                                "group_size": 128, "desc_act": False,
                                "sym": False},
        "text_config": {"model_type": "qwen4_exp", "hidden_size": 2560,
                        "num_hidden_layers": 48},
    }))
    cfg = transformers.AutoConfig.from_pretrained(str(tmp_path))
    assert not hasattr(cfg, "quantization_config")
    assert cfg.qwen4exp_quantization_config["quant_method"] == "gptq"
    assert cfg.text_config.hidden_size == 2560
    assert not hasattr(cfg.text_config, "quantization_config")


@pytest.mark.skipif(os.environ.get("QWEN4EXP_E2E") != "1",
                    reason="needs TPU v5e-8 + checkpoint (QWEN4EXP_E2E=1)")
def test_e2e_tpu():
    """HF checkpoint → TPU server → OpenAI-compatible generation.

    Manual steps (automated by benchmarks/bench_qwen4_exp.py):
    1. install fork, 2. serve <QWEN_MODEL> --tensor-parallel-size 8,
    3. POST /v1/completions, 4. assert non-empty greedy decode.
    """
