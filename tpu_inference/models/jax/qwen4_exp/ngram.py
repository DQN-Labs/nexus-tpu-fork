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

"""PLE / n-gram embedding layer for Qwen4Exp, in JAX.

Upstream references:
- ``vllm/models/qwen4_exp/nvidia/ple_layer.py``: ``Qwen4ExpNGramEmbedding``
  (hash → embed → dequant), ``Qwen4ExpPLELayer`` (merged kv_proj, grouped
  norms, ``ple_gate`` math, dilated short-conv with persistent conv_state).
- ``vllm/models/qwen4_exp/nvidia/ops/ple.py``: ``ple_ngram_ids`` (hash),
  ``ple_gate`` (fused RMS/dot/sigmoid/second-norm), ``ple_conv``.
- ``vllm/models/qwen4_exp/common/ple.py``: ``PLEShardOverlap``,
  ``copy_ple_embedding_shard_``, ``PLEVocabParallelEmbedding``.
- ``vllm/models/qwen4_exp/nvidia/model_state.py``: ``ngram_context``
  (rollback-safe last ``ngram_size-1`` tokens per request).

Hash (must match exactly — it determines embedding rows):
    mult[i] = 2*(splitmix64(seed + 10007*ple_layer + gamma*(i+1)) % half)+1
    sizes/offsets: per-head prime vocab sizes derived from
      ``ngram_vocab_size_base`` (see ``_prime_sizes``).
    mixed[t,h] = tok[t]*m0 XOR tok[t-1]*m1 ... (older tokens EOS-clamped,
      pre-chunk history from ``ngram_context``)
    id[t,h] = remainder(mixed, size[h]) + offset[h]   # torch.remainder sign

Gate (``ple_gate``):
    kn = RMS(k)/per-H group; qn = RMS(hidden)/per-H group
    d = dot(kn, qn)/sqrt(H); g = sigmoid(sign(d)*sqrt(max(|d|, 1e-6)))
    gated = g * v                                     # [T, HC*H]
    conv_in = RMS(gated)
    out += silu(sum_k W[k]*history[t + k*dilation])   # dilation=ngram_size
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from ._jax_compat import JaxEinsum, JaxEmbed


class PLEFp8Table(JaxModule):
    """FP8 n-gram embedding table with dequant-on-lookup.

    Params: ``weight`` fp8-e4m3 ``[rows, head_dim]`` + ``table_scale``
    fp32 scalar (the export's single global ``weight_scale``). Forward
    gathers rows, widens to fp32 exactly, and scales. Rows shard over the
    ``model`` TP axis like a vocabulary embedding.
    """

    def __init__(self, num_embeddings: int, features: int, rngs=None,
                 prefix: str = "") -> None:
        self.num_embeddings = num_embeddings
        self.features = features
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        del rngs  # values come from the checkpoint, never random
        self.weight = nnx.Param(
            jnp.zeros((num_embeddings, features), dtype=jnp.float8_e4m3fn))
        self.table_scale = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def __call__(self, ids: jax.Array) -> jax.Array:
        rows = self.weight.value[ids]
        return rows.astype(jnp.float32) * self.table_scale.value

_init = nnx.initializers.uniform()
MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def ple_multipliers(
    seed: int,
    ple_dense_layer_id: int,
    ngram_size: int,
    unigram_vocab_size: int,
) -> List[int]:
    """Deterministic hash multipliers for one PLE layer (exact upstream port).

    Upstream ``Qwen4ExpNGramEmbedding._make_layer_multipliers``:
        half_bound = max(1, (((2**63 - 1) // vocab) // 2))
        base = seed + 10007 * ple_dense_layer_id
        mult[i] = 2 * (splitmix64(base + gamma * (i+1)) % half_bound) + 1
    """
    gamma = 0x9E3779B97F4A7C15
    half_bound = max(1, (((1 << 63) - 1) // int(unigram_vocab_size)) // 2)
    base_seed = int(seed) + 10007 * int(ple_dense_layer_id)
    mults = []
    for i in range(ngram_size):
        # No pre-masking: splitmix64 masks internally after adding gamma,
        # so masking here would change the result for i >= 1 (gamma*(i+1)
        # exceeds 2**64). Python big-int arithmetic preserves the wrap.
        h = splitmix64(base_seed + gamma * (i + 1))
        mults.append(2 * (h % half_bound) + 1)
    return mults


def _is_prime_64(value: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit ints (exact upstream port)."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    """The ``count``-th prime strictly greater than ``start`` (upstream)."""
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def ple_vocab_sizes_offsets(
    ngram_size: int,
    heads_per_ngram: int,
    base: int,
    divisible_by: int,
    ple_dense_layer_id: int = 0,
) -> Tuple[List[int], List[int]]:
    """Per-head prime sizes + row offsets (exact upstream port).

    Upstream ``_make_vocab_layout``: for local head ``h``,
    ``global_h = ple_dense_layer_id * ngram_heads + h`` and
    ``size = nth_prime_after(base - 1, global_h + 1)``. Offsets are the
    unpadded cumsum; the embedding table pads only the *total* up to
    ``divisible_by`` (trailing padding rows, never looked up).

    Returns PLAIN PYTHON INT LISTS (not JAX arrays): sizes feed shape
    computation (``ple_padded_rows`` -> ``nnx.Embed``), which must stay
    concrete under ``nnx.eval_shape``/jit tracing (v54 died on
    ``int(jnp.sum(...))``). Callers convert to arrays where values (not
    shapes) are needed.
    """
    n_heads = (ngram_size - 1) * heads_per_ngram
    sizes_l = [
        _nth_prime_after(
            int(base) - 1, int(ple_dense_layer_id) * n_heads + h + 1
        )
        for h in range(n_heads)
    ]
    offsets_l = []
    running = 0
    for s in sizes_l:
        offsets_l.append(running)
        running += s
    return sizes_l, offsets_l


def ple_padded_rows(sizes, divisible_by: int) -> int:
    """Total embedding rows: ``ceil(sum(sizes) / div) * div`` (upstream).

    Pure Python (see above): must stay concrete under tracing.
    """
    total = int(sum(int(v) for v in list(sizes)))
    div = int(divisible_by)
    return ((total + div - 1) // div) * div


_MASK32 = (1 << 32) - 1
_MASK16 = (1 << 16) - 1


def _u32_add3(x: jax.Array, y: jax.Array, z: jax.Array):
    """Mod-2^32 sum of three u32 arrays + total carry (0..2).

    All ops wrap-safe: each add's carry is detected by comparison.
    """
    s1 = x + y
    c1 = (s1 < x).astype(jnp.uint32)
    s2 = s1 + z
    c2 = (s2 < s1).astype(jnp.uint32)
    return s2, c1 + c2


def _u64_mul(a_lo: jax.Array, a_hi: jax.Array,
             b_lo: jax.Array, b_hi: jax.Array):
    """Low 64 bits of a 64x64-bit product (u32 limb pairs, exact).

    Operands as (lo, hi) u32 pairs; returns (lo, hi) u32 pair holding the
    product mod 2**64. No 64-bit arithmetic anywhere (TPU x64-free).
    """
    a0 = a_lo & _MASK16
    a1 = (a_lo >> 16) & _MASK16
    a2 = a_hi & _MASK16
    a3 = (a_hi >> 16) & _MASK16
    b0 = b_lo & _MASK16
    b1 = (b_lo >> 16) & _MASK16
    b2 = b_hi & _MASK16
    b3 = (b_hi >> 16) & _MASK16
    # result limbs r0..r3 (16 bits each) with carry propagation.
    # Discipline: the third arg of _u32_add3 is a genuine addend (the small
    # incoming 2**16-granularity carry ``c``). A returned carry ``t`` counts
    # 2**32-multiples and must NEVER be fed back as an addend (that was a
    # real bug: silent +t poison); it is added to the outgoing count.
    r0_full = a0 * b0
    r0 = r0_full & _MASK16
    c = r0_full >> 16
    s, t = _u32_add3(a0 * b1, a1 * b0, c)
    r1 = s & _MASK16
    c = (s >> 16) + (t << 16)
    s, t = _u32_add3(a0 * b2, a1 * b1, c)
    s, t2 = _u32_add3(s, a2 * b0, 0)
    t2 = t2 + t
    r2 = s & _MASK16
    c = (s >> 16) + (t2 << 16)
    s, t = _u32_add3(a0 * b3, a1 * b2, c)
    s, t2 = _u32_add3(s, a2 * b1, 0)
    t2 = t2 + t
    s, t3 = _u32_add3(s, a3 * b0, 0)
    t3 = t3 + t2
    r3 = s & _MASK16
    lo = r0 | (r1 << 16)
    hi = r2 | (r3 << 16)
    _ = t3  # overflow past bit 64 is discarded (mod 2**64)
    return lo, hi


def _u64_mod(lo: jax.Array, hi: jax.Array, d: jax.Array) -> jax.Array:
    """u64 (lo, hi) mod u32 d (requires d < 2**31; sizes qualify).

    (hi * 2**32 + lo) % d via pow2 precomputation (overflow-safe doubling)
    and Russian-peasant mulmod (adds stay < 2d). All u32 exact.
    """
    p = jnp.ones_like(d)
    for _ in range(32):
        dbl = p + p
        p = jnp.where(dbl >= d, dbl - d, dbl)
    # p == 2**32 % d (d < 2**31 keeps every double exact).
    res = jnp.zeros_like(d)
    a = hi % d
    b = p
    for _ in range(32):
        odd = (b & 1) == 1
        res = jnp.where(odd, res + a, res)
        res = jnp.where(res >= d, res - d, res)
        a = a + a
        a = jnp.where(a >= d, a - d, a)
        b = b >> 1
    res = res + (lo % d)
    return jnp.where(res >= d, res - d, res)


def compute_ngram_ids(
    input_ids: jax.Array,  # [T] int32/64
    query_start_loc: jax.Array,  # [B+1] token offsets per sequence
    ngram_context: jax.Array,  # [R, ngram_size-1] history (EOS-padded)
    token_to_req: jax.Array,  # [T] request index per token
    sizes: jax.Array,  # [H] per-head vocab sizes (< 2**31, int32)
    offsets: jax.Array,  # [H] per-head row offsets (int32)
    multipliers: List[int],
    ngram_size: int,
    eos_id: int = 0,
) -> jax.Array:
    """Exact n-gram hash ids [T, H] (int32), fully XLA-traceable.

    Same math as the eager reference (per-order XOR-mix of
    ``hist[i] * multipliers[i]`` mod 2**64, then ``% sizes + offsets``),
    but expressed with static shapes/indices only: no ``int()`` on traced
    values, no Python loops over data. The 64-bit hash is emulated with
    u32 limb pairs (``_u64_mul``/``_u64_mod``) so the result is bit-exact
    without requiring JAX x64 (unavailable/unsuitable on TPU).

    ``ngram_size``/shapes must be static (they are: attributes/shapes).
    """
    n = ngram_size
    t = input_ids.shape[0]
    n_heads = sizes.shape[0]
    heads_per_ngram = n_heads // max(n - 1, 1)
    n_ctx = ngram_context.shape[0]
    pos = jnp.arange(t)
    # Sequence start per token via the start-loc table (static search).
    b = jnp.clip(
        jnp.searchsorted(query_start_loc, pos, side="right") - 1, 0,
        query_start_loc.shape[0] - 2)
    seq_start = query_start_loc[b]
    rel = pos - seq_start
    req = jnp.clip(token_to_req, 0, max(n_ctx - 1, 0))
    # History matrix [T, n]: current + previous tokens, else context tail.
    backs = jnp.arange(n)
    use_tok = backs[None, :] <= rel[:, None]
    tok_idx = jnp.clip(pos[:, None] - backs[None, :], 0, max(t - 1, 0))
    gathered_tok = input_ids[tok_idx]
    need = backs[None, :] - rel[:, None]  # 1-based tail index (>=1 off-token)
    crow = ngram_context[req]  # [T, n-1]
    # Context row has n-1 entries; 1-based need -> index (n-1)-need.
    ctx_idx = jnp.clip((n - 1) - need, 0, max(n - 2, 0))
    gathered_ctx = jnp.take_along_axis(crow, ctx_idx, axis=1)
    req_ok = (token_to_req >= 0) & (token_to_req < n_ctx)
    gathered_ctx = jnp.where(req_ok[:, None], gathered_ctx, eos_id)
    hist = jnp.where(use_tok, gathered_tok,
                     gathered_ctx).astype(jnp.uint32)  # [T, n]
    mults_lo = jnp.asarray([m & _MASK32 for m in multipliers],
                           dtype=jnp.uint32)
    mults_hi = jnp.asarray([(m >> 32) & _MASK32 for m in multipliers],
                           dtype=jnp.uint32)
    outs = []
    for order in range(2, n + 1):
        vlo = jnp.zeros((t,), dtype=jnp.uint32)
        vhi = jnp.zeros((t,), dtype=jnp.uint32)
        for i in range(order):
            a_lo, a_hi = hist[:, i], jnp.zeros((t,), dtype=jnp.uint32)
            p_lo, p_hi = _u64_mul(a_lo, a_hi, mults_lo[i], mults_hi[i])
            vlo, vhi = vlo ^ p_lo, vhi ^ p_hi
        base_h = (order - 2) * heads_per_ngram
        d = sizes[base_h:base_h + heads_per_ngram].astype(jnp.uint32)
        o = offsets[base_h:base_h + heads_per_ngram].astype(jnp.uint32)
        r = _u64_mod(
            jnp.broadcast_to(vlo[:, None], (t, heads_per_ngram)),
            jnp.broadcast_to(vhi[:, None], (t, heads_per_ngram)),
            jnp.broadcast_to(d[None, :], (t, heads_per_ngram)),
        )
        outs.append((r + o[None, :]).astype(jnp.int32))
    return jnp.concatenate(outs, axis=1)


class Qwen4ExpPLE(JaxModule):
    """PLE layer: n-gram embed → gate → dilated short-conv → add to HC state."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        ple_embed_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        conv_kernel: int = 4,
        eps: float = 1e-6,
        seed: int = 1234,
        ple_dense_layer_id: int = 0,
        unigram_vocab_size: int | None = None,
        vocab_base: int = 20_000_000,
        divisible_by: int = 128,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.ple_embed_dim = ple_embed_dim
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.conv_kernel = conv_kernel
        self.eps = eps
        self.ple_dense_layer_id = ple_dense_layer_id
        self.prefix = prefix
        if unigram_vocab_size is None:
            raise ValueError(
                "unigram_vocab_size (config vocab_size) is required: it sets "
                "the splitmix64 multiplier bound. Pass arch.vocab_size.")
        rngs = rngs or nnx.Rngs(0)
        self.multipliers = ple_multipliers(
            seed, ple_dense_layer_id, ngram_size, unigram_vocab_size)
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        if ple_embed_dim % ngram_heads:
            raise ValueError(
                f"ple_embed_dim must be divisible by total ngram heads: "
                f"{ple_embed_dim} % {ngram_heads} != 0")
        self.ngram_heads = ngram_heads
        self.head_dim = ple_embed_dim // ngram_heads
        sizes, offsets = ple_vocab_sizes_offsets(
            ngram_size, heads_per_ngram, vocab_base, divisible_by,
            ple_dense_layer_id,
        )
        self.sizes = jnp.asarray(sizes, dtype=jnp.int32)
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        total_rows = ple_padded_rows(sizes, divisible_by)
        # INT4-residency serving: the n-gram table (320M rows x head_dim,
        # ~51 GB as fp8) lives in HBM as fp8 codes + one global fp32 scale
        # and is dequantized per lookup. A full-precision copy (102-205 GB)
        # cannot fit v5e-8 HBM alongside the body. JAX name stays
        # ``...ple.embedding.weight`` (fp8) + ``...ple.embedding.table_scale``.
        self.embedding = PLEFp8Table(
            num_embeddings=total_rows,
            features=self.head_dim,
            rngs=rngs,
            prefix=prefix + ".embedding",
        )
        wide = hidden_size * hc_count
        # Merged kv: [ple_embed_dim] -> [HC*H + H]. NOTE: attribute is ``kv``
        # (not ``kv_proj``): "kv_proj" contains the loader heuristic
        # substring "v_proj", which would route it into the 3D k/v path and
        # crash construction. JAX name: ``...ple.kv.weight``.
        self.kv = JaxEinsum(
            "TD,DK->TK", (ple_embed_dim, wide + hidden_size),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")), rngs=rngs,
            prefix=prefix + ".kv")
        self.norm_key_w = nnx.Param(jnp.zeros((wide,), dtype=jnp.float32))
        self.norm_query_w = nnx.Param(jnp.zeros((wide,), dtype=jnp.float32))
        self.norm_conv_w = nnx.Param(jnp.zeros((wide,), dtype=jnp.float32))
        # Dilated depthwise conv weight, zero-init (upstream _no_reinit).
        self.conv_w = nnx.Param(jnp.zeros((wide, conv_kernel), dtype=jnp.float32))

    def gate(
        self, hidden: jax.Array, kv: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Return (gated [T,HC*H], conv_in [T,HC*H])."""
        wide = self.hidden_size * self.hc_count
        h = self.hidden_size
        k = kv[..., :wide]
        v = kv[..., wide : wide + h]
        # v is [T, H]; broadcast over HC streams to [T, HC*H].
        v = jnp.concatenate([v] * self.hc_count, axis=-1)
        # Grouped RMS over per-H groups.
        def grms(x, w):
            xf = x.astype(jnp.float32).reshape(*x.shape[:-1], self.hc_count, h)
            var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
            n = (xf * jax.lax.rsqrt(var + self.eps)).reshape(x.shape)
            return (n * (1.0 + w.astype(jnp.float32))).astype(x.dtype)

        kn = grms(k, self.norm_key_w.value).astype(jnp.float32)
        qn = grms(hidden, self.norm_query_w.value).astype(jnp.float32)
        d = jnp.sum(kn * qn, axis=-1) / (h**0.5)
        g = jax.nn.sigmoid(
            jnp.sign(d) * jnp.sqrt(jnp.maximum(jnp.abs(d), 1e-6))
        )[..., None]
        gated = (g * v.astype(jnp.float32)).astype(hidden.dtype)
        conv_in = grms(gated, self.norm_conv_w.value)
        return gated, conv_in

    def dilated_conv(
        self, conv_in: jax.Array, conv_state: Optional[jax.Array] = None
    ) -> jax.Array:
        # Depthwise dilated conv1d, dilation=ngram_size, causal.
        # conv_in: [T, C]; weight [C, K] zero-init at start of training.
        # Gather-based (no per-token Python loop): the loop unrolls T times
        # under jit and explodes compile time on long prefills.
        w = self.conv_w.value.astype(jnp.float32)  # [C, K]
        k, dil = self.conv_kernel, self.ngram_size
        hist_len = (k - 1) * dil
        if conv_state is not None:
            # Prepend persistent history [C, hist_len].
            hist = conv_state.astype(jnp.float32).T  # [hist_len?, C]
            full = jnp.concatenate([hist[-hist_len:], conv_in.astype(jnp.float32)])
        else:
            full = jnp.pad(conv_in.astype(jnp.float32), ((hist_len, 0), (0, 0)))
        t = conv_in.shape[0]
        rows = (jnp.arange(t)[:, None] + hist_len
                - jnp.arange(k)[None, :] * dil)  # [T, K]
        gather = full[rows]  # [T, K, C]
        acc = jnp.einsum("CK,TKC->TC", w, gather)
        return jax.nn.silu(acc).astype(conv_in.dtype)

    def __call__(
        self,
        hidden_hc: jax.Array,  # [T, HC*H] multi-stream state
        input_ids: jax.Array,
        query_start_loc: jax.Array,
        ngram_context: jax.Array,
        token_to_req: jax.Array,
        conv_state: Optional[jax.Array] = None,
    ) -> jax.Array:
        ids = compute_ngram_ids(
            input_ids, query_start_loc, ngram_context, token_to_req,
            jnp.asarray(self.sizes), jnp.asarray(self.offsets),
            self.multipliers, self.ngram_size,
        )
        # [T, ngram_heads] -> per-head [head_dim] rows -> concat to
        # [T, ple_embed_dim] (upstream ``flatten(-2)`` over heads).
        e = self.embedding(ids.reshape(-1)).reshape(
            input_ids.shape[0], self.ngram_heads * self.head_dim
        )
        emb = e.astype(hidden_hc.dtype)
        kv = jnp.einsum(
            "TD,DK->TK", emb.astype(jnp.float32),
            self.kv.weight.value.astype(jnp.float32)).astype(hidden_hc.dtype)
        gated, conv_in = self.gate(hidden_hc, kv)
        # Upstream accumulates the dilated short-conv of conv_in directly
        # into the multi-stream state (see ple_conv kernel).
        return self.dilated_conv(conv_in, conv_state)


__all__ = [
    "Qwen4ExpPLE",
    "compute_ngram_ids",
    "ple_multipliers",
    "ple_padded_rows",
    "ple_vocab_sizes_offsets",
    "splitmix64",
]
