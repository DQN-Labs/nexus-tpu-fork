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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-head prime sizes + row offsets (exact upstream port).

    Upstream ``_make_vocab_layout``: for local head ``h``,
    ``global_h = ple_dense_layer_id * ngram_heads + h`` and
    ``size = nth_prime_after(base - 1, global_h + 1)``. Offsets are the
    unpadded cumsum; the embedding table pads only the *total* up to
    ``divisible_by`` (trailing padding rows, never looked up).
    """
    n_heads = (ngram_size - 1) * heads_per_ngram
    sizes_l = [
        _nth_prime_after(
            int(base) - 1, int(ple_dense_layer_id) * n_heads + h + 1
        )
        for h in range(n_heads)
    ]
    # int32 suffices (base ~2e7) and keeps JAX x64-disabled configs happy.
    sizes = jnp.asarray(sizes_l, dtype=jnp.int32)
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(sizes)[:-1]]
    )
    return sizes, offsets


def ple_padded_rows(sizes: jnp.ndarray, divisible_by: int) -> int:
    """Total embedding rows: ``ceil(sum(sizes) / div) * div`` (upstream)."""
    total = int(jnp.sum(sizes))
    div = int(divisible_by)
    return ((total + div - 1) // div) * div


def compute_ngram_ids(
    input_ids: jax.Array,  # [T] int32/64
    query_start_loc: jax.Array,  # [B+1] token offsets per sequence
    ngram_context: jax.Array,  # [R, ngram_size-1] history (EOS-padded)
    token_to_req: jax.Array,  # [T] request index per token
    sizes: jax.Array,  # [H] per-head vocab sizes
    offsets: jax.Array,  # [H] per-head row offsets
    multipliers: List[int],
    ngram_size: int,
    eos_id: int = 0,
) -> jax.Array:
    """Exact n-gram hash ids [T, H] (int32).

    Hash arithmetic uses Python ints (uint64 wrap) so the result is
    bit-exact regardless of JAX x64 mode; only the final ids are JAX arrays.
    """
    t = int(input_ids.shape[0])
    n_heads = int(sizes.shape[0])
    heads_per_ngram = n_heads // max(int(ngram_size) - 1, 1)
    # Gather per-token history: current + previous ngram_size-1 tokens,
    # with sequence-start clamping to EOS and cross-chunk ngram_context.
    toks = [int(v) for v in list(input_ids.reshape(-1))]
    qsl = [int(v) for v in list(query_start_loc.reshape(-1))]
    t2r = [int(v) for v in list(token_to_req.reshape(-1))]
    ctx = [[int(v) for v in list(row)] for row in list(ngram_context.reshape(
        ngram_context.shape[0], -1))]
    sizes_l = [int(v) for v in list(sizes.reshape(-1))]
    offs_l = [int(v) for v in list(offsets.reshape(-1))]
    out_rows = []
    for pos in range(t):
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
                need = back - (pos - seq_start)  # 1-based into context tail
                crow = ctx[req] if 0 <= req < len(ctx) else []
                hist.append(crow[len(crow) - need] if 0 < need <= len(crow) else eos_id)
        # Per-order mixing (upstream eager path): order n mixes only the
        # first n history tokens and fills only that order's heads.
        # torch.remainder semantics with positive divisor == Python % here.
        row = []
        for order in range(2, int(ngram_size) + 1):
            val = 0
            for i in range(order):
                val = (val ^ ((hist[i] & MASK64) * (multipliers[i] & MASK64))) & MASK64
            base_h = (order - 2) * heads_per_ngram
            for h in range(heads_per_ngram):
                hh = base_h + h
                row.append((val % sizes_l[hh]) + offs_l[hh])
        out_rows.append(row)
    return jnp.asarray(out_rows, dtype=jnp.int32)


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
        self.sizes = jnp.asarray(sizes)
        self.offsets = jnp.asarray(offsets)
        total_rows = ple_padded_rows(sizes, divisible_by)
        # Per-head rows of width head_dim; heads are concatenated (flatten)
        # to [T, ple_embed_dim], matching upstream
        # ``ngram_embedding(ngram_ids).flatten(-2)``.
        self.embedding = JaxEmbed(
            num_embeddings=total_rows,
            features=self.head_dim,
            param_dtype=jnp.float32,
            embedding_init=nnx.with_partitioning(_init, ("model", None)),
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
        w = self.conv_w.value.astype(jnp.float32)  # [C, K]
        k, dil = self.conv_kernel, self.ngram_size
        hist_len = (k - 1) * dil
        if conv_state is not None:
            # Prepend persistent history [C, hist_len].
            hist = conv_state.astype(jnp.float32).T  # [hist_len?, C]
            full = jnp.concatenate([hist[-hist_len:], conv_in.astype(jnp.float32)])
        else:
            full = jnp.pad(conv_in.astype(jnp.float32), ((hist_len, 0), (0, 0)))
        outs = []
        for t in range(conv_in.shape[0]):
            acc = jnp.zeros((conv_in.shape[1],), dtype=jnp.float32)
            for j in range(k):
                acc = acc + w[:, j] * full[t + hist_len - j * dil, :]
            outs.append(acc)
        return jax.nn.silu(jnp.stack(outs)).astype(conv_in.dtype)

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
