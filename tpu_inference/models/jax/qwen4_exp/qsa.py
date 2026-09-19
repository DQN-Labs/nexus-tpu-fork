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

"""QSA (Qwen-style sparse attention) indexer + sparse attend, in JAX.

Upstream references:
- ``vllm/models/qwen4_exp/nvidia/indexer_qsa.py``: ``QSAIndexer``,
  ``apply_qsa_rope`` (fused norm+rope+compress+ring-update Triton path and
  the unfused reference).
- ``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py``: MQA paged scoring,
  top-k, expand kernels.
- ``vllm/models/qwen4_exp/nvidia/ops/qsa.py``: sparse paged GQA kernel
  (split-K + online softmax) and ``_expand_qsa_indices_kernel``.
- ``vllm/models/qwen4_exp/nvidia/qsa.py``: owner ``Qwen4ExpQSAAttention``
  (main QKV+gate projection, custom op ``qwen4_exp_qsa_with_output``).
- ``vllm/models/qwen4_exp/common/qsa_cache.py``: dual side caches
  (``QSAKeyStateCache`` circular ring + ``QSACompressedKeyCache``), slot
  mapping rules, metadata builder.

JAX/XLA translation (correctness-first):
- Indexer projection/norm/RoPE: plain einsums + ``attention.apply_partial_rope``.
- Compression: exact group-mean over ``compress_ratio`` raw keys
  (``_compress_qsa_groups_kernel`` semantics, including causal-tail
  handling for the open group).
- Scoring: ``logits[r,c] = sum_heads ReLU(q[r,h] . kc[c])`` then top-k over
  visible blocks (``budget/ratio`` blocks) — ``jnp.top_k`` replaces the C++
  persistent top-k.
- Expand: complete blocks expand to ``block*ratio+off`` tokens; the open
  (incomplete) group contributes its causal tail ``[tail_start..pos]``.
- Sparse attend: gather selected K/V rows and run masked softmax attention
  (mathematically identical to the Triton split-K kernel; slower but exact).

The paged physical layout (block tables, PDL) is handled by the TPU runner;
here indices are logical token positions so tests can run on CPU.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from ._jax_compat import JaxEinsum
from .attention import apply_partial_rope, gemma_rmsnorm_last_dim

_init = nnx.initializers.uniform()


class QSAIndexer(JaxModule):
    """Weight-free-scoring indexer: H -> (n_heads+1)*head_dim.

    Checkpoint: ``self_attn.indexer.index_qk_proj.weight`` plus
    ``q_layernorm`` / ``k_layernorm`` Gemma scales.
    ``indexer_kv_heads == 1`` is enforced by config validation.
    """

    def __init__(
        self,
        hidden_size: int,
        indexer_n_heads: int,
        indexer_head_dim: int,
        eps: float = 1e-6,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 0.25,
        main_head_dim: int = 256,
        indexer_budget: int | None = None,
        indexer_compress_ratio: int | None = None,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.n_heads = indexer_n_heads
        self.head_dim = indexer_head_dim
        self.eps = eps
        self.rope_theta = rope_theta
        # Indexer RoPE must cover the main attention rotary dim.
        self.main_rotary_dim = int(main_head_dim * partial_rotary_factor)
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.prefix = prefix
        rngs = rngs or nnx.Rngs(0)
        # NOTE: attribute is ``index_qk`` (not ``index_qk_proj``):
        # "index_qk_proj" contains the loader heuristic substring "k_proj",
        # which would route it into the 3D k/v path and crash construction.
        self.index_qk = JaxEinsum(
            "TD,DK->TK",
            (hidden_size, (indexer_n_heads + 1) * indexer_head_dim),
            param_dtype=jnp.float32,
            kernel_init=nnx.with_partitioning(_init, (None, "model")),
            rngs=rngs,
            prefix=prefix + ".index_qk",
        )
        self.q_norm_w = nnx.Param(jnp.zeros((indexer_head_dim,), dtype=jnp.float32))
        self.k_norm_w = nnx.Param(jnp.zeros((indexer_head_dim,), dtype=jnp.float32))

    @property
    def output_width(self) -> int:
        """Selection columns per row: ``budget + ratio - 1`` (upstream)."""
        assert self.indexer_budget is not None
        assert self.indexer_compress_ratio is not None
        return self.indexer_budget + self.indexer_compress_ratio - 1

    @property
    def packed_output_width(self) -> int:
        """Upstream packed width (trailing valid-count column)."""
        return self.output_width + 1

    def project_qk(
        self, hidden: jax.Array, positions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Return RoPE'd indexer queries [T,H,D] and raw keys [T,D]."""
        fused = jnp.einsum(
            "TD,DK->TK",
            hidden.astype(jnp.float32),
            self.index_qk.weight.value.astype(jnp.float32),
        )
        q_raw = fused[..., : self.n_heads * self.head_dim].reshape(
            *hidden.shape[:-1], self.n_heads, self.head_dim
        )
        k_raw = fused[..., self.n_heads * self.head_dim :].reshape(
            *hidden.shape[:-1], self.head_dim
        )
        q = gemma_rmsnorm_last_dim(q_raw, self.q_norm_w.value, self.eps)
        # Queries get RoPE at their own positions over the rotary prefix
        # (validated >= main rotary dim upstream); tail passes through.
        rotary_dim = min(self.main_rotary_dim, self.head_dim)
        q_rope, _ = apply_partial_rope(
            q,
            q,
            positions,
            self.head_dim,
            rotary_dim,
            self.rope_theta,
        )
        return q_rope.astype(hidden.dtype), k_raw.astype(hidden.dtype)


def compress_keys_mean(
    k_raw: jax.Array,
    positions: jax.Array,
    compress_ratio: int,
) -> Tuple[jax.Array, jax.Array]:
    """Group-mean compression over complete groups.

    Returns (compressed_keys [G,D], first_positions [G]).
    The trailing incomplete group has no compressed row (it is handled as the
    causal tail at select time), matching
    ``compressed_qsa_slot_mapping`` (only boundary rows are stored).
    """
    t = k_raw.shape[0]
    n_complete = t // compress_ratio
    if n_complete == 0:
        return (
            jnp.zeros((0,) + k_raw.shape[1:], dtype=k_raw.dtype),
            jnp.zeros((0,), dtype=jnp.int32),
        )
    trimmed = k_raw[: n_complete * compress_ratio]
    grouped = trimmed.reshape(n_complete, compress_ratio, *k_raw.shape[1:])
    pooled = jnp.mean(grouped, axis=1)
    first_pos = positions[: n_complete * compress_ratio].reshape(n_complete, compress_ratio)[
        :, 0
    ]
    return pooled, first_pos


def qsa_select_indices(
    q: jax.Array,  # [T, H, D] RoPE'd indexer queries
    kc_norm_rope: jax.Array,  # [G, D] normed+RoPE'd compressed keys
    positions: jax.Array,  # [T] absolute positions
    compress_ratio: int,
    budget: int,
) -> Tuple[jax.Array, jax.Array]:
    """Top-k block selection + expansion to token indices.

    Returns (token_indices [T, W] int32 with -1 padding, counts [T]).
    ``W = budget + ratio - 1`` (== upstream ``output_width``).
    """
    n_blocks = budget // compress_ratio
    width = budget + compress_ratio - 1
    t = q.shape[0]
    g = kc_norm_rope.shape[0]

    def score_row(qr):
        # qr: [H, D]; kc: [G, D]
        dots = jnp.einsum("HD,GD->HG", qr.astype(jnp.float32),
                          kc_norm_rope.astype(jnp.float32))
        return jnp.sum(jax.nn.relu(dots), axis=0)  # [G]

    if g == 0:
        # Only causal tail visible.
        idx = jnp.full((t, width), -1, dtype=jnp.int32)
        counts = jnp.minimum(positions + 1, width)
        for r in range(t):
            c = int(jnp.minimum(positions[r] + 1, width))
            idx = idx.at[r, :c].set(
                jnp.arange(positions[r] - c + 1, positions[r] + 1)
            )
        return idx, counts

    scores = jax.vmap(score_row)(q)  # [T, G]
    # Causal visibility: block b covers tokens [b*ratio,(b+1)*ratio); visible
    # iff its first token <= query position.
    block_first = jnp.arange(g) * compress_ratio
    visible = block_first[None, :] <= positions[:, None]
    scores = jnp.where(visible, scores, -jnp.inf)
    k = min(n_blocks, g)
    _, top_blocks = jax.lax.top_k(scores, k)  # [T, k]
    top_blocks = jnp.sort(top_blocks, axis=-1)

    # Expand complete blocks; append causal tail of the open group.
    idx = jnp.full((t, width), -1, dtype=jnp.int32)
    counts = jnp.zeros((t,), dtype=jnp.int32)
    for r in range(t):
        pos = int(positions[r])
        tail_start = (pos // compress_ratio) * compress_ratio
        # Complete blocks strictly before the open group.
        cols = []
        for b in range(k):
            blk = int(top_blocks[r, b])
            if blk * compress_ratio >= tail_start and blk * compress_ratio + compress_ratio > pos + 1:
                continue
            if blk * compress_ratio > pos:
                continue
            for off in range(compress_ratio):
                tok = blk * compress_ratio + off
                if tok <= pos:
                    cols.append(tok)
        for tok in range(tail_start, pos + 1):
            if tok not in cols:
                cols.append(tok)
        cols = cols[:width]
        idx = idx.at[r, : len(cols)].set(jnp.asarray(cols, dtype=jnp.int32))
        counts = counts.at[r].set(len(cols))
    return idx, counts


def sparse_gqa(
    q: jax.Array,  # [T, N, D] main queries (RoPE'd)
    k_cache: jax.Array,  # [S, K, D] full key history (logical order)
    v_cache: jax.Array,  # [S, K, D]
    token_indices: jax.Array,  # [T, W] logical positions, -1 padded
    counts: jax.Array,  # [T]
    num_heads: int,
    num_kv_heads: int,
) -> jax.Array:
    """Gather-selected softmax attention (exact, XLA-friendly)."""
    t, w = token_indices.shape
    scale = 1.0 / (q.shape[-1] ** 0.5)
    safe_idx = jnp.where(token_indices >= 0, token_indices, 0)
    # Gather: [T, W, K, D]
    kg = k_cache[safe_idx]
    vg = v_cache[safe_idx]
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        kg = jnp.repeat(kg, rep, axis=2)
        vg = jnp.repeat(vg, rep, axis=2)
    logits = jnp.einsum("TNH,TWNH->NTW", q.astype(jnp.float32),
                        kg.astype(jnp.float32)) * scale
    # logits are [N, T, W]; the validity mask is per (token, slot).
    valid = (jnp.arange(w)[None, :] < counts[:, None])[None, :, :]
    logits = jnp.where(valid, logits, -1e9)
    probs = jax.nn.softmax(logits, axis=-1).astype(q.dtype)
    return jnp.einsum("NTW,TWNH->TNH", probs, vg)


__all__ = [
    "QSAIndexer",
    "compress_keys_mean",
    "qsa_select_indices",
    "sparse_gqa",
]
