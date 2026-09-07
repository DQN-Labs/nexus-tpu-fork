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

"""Qwen4Exp cache / recurrent-state specifications.

Upstream references:
- ``vllm/models/qwen4_exp/common/qsa_cache.py``: ``QSAKeyStateCache``
  (circular ring, ``CircularBufferSpec``) + ``QSACompressedKeyCache``
  (``MLAAttentionSpec(tokens_per_state=ratio)``), slot-mapping helpers
  ``circular_qsa_slot_mapping`` / ``compressed_qsa_slot_mapping``,
  ``QSAMetadataBuilder``.
- ``vllm/models/qwen4_exp/nvidia/model.py``:
  ``get_mamba_specs_from_config`` (two ``MambaSpec``s: GDN state +
  PLE short-conv state), ``get_mamba_state_*`` dtype/shape/copy funcs.
- ``vllm/models/qwen4_exp/nvidia/model_state.py``: ``Qwen4ExpModelState``
  (rollback-safe ``ngram_context`` + ``ple_query_start_loc``).
- ``vllm/v1/kv_cache_interface.py``: ``FullAttentionSpec``,
  ``MambaSpec``, ``MLAAttentionSpec``, ``CircularBufferSpec``.

TPU mapping (``tpu_inference/runner/kv_cache.py`` owns physical blocks):
- Main KV: standard paged ``[num_blocks, block_size, nKV, head_dim]`` per
  full-attention layer (BF16; QSA layers share the same main cache — the
  sparse kernel gathers from it).
- QSA side caches: two extra per-QSA-layer caches with the slot rules below.
  Phase 1 may recompute indexer keys from the main cache (correct, extra
  compute); the side caches are the production optimization.
- GDN: conv state ``[R, C, K-1]`` + recurrent ``[R, V, Dk, Dv]``.
- PLE short-conv: ``[R, HC*H, (K-1)*ngram+1]`` (TP-replicated).
- ngram_context: ``[max_reqs, ngram_size-1]`` int32, EOS-padded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen4ExpCacheSpec:
    num_full_layers: int
    num_qsa_layers: int
    num_gdn_layers: int
    num_ple_layers: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    # QSA
    indexer_head_dim: int = 0
    indexer_compress_ratio: int = 1
    indexer_budget: int = 0
    num_spec_tokens: int = 0
    # GDN
    gdn_k_heads: int = 16
    gdn_v_heads: int = 32
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    gdn_conv_kernel: int = 4
    # PLE / HC
    hidden_size: int = 2048
    hc_count: int = 4
    ple_conv_kernel: int = 4
    ngram_size: int = 3
    dtype: str = "bfloat16"


def circular_capacity(compress_ratio: int, num_spec: int) -> int:
    """Ring capacity for raw indexer keys (upstream formula)."""
    import math

    return compress_ratio * math.ceil((compress_ratio + num_spec) / compress_ratio)


def circular_slot(position: int, capacity: int) -> int:
    """``circular_qsa_slot_mapping``: ring ``pos % capacity``."""
    return position % capacity


def compressed_slot(position: int, compress_ratio: int) -> int | None:
    """``compressed_qsa_slot_mapping``: only boundary rows stored.

    Returns ``pos // ratio`` iff ``(pos+1) % ratio == 0`` else None.
    """
    if (position + 1) % compress_ratio != 0:
        return None
    return position // compress_ratio


def short_conv_state_len(ple_conv_kernel: int, ngram_size: int) -> int:
    """``(K-1)*ngram + 1`` persistent history per channel."""
    return (ple_conv_kernel - 1) * ngram_size + 1


__all__ = [
    "Qwen4ExpCacheSpec",
    "circular_capacity",
    "circular_slot",
    "compressed_slot",
    "short_conv_state_len",
]
