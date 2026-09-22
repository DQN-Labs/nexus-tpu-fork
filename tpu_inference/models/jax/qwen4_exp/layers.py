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

"""Qwen4Exp decoder layer (JAX): PLE → HC-mix → attn → HC combine/mix → MoE.

Upstream reference: ``vllm/models/qwen4_exp/nvidia/model.py::
Qwen4ExpDecoderLayer`` — including the *delayed combine* optimization where
the MLP combine is consumed by the *next* layer's ``combine_and_mix``.

Forward contract (matches upstream ``forward`` exactly)::
    (hidden, prev_out, prev_inj) -> (hidden', mlp_out, inj')
where ``hidden`` is the multi-stream state ``[T, HC*H]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

try:
    from tpu_inference.layers.jax import JaxModule
except ImportError:  # pragma: no cover

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

from .attention import Qwen4ExpDenseAttention
from .config import Qwen4ExpArch, is_moe_layer
from .gdn import Qwen4ExpGDN
from .hyperconnection import GatedResidual
from .moe import Qwen4ExpMLP, Qwen4ExpMoE
from .ngram import Qwen4ExpPLE
from .qsa import (
    QSAIndexer,
    compress_keys_mean,
    qsa_select_indices,
    sparse_gqa,
)
from .attention import apply_partial_rope, gemma_rmsnorm_last_dim


class Qwen4ExpDecoderLayer(JaxModule):
    def __init__(
        self,
        arch: Qwen4ExpArch,
        layer_idx: int,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        prefix: str = "",
    ):
        self.arch = arch
        self.layer_idx = layer_idx
        self.layer_type = arch.layer_types[layer_idx]
        self.prefix = prefix
        self.dtype = dtype
        rngs = rngs or nnx.Rngs(layer_idx + 1)
        H = arch.hidden_size

        # PLE attached to 1-based ids in ple_layer_ids.
        # NOTE: each attribute below is assigned exactly once: modern flax
        # infers ``None`` as static metadata and rejects later reassignment
        # to a module, so the dead branch assigns ``None`` directly.
        if (layer_idx + 1) in set(arch.ple_layer_ids or []):
            order = sorted(set(arch.ple_layer_ids))
            dense_id = order.index(layer_idx + 1)
            self.ple = Qwen4ExpPLE(
                hidden_size=H,
                hc_count=arch.hc_count,
                ple_embed_dim=arch.ple_embed_dim,
                ngram_size=arch.ngram_size,
                heads_per_ngram=arch.heads_per_ngram,
                conv_kernel=arch.ple_conv_kernel_size,
                eps=arch.rms_norm_eps,
                ple_dense_layer_id=dense_id,
                unigram_vocab_size=arch.vocab_size,
                vocab_base=arch.ngram_vocab_size_base,
                divisible_by=arch.make_ngram_vocab_size_divisible_by,
                dtype=dtype,
                rngs=rngs,
                prefix=prefix + ".ple",
            )
        else:
            self.ple = None

        if self.layer_type == "linear_attention":
            self.linear_attn: Qwen4ExpGDN | None = Qwen4ExpGDN(
                hidden_size=H,
                num_k_heads=arch.linear_num_key_heads,
                num_v_heads=arch.linear_num_value_heads,
                k_head_dim=arch.linear_key_head_dim,
                v_head_dim=arch.linear_value_head_dim,
                conv_kernel=arch.linear_conv_kernel_dim,
                eps=arch.rms_norm_eps,
                output_gate_type=arch.output_gate_type,
                dtype=dtype,
                rngs=rngs,
                prefix=prefix + ".linear_attn",
            )
            self.self_attn = None
        elif self.layer_type == "full_attention":
            self.linear_attn = None
            # Qwen4Exp forces attn_output_gate=True (see nvidia/qsa.py).
            self.self_attn: Qwen4ExpDenseAttention | None = Qwen4ExpDenseAttention(
                hidden_size=H,
                num_heads=arch.num_attention_heads,
                num_kv_heads=arch.num_key_value_heads,
                head_dim=arch.head_dim,
                eps=arch.rms_norm_eps,
                rope_theta=arch.rope_theta,
                partial_rotary_factor=arch.partial_rotary_factor,
                attn_output_gate=True,
                dtype=dtype,
                rngs=rngs,
                prefix=prefix + ".self_attn",
            )
            # NOTE: the indexer lives UNDER self_attn (not as a layer-level
            # ``self.indexer``): the loader contract — and upstream vLLM —
            # names it ``...self_attn.indexer.index_qk.weight``, and flax
            # nnx derives param paths from attribute names (the ``prefix``
            # args above are inert labels). A layer-level attribute would
            # surface as ``...layers.N.indexer...`` and fail load loudly
            # (diagnosed 2026-09-22: v58 LOAD-FAIL on HC names, same cause).
            if arch.use_qsa:
                self.self_attn.indexer: QSAIndexer | None = QSAIndexer(
                    hidden_size=H,
                    indexer_n_heads=int(arch.indexer_n_heads),
                    indexer_head_dim=int(arch.indexer_head_dim),
                    eps=arch.rms_norm_eps,
                    rope_theta=arch.rope_theta,
                    partial_rotary_factor=arch.partial_rotary_factor,
                    main_head_dim=arch.head_dim,
                    indexer_budget=int(arch.indexer_budget),
                    indexer_compress_ratio=int(arch.indexer_compress_ratio),
                    dtype=dtype,
                    rngs=rngs,
                    prefix=prefix + ".self_attn.indexer",
                )
            else:
                self.self_attn.indexer = None
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer(arch, layer_idx):
            self.mlp: Qwen4ExpMoE | Qwen4ExpMLP = Qwen4ExpMoE(
                hidden_size=H,
                moe_intermediate_size=arch.moe_intermediate_size,
                shared_intermediate_size=arch.shared_expert_intermediate_size,
                num_experts=arch.num_experts,
                num_experts_per_tok=arch.num_experts_per_tok,
                norm_topk_prob=arch.norm_topk_prob,
                dtype=dtype,
                rngs=rngs,
                prefix=prefix + ".mlp",
            )
            self.is_moe = True
        else:
            self.mlp = Qwen4ExpMLP(
                hidden_size=H,
                intermediate_size=arch.intermediate_size,
                act=arch.hidden_act,
                dtype=dtype,
                rngs=rngs,
                prefix=prefix + ".mlp",
            )
            self.is_moe = False

        # NOTE: attribute names ARE the load contract — flax nnx derives
        # ``named_parameters()`` paths from them (``prefix`` args are inert
        # labels). These must read ``attn_hyper_connection`` /
        # ``mlp_hyper_connection`` to match the checkpoint + loader
        # (v58 died here with abbreviated ``attn_hc`` live names).
        self.attn_hyper_connection = GatedResidual(
            hidden_size=H, hc_count=arch.hc_count, hc_lowrank=arch.hc_lowrank,
            eps=arch.rms_norm_eps, dtype=dtype, rngs=rngs,
            prefix=prefix + ".attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(
            hidden_size=H, hc_count=arch.hc_count, hc_lowrank=arch.hc_lowrank,
            eps=arch.rms_norm_eps, dtype=dtype, rngs=rngs,
            prefix=prefix + ".mlp_hyper_connection")

    # -- attention dispatch -------------------------------------------
    def _full_attn(
        self,
        block_in: jax.Array,
        positions: jax.Array,
        kv_history: Optional[dict] = None,
    ) -> jax.Array:
        assert self.self_attn is not None
        if self.self_attn.indexer is None or kv_history is None:
            _, out = self.self_attn(block_in, positions)
            return out
        # QSA path: indexer select + sparse attend over main KV history.
        H, N, Kv, D = (self.arch.hidden_size, self.arch.num_attention_heads,
                       self.arch.num_key_value_heads, self.arch.head_dim)
        q, k, v, gate = self.self_attn.project(block_in)
        q, k = apply_partial_rope(q, k, positions, D,
                                  self.self_attn.rotary_dim,
                                  self.self_attn.rope_theta)
        # Append current k/v to history (logical order).
        kh = kv_history["k"]  # [S, Kv, D]
        vh = kv_history["v"]
        # History holds past tokens; current chunk appended for scoring.
        k_full = jnp.concatenate([kh, k], axis=0) if kh.shape[0] else k
        v_full = jnp.concatenate([vh, v], axis=0) if vh.shape[0] else v
        kv_history["k"], kv_history["v"] = k_full, v_full
        # Indexer queries + compressed keys.
        iq, k_raw = self.self_attn.indexer.project_qk(block_in, positions)
        ratio = int(self.arch.indexer_compress_ratio)
        budget = int(self.arch.indexer_budget)
        # Causally-correct compression needs the *full* raw-key history, not
        # just the current chunk (group g covers absolute tokens
        # [g*ratio, (g+1)*ratio)). The paged ring + compressed side caches
        # (cache.py slot rules) are the production optimization; Phase 1
        # retains the logical history and recomputes (correct, extra compute).
        raw_hist = kv_history.get("k_raw")
        k_raw_full = (
            jnp.concatenate([raw_hist, k_raw], axis=0)
            if raw_hist is not None and raw_hist.shape[0]
            else k_raw
        )
        kv_history["k_raw"] = k_raw_full
        pos_hist = kv_history.get("pos")
        pos_full = (
            jnp.concatenate([pos_hist, positions], axis=0)
            if pos_hist is not None and pos_hist.shape[0]
            else positions
        )
        kv_history["pos"] = pos_full
        pooled, first_pos = compress_keys_mean(k_raw_full, pos_full, ratio)
        # Norm pooled keys like queries then RoPE at first positions.
        if pooled.shape[0]:
            pooled_n = gemma_rmsnorm_last_dim(
                pooled[:, None, :].astype(block_in.dtype),
                self.self_attn.indexer.k_norm_w.value,
                self.self_attn.indexer.eps,
            )[:, 0, :]
            kc, _ = apply_partial_rope(
                pooled_n[:, None, :],
                pooled_n[:, None, :],
                first_pos,
                self.self_attn.indexer.head_dim,
                min(self.self_attn.indexer.main_rotary_dim,
                    self.self_attn.indexer.head_dim),
                self.self_attn.indexer.rope_theta,
            )
            kc = kc[:, 0, :]
        else:
            kc = pooled
        # Selection yields *absolute* logical rows into k_full/v_full.
        tok_idx, counts = qsa_select_indices(iq, kc.astype(block_in.dtype),
                                             positions, ratio, budget)
        o = sparse_gqa(q, k_full, v_full, tok_idx, counts, N, Kv)
        if gate is not None:
            o = o * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(o.dtype)
        w_o = self.self_attn.o_proj.weight.value.astype(jnp.float32)
        return jnp.einsum("TNH,NHD->TD", o.astype(jnp.float32), w_o).astype(
            block_in.dtype)

    def forward(
        self,
        hidden: jax.Array,  # [T, HC*H]
        prev_out: Optional[jax.Array],
        prev_inj: Optional[jax.Array],
        positions: jax.Array,
        input_ids: Optional[jax.Array] = None,
        query_start_loc: Optional[jax.Array] = None,
        ngram_context: Optional[jax.Array] = None,
        token_to_req: Optional[jax.Array] = None,
        kv_history: Optional[dict] = None,
        attention_metadata=None,
        mesh=None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        if self.ple is not None:
            if prev_out is not None:
                hidden = self.attn_hyper_connection.combine(
                    hidden, prev_out, prev_inj)
                prev_out = prev_inj = None
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden = hidden + self.ple(hidden, input_ids, query_start_loc,
                                       ngram_context, token_to_req)
        if prev_out is not None:
            hidden, block_in, inj = self.attn_hyper_connection.combine_and_mix(
                hidden, prev_out, prev_inj)
        else:
            hidden, block_in, inj = self.attn_hyper_connection.mix(hidden)

        if self.layer_type == "linear_attention":
            assert self.linear_attn is not None
            attn_out = self.linear_attn(block_in, attention_metadata, mesh)
        else:
            attn_out = self._full_attn(block_in, positions, kv_history)

        hidden, block_in, inj = self.mlp_hyper_connection.combine_and_mix(
            hidden, attn_out, inj)
        if self.is_moe:
            assert isinstance(self.mlp, Qwen4ExpMoE)
            mlp_out, _ = self.mlp(block_in)
        else:
            assert isinstance(self.mlp, Qwen4ExpMLP)
            mlp_out = self.mlp(block_in)
        return hidden, mlp_out, inj


__all__ = ["Qwen4ExpDecoderLayer"]
