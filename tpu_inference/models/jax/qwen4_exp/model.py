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

"""Qwen4Exp full model + CausalLM + MTP stub (JAX / TPU).

Upstream references:
- ``vllm/models/qwen4_exp/nvidia/model.py``: ``Qwen4ExpModel`` (embed +
  repeat HC, delayed-combine layer loop, final mixer ``use_combine=False``,
  ``_mtp_hidden_buffer`` scheme A), ``Qwen4ExpForCausalLM``
  (``compute_logits`` via ``LogitsProcessor``, MTP target states, MRoPE
  positions, mamba specs/shapes/dtypes/copy funcs), ``Qwen4ExpMixtureOfExperts``
  (EPLB), ``Qwen4ExpForConditionalGeneration`` (vision tower — text path
  reused here).
- ``vllm/models/qwen4_exp/nvidia/mtp.py``: ``Qwen4ExpMultiTokenPredictor``
  / ``Qwen4ExpMTP`` (dual-stream sample/multi, PLE forced off, index-share).

TPU interface (must match ``tpu_inference`` runner contract):
- ``__init__(self, vllm_config, rng_key, mesh)``
- ``__call__(kv_caches, input_ids, attention_metadata, ...) ->
  (kv_caches, hidden | JaxIntermediateTensors, aux, expert_ids?)``
- ``compute_logits(hidden)``
See ``tpu_inference/models/jax/qwen3.py::Qwen3ForCausalLM`` (primary
skeleton) and ``models/common/model_loader.py::register_model`` contract.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

try:
    from tpu_inference.distributed.jax_parallel_state import get_pp_group
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata
    from tpu_inference.layers.jax import JaxModule
    from tpu_inference.layers.jax.embed import JaxEmbed
    from tpu_inference.layers.jax.linear import JaxLmHead
    from tpu_inference.layers.jax.norm import JaxRmsNorm
    from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
    from tpu_inference.models.jax.jax_intermediate_tensor import (
        JaxIntermediateTensors,
    )
    from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator
    _TPU_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only test fallback
    _TPU_AVAILABLE = False

    class JaxModule(nnx.Module):  # type: ignore[no-redef]
        pass

    class LoadableWithIterator:  # type: ignore[no-redef]
        pass

    JaxEmbed = None  # type: ignore[assignment]
    JaxLmHead = None  # type: ignore[assignment]
    JaxRmsNorm = None  # type: ignore[assignment]
    PPMissingLayer = None  # type: ignore[assignment]
    JaxIntermediateTensors = dict  # type: ignore[assignment]

    def get_pp_group():  # type: ignore[misc]
        class _G:
            is_first_rank = True
            is_last_rank = True

        return _G()

    def make_layers(n, fn):  # type: ignore[misc]
        return 0, n, [fn(i) for i in range(n)]

    AttentionMetadata = object  # type: ignore[assignment]

from .config import Qwen4ExpArch, arch_from_hf_config
from .hyperconnection import GatedResidual
from .layers import Qwen4ExpDecoderLayer

_init = nnx.initializers.uniform()


def _hf_config_of(vllm_config) -> object:
    mc = vllm_config.model_config
    return getattr(mc, "hf_config", getattr(mc, "hf_text_config", None))


class Qwen4ExpModel(JaxModule):
    def __init__(self, vllm_config, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str = "model") -> None:
        mc = vllm_config.model_config
        hf_config = _hf_config_of(vllm_config)
        vocab_size = mc.get_vocab_size() if hasattr(mc, "get_vocab_size") else 151936
        self.arch: Qwen4ExpArch = arch_from_hf_config(hf_config, vocab_size)
        self.dtype = mc.dtype if hasattr(mc, "dtype") else jnp.bfloat16
        self.mesh = mesh
        self.prefix = prefix
        H = self.arch.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=H,
                dtype=self.dtype,
                param_dtype=self.dtype,
                embedding_init=nnx.with_partitioning(_init, ("model", None)),
                rngs=rng,
                quant_config=getattr(vllm_config, "quant_config", None),
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        arch = self.arch
        self.start_layer, self.end_layer, self.layers = make_layers(
            arch.num_hidden_layers,
            lambda i: Qwen4ExpDecoderLayer(
                arch=arch, layer_idx=i, dtype=self.dtype, rngs=rng,
                prefix=f"{prefix}.layers.{i}"),
        )
        # Final mixer: mix-only (no combine), emits sample stream [T, H].
        self.mixer = GatedResidual(
            hidden_size=H, hc_count=arch.hc_count, hc_lowrank=arch.hc_lowrank,
            eps=arch.rms_norm_eps, dtype=self.dtype, rngs=rng,
            use_combine=False, prefix=prefix + ".hyper_connection_mixer")

    def embed_and_expand(self, input_ids: jax.Array) -> jax.Array:
        e = self.embed_tokens(input_ids)  # [T, H]
        # Checkpoint-native layout: repeat HC times, HC-outer/H-inner
        # (``.repeat(1, hc_count)`` upstream; equivalent to tile on last dim).
        return jnp.concatenate([e] * self.arch.hc_count, axis=-1)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata,
        inputs_embeds: Optional[jax.Array] = None,
        positions: Optional[jax.Array] = None,
        query_start_loc: Optional[jax.Array] = None,
        ngram_context: Optional[jax.Array] = None,
        token_to_req: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        md = attention_metadata
        if inputs_embeds is not None:
            x = inputs_embeds
            # Intermediate (non-first-rank) tensors already carry HC width.
        elif not self.is_first_rank:
            raise RuntimeError("Non-first PP rank requires inputs_embeds")
        else:
            assert input_ids is not None
            x = self.embed_and_expand(input_ids)

        if positions is None:
            positions = getattr(md, "input_positions", None)
        if query_start_loc is None:
            query_start_loc = getattr(md, "query_start_loc", None)

        hidden, prev_out, prev_inj = x, None, None
        aux: List[jax.Array] = []
        # Per-layer KV histories for the functional QSA path live outside
        # kv_caches in Phase 1 (documented gap: side-cache optimization).
        histories: dict = getattr(self, "_histories", {})
        for i, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]):
            gid = self.start_layer + i
            hist = histories.get(gid)
            if hist is None and layer.indexer is not None:
                hist = {"k": jnp.zeros((0, layer.arch.num_key_value_heads,
                                        layer.arch.head_dim), dtype=x.dtype),
                        "v": jnp.zeros((0, layer.arch.num_key_value_heads,
                                        layer.arch.head_dim), dtype=x.dtype),
                        "k_raw": jnp.zeros((0, layer.indexer.head_dim),
                                           dtype=x.dtype),
                        "pos": jnp.zeros((0,), dtype=jnp.int32)}
                histories[gid] = hist
            hidden, prev_out, prev_inj = layer.forward(
                hidden, prev_out, prev_inj, positions,
                input_ids=input_ids, query_start_loc=query_start_loc,
                ngram_context=ngram_context, token_to_req=token_to_req,
                kv_history=hist, attention_metadata=md, mesh=self.mesh)
        self._histories = histories
        if not self.is_last_rank:
            # Materialize pending tuple for PP handoff (upstream does the
            # same before returning IntermediateTensors on non-last ranks).
            if prev_out is not None:
                hidden = layer.mlp_hc.combine(hidden, prev_out, prev_inj)
            return kv_caches, hidden, aux
        _, sample, _ = self.mixer.combine_and_mix(hidden, prev_out, prev_inj) \
            if prev_out is not None else self.mixer.mix(hidden)
        # NOTE: MTP scheme-A multi-stream snapshot would stash ``hidden``
        # (pre-mixer [T, HC*H]) into _mtp_hidden_buffer here; the draft model
        # (Qwen4ExpMTP below) consumes it at spec_step_idx=0.
        return kv_caches, sample, aux


class Qwen4ExpForCausalLM(JaxModule, LoadableWithIterator):
    """Registered arch: ``Qwen4ExpForCausalLM`` (+ conditional-gen alias)."""

    def __init__(self, vllm_config, rng_key: jax.Array, mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = Qwen4ExpModel(vllm_config, rng, mesh, prefix="model")
        mc = vllm_config.model_config
        is_pooling = getattr(mc, "runner_type", "generate") == "pooling"
        tie = bool(getattr(_hf_config_of(vllm_config), "tie_word_embeddings", False))
        if not tie and not is_pooling and self.model.is_last_rank:
            vocab_size = mc.get_vocab_size() if hasattr(mc, "get_vocab_size") else 151936
            try:
                tp = vllm_config.parallel_config.tensor_parallel_size
            except AttributeError:
                tp = 1
            try:
                from tpu_inference import utils as _u
                vocab_size = _u.align_to(vocab_size, tp)
            except ImportError:
                pass
            self.lm_head = JaxLmHead(
                hidden_size=self.model.arch.hidden_size,
                vocab_size=vocab_size,
                dtype=self.model.dtype,
                param_dtype=self.model.dtype,
                rngs=rng,
                prefix="lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata,
        inputs_embeds: Optional[jax.Array] = None,
        intermediate_tensors=None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        **kwargs,
    ):
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        md = attention_metadata
        kv_caches, x, aux = self.model(
            kv_caches, input_ids, md, inputs_embeds,
            positions=getattr(md, "input_positions", None),
            query_start_loc=getattr(md, "query_start_loc", None),
            ngram_context=kwargs.get("ngram_context"),
            token_to_req=kwargs.get("token_to_req"),
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})
        return kv_caches, x, aux, None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if not isinstance(self.lm_head, PPMissingLayer):
            return self.lm_head(hidden_states)
        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)


class Qwen4ExpMTP(JaxModule):
    """Speculative draft model (arch ``Qwen4ExpMTP``).

    Upstream ``mtp.py``: text-only, PLE forced off, full-attention decoder
    layers from ``mtp_start_layer_idx``, dual-stream (sample, multi) I/O.
    Phase 1 provides a shape-correct stub sharing the target trunk config;
    full draft training/acceptance wiring follows the ``gemma4_mtp.py``
    pattern (tracked in docs as unsupported).
    """

    def __init__(self, vllm_config, rng_key: jax.Array, mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = Qwen4ExpModel(vllm_config, rng, mesh, prefix="model")


__all__ = ["Qwen4ExpForCausalLM", "Qwen4ExpMTP", "Qwen4ExpModel"]
