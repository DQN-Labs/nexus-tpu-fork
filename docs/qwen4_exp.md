# Qwen4Exp (Qwen3.8-Flash-Next) on TPU — architecture mapping

> Phase 1 deliverable (task §10). Every row maps an upstream vLLM
> class/function to its JAX/TPU counterpart in this fork. Upstream is the
> behavioral reference; the JAX column preserves tensor shapes, parameter
> structure, weight naming, config semantics, and forward-pass ordering.

Upstream commit surveyed: `vllm-project/vllm` (2026-09-04, shallow HEAD
`701a7444909d`). TPU base: `vllm-project/tpu-inference` HEAD `c824927`.

## Upstream file inventory (canonical)

```
vllm/models/qwen4_exp/__init__.py                  platform dispatch (TPU gap filled here)
vllm/models/qwen4_exp/config.py                    Qwen4ExpTextConfig / Qwen4ExpConfig
vllm/models/qwen4_exp/common/hyperconnection.py    portable GatedResidual + GroupedGemmaRMSNorm
vllm/models/qwen4_exp/common/ple.py                PLEShardOverlap, copy_ple_embedding_shard_, PLEVocabParallelEmbedding
vllm/models/qwen4_exp/common/qsa_cache.py          QSA side-cache specs, slot mappings, metadata builder
vllm/models/qwen4_exp/nvidia/model.py              Qwen4ExpModel / ForCausalLM / ForConditionalGeneration
vllm/models/qwen4_exp/nvidia/qsa.py                Qwen4ExpQSAAttention + FA backend + custom op
vllm/models/qwen4_exp/nvidia/indexer_qsa.py        QSAIndexer, apply_qsa_rope
vllm/models/qwen4_exp/nvidia/ple_layer.py          Qwen4ExpNGramEmbedding + Qwen4ExpPLELayer
vllm/models/qwen4_exp/nvidia/hyperconnection.py    GatedResidual w/ delayed combine
vllm/models/qwen4_exp/nvidia/mtp.py                Qwen4ExpMultiTokenPredictor + Qwen4ExpMTP
vllm/models/qwen4_exp/nvidia/model_state.py        Qwen4ExpModelState (ngram context)
vllm/models/qwen4_exp/nvidia/low_latency_gemm.py   skinny-GEMM decode dispatch
vllm/models/qwen4_exp/nvidia/ops/hc.py             5 Triton HC kernels
vllm/models/qwen4_exp/nvidia/ops/ple.py            ngram-ids + gate + dilated conv kernels
vllm/models/qwen4_exp/nvidia/ops/qsa.py            sparse paged GQA + compress + store kernels
vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py    MQA paged score + topk + expand kernels
vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py fused norm+rope+compress+ring-update
vllm/model_executor/models/qwen3_next.py           Qwen3NextAttention / MLP / SparseMoeBlock (reused)
vllm/model_executor/models/qwen3_5.py              Qwen3_5Model.hf_to_vllm_mapper base + vision tower
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py  QwenGatedDeltaNetAttention
vllm/transformers_utils/configs/qwen3_next.py      Qwen3NextConfig base
```

## Class/function mapping

| Upstream vLLM | JAX/TPU in this fork | Notes |
|---|---|---|
| `Qwen4ExpTextConfig` (`config.py`) | `qwen4_exp/config.py::Qwen4ExpArch` + `arch_from_hf_config` + `validate_qwen4exp_text_config` | Same defaults/validators (hc_count>1, ple divisibility, all-or-none QSA, budget/ratio ∈ {512,2048}, indexer_head_dim ≥ rotary_dim); reads flat or nested `text_config` |
| `Qwen4ExpConfig` | same module (`get_text_config`) | `model_type="qwen4_exp"`, vision keys passed through; text-only serving supported |
| `GroupedGemmaRMSNorm` (`common/hyperconnection.py`) | `hyperconnection.py::grouped_gemma_rmsnorm` + `GroupedGemmaRMSNorm` | Identical math `x/sqrt(mean(x²)+eps)*(1+w)` per-H stream; verified bit-near (bf16 noise) against the torch reference, `combine` exact in fp32 |
| `GatedResidual` (`common` portable + `nvidia/hyperconnection.py` fused) | `hyperconnection.py::GatedResidual` (`mix`/`combine`/`combine_and_mix`) | Same low-rank gate math (`silu(down/HC)`, `sigmoid(gate)`, `mean_hc`, `w=2σ(inj/HC)`); merged `down_block_inject` param matches `_EXTRA_WEIGHTS_MAPPER`; closed-form pinned by `test_hc_combine_matches_reference` |
| `Qwen3NextAttention` (dense branch) | `attention.py::Qwen4ExpDenseAttention` | Same QK-norm + partial RoPE (`head_dim*partial_rotary_factor`) + `sigmoid(gate)` + `o_proj`; fused `qkv_proj` split at load. Split order verified against `qwen3_next.py::_project_qkv_gate`: leading `2*q` rows are per-head interleaved `[q_h, g_h]` pairs (reshape + chunk), pinned by `test_dense_qkv_gate_interleave` |
| `Qwen4ExpQSAAttention` + `qwen4_exp_qsa_with_output` (`nvidia/qsa.py`, `ops/qsa.py`) | `layers.py::_full_attn` QSA branch + `qsa.py::sparse_gqa` | Sparse GQA gather+softmax is mathematically identical to Triton split-K kernel; PDL/block-table handled by runner; BF16-only preserved. Chunked prefill == full forward (`test_qsa_chunked_matches_full`) |
| `QSAIndexer`, `apply_qsa_rope` (`indexer_qsa.py`, `ops/qsa_indexer.py`, `ops/qsa_pre_indexer.py`) | `qsa.py::QSAIndexer::project_qk`, `compress_keys_mean`, `qsa_select_indices` | Same `index_qk_proj` → RMS → RoPE → group-mean → `ReLU(q·kc)` sum-heads → top-k (`budget/ratio`) → expand (+causal tail); `jnp.top_k` replaces C++ top-k |
| `QSAKeyStateCache` / `QSACompressedKeyCache`, slot mappings, `QSAMetadataBuilder` (`common/qsa_cache.py`) | `cache.py` (`circular_capacity/slot`, `compressed_slot`, `Qwen4ExpCacheSpec`) | Same ring (`pos % capacity`) + boundary-only (`(pos+1)%ratio==0 → pos//ratio`) rules; Phase 1 retains logical history and recompresses causally per forward (chunk-consistent), side caches are the production optimization |
| `QwenGatedDeltaNetAttention` (`qwen_gdn_linear_attn.py`) | `gdn.py::Qwen4ExpGDN` | Same `in_proj_qkvz`/`in_proj_ba` split, causal depthwise SiLU conv, `RMSNormGated(sigmoid)` (Qwen4Exp `output_gate_type`), `out_proj`; production path calls `run_jax_gdn_attention` (Pallas v3) with conv+recurrent state, CPU/test path uses documented chunked reference |
| `Qwen3NextSparseMoeBlock` + `Qwen4ExpSparseMoeBlock` | `moe.py::Qwen4ExpMoE` | Same gate → top-k(`num_experts_per_tok`) → `norm_topk_prob` → `silu(gate)*up` → `down`, plus shared-expert SwiGLU; fused `gate_up` layout handled at load; EPLB via runner |
| `Qwen3NextMLP` | `moe.py::Qwen4ExpMLP` | Same SwiGLU |
| `Qwen4ExpNGramEmbedding` + `Qwen4ExpPLELayer` + `ple_*` ops | `ngram.py::Qwen4ExpPLE` + `compute_ngram_ids` + `ple_multipliers` + `ple_vocab_sizes_offsets` | Same splitmix64 hash (vocab-aware bound, no pre-mask), global-head prime sizes/offsets (`_is_prime_64`/`_nth_prime_after` port), per-order mixing (`order n` mixes first `n` tokens into that order's heads), `remainder+offset` ids, per-head `head_dim` embed + concat flatten, merged `kv_proj`, grouped-norm gate `σ(sign(d)√max(|d|,1e-6))`, dilated (`dilation=ngram_size`) zero-init conv accumulated to HC state. Per-order isolation pinned by `test_ple_ngram_per_order_isolation` |
| `copy_ple_embedding_shard_`, `PLEShardOverlap` | `weight_loader.py` (shard math documented) + loader test | Same `split_ngram_parts` row-range copy semantics |
| `Qwen4ExpDecoderLayer` (delayed combine) | `layers.py::Qwen4ExpDecoderLayer::forward` | Same PLE-materialize → `combine_and_mix`/`mix` → attn → `mlp_hc.combine_and_mix` (delayed) → MLP ordering; same `is_moe_layer` rule |
| `Qwen4ExpModel` (embed repeat, final mixer, `_mtp_hidden_buffer`) | `model.py::Qwen4ExpModel` | Same `[T,H]→[T,HC*H]` repeat, per-layer delayed tuple, non-last-rank materialize, final `mixer(use_combine=False)` → sample `[T,H]`; MTP snapshot point marked |
| `Qwen4ExpForCausalLM` (`compute_logits`, MRoPE, mamba specs) | `model.py::Qwen4ExpForCausalLM` | Same `__call__`/`compute_logits` runner contract (`qwen3.py` skeleton); PP `JaxIntermediateTensors`; tied-embedding decode fallback |
| `Qwen4ExpForConditionalGeneration` (vision tower) | registered alias to text model (vision embeds via deepstack path — unsupported, see below) | Same trunk; vision tower reuse tracked |
| `Qwen4ExpMultiTokenPredictor` / `Qwen4ExpMTP` + spec rewrites | `model.py::Qwen4ExpMTP` (shape-correct stub) | Same dual-stream/scheme-A + PLE-off + index-share design; full draft acceptance wiring deferred (see unsupported list) |
| `Qwen4ExpModelState` (ngram_context) | `cache.py` + `model.py` kwargs (`ngram_context`, `token_to_req`, `query_start_loc`) | Same rollback-safe `[R, ngram-1]` EOS-padded history |
| `qwen4_exp_*` custom ops + Triton kernels | plain JAX ops (no Pallas initially) | Each op's math documented in its module docstring; fusion is a later optimization |
| `AutoWeightsLoader` + mappers + `load_weights` | `weight_loader.py` (`PREFIX_MAP`, `STACKED_MAP`, QSA scale remap, `TRANSPOSE_SUBSTR`) + `StandardWeightLoader` | Same stacked/fused handling, expert layout, PLE shards, scales/biases/heads; Q4 via `quant.py::dequantize_q4_packed` |
| Quantization (FP8 embed global-scale, ModelOpt, `without_modelopt_fp4`) | `quant.py` (`QUANT_SKIP_SUBSTR`, Q4-first) | Same HC/QSA-cache skips; FP8 global-scale path preserved for reference checkpoints; Q4 packed + JAX dequant is Phase-1 target |
| Registration (`registry.py` + `__init__` dispatch) | `__init__.py::register()` (out-of-tree `register_model`) + `patches/model_loader.patch` (in-tree) | Maps `Qwen4ExpForCausalLM` / `...ConditionalGeneration` / `Qwen4ExpMTP` to JAX impls |

## Custom-op → JAX replacement checklist

| CUDA/Triton op | Math | JAX replacement |
|---|---|---|
| `qwen4_exp_grouped_gemma_rmsnorm` | per-H Gemma RMS | `grouped_gemma_rmsnorm` (einsum-free reshape) |
| `hc_silu` / `hc_gate_mix` / `hc_combine` / `hc_combine_norm` | silu/HC, sigmoid gate mean, `res+block*w`, fused combine+norm | `GatedResidual.mix/combine/combine_and_mix` |
| `ple_ngram_ids` | splitmix64 XOR hash + remainder | `compute_ngram_ids` |
| `ple_gate` | RMS/RMS/dot/`σ(sign√max)`/gate/RMS | `Qwen4ExpPLE.gate` |
| `ple_conv` | dilated causal depthwise conv + state writeback | `Qwen4ExpPLE.dilated_conv` (+ runner `conv_state`) |
| MQA paged score / top-k / expand | `ΣHeads ReLU(q·kc)`, top-k, block→token expand + tail | `qsa_select_indices` |
| sparse paged GQA split-K + LSE merge | masked softmax over selected rows | `sparse_gqa` (gather + softmax) |
| fused norm+rope+compress+ring-update | Q-norm, RoPE, group-mean, ring store | `QSAIndexer.project_qk` + `compress_keys_mean` |
| `low_latency_gemm` skinny GEMM | shape-specialized decode GEMM | plain einsums (XLA fuses; optimize later) |
| FLA `chunk_gated_delta_rule` | gated delta recurrent rule | Pallas `run_jax_gdn_attention`; documented JAX reference fallback |

## TPU v5e-8 qualification (2026-09-05, Kaggle `TpuV5E8`)

Notebook: `hemanthvattikuti/nexus-tpu-qwen3-8flashnext` (v17, `SaveAndRunAll`;
reproducible via `scripts/kaggle_qualify.py --all`). The fork source is
embedded in the notebook (no GitHub dependency); the VM runs Python 3.12 with
current JAX/flax.

- 8 TPU devices visible (`TPU_0`..`TPU_7`); TPU gate refuses CPU fallback.
- `pytest tests/models/jax/test_qwen4_exp.py`: **exit code 0** (all 15 tests).
- QSA decoder layer executes **on TPU** (output buffers on `TpuDevice`) with
  **CPU-identical numerics**, and chunked prefill == full forward.
- Tiny-model functional timings (correctness scale, not production claims):
  first-call XLA compile ~4.3 s, steady 8-token prefill ~58 ms
  (~138 tok/s). Full-model throughput must be measured with real weights
  (see `benchmarks/bench_qwen4_exp.py`).

Two modern-stack incompatibilities were found by this run and fixed in the
fork (both reproduced locally on JAX 0.11 / flax 0.12 / Python 3.12):

1. flax eager sharding requires an active mesh at construction
   (`ValueError: ... auto mesh context ...`); newer JAX only exposes it via
   `jax.set_mesh` (plain `with Mesh` is deprecated and no longer sets the
   abstract mesh). Tests construct modules under a version-adaptive mesh
   fixture; serving always builds under the runner mesh.
2. Modern flax infers `None` attribute assignments as static and rejects
   later reassignment to a submodule; `Qwen4ExpDecoderLayer` now assigns
   each conditional submodule (`ple`, `linear_attn`, `self_attn`, `indexer`)
   exactly once.

## Known gaps (correctness-first, tracked explicitly)

1. **GDN recurrent kernel**: CPU/test fallback is a simplified gated recurrence (shapes/gating/norm ordering correct; NOT the full FLA Householder delta rule). Production TPU path uses `run_jax_gdn_attention`. Numerical parity for GDN layers must be validated against GPU reference before claiming end-to-end parity (see tests Phase 5).
2. **QSA side-cache optimization**: Phase 1 retains the logical raw-key/KV history and recompresses causally per forward (chunk-consistent, verified by `test_qsa_chunked_matches_full`) instead of maintaining the ring + compressed caches across steps. Slot rules are implemented (`cache.py`) and ready to wire into the runner; batching + long-context perf depend on it.
3. **MRoPE vision branches**: text 1D path implemented; height/width branches fall back to temporal positions (text-only serving unaffected).
4. **MTP draft acceptance / index-share**: stub only; target-model prefill/decode does not require it.
5. **Vision tower** (`Qwen4ExpForConditionalGeneration`): alias only; multimodal tracked as unsupported in v1.
6. **Q4 format**: generic GPTQ/AWQ-style + compressed-tensors W4A16 dequant provided; exact official Q4 export must be pinned against the released checkpoint (see `quant.py`).
