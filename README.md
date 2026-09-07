# tpu-inference-qwen4exp — Qwen3.8-Flash-Next (Qwen4Exp) on TPU

Clean fork overlay of [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference)
adding first-class JAX/TPU support for **Qwen3.8-Flash-Next**, internally `Qwen4Exp`
(`Qwen4ExpForCausalLM`), based directly on upstream
[`vllm-project/vllm`](https://github.com/vllm-project/vllm) `vllm/models/qwen4_exp/`.

> Take the exact model behavior, tensor shapes, parameter structure, weight naming,
> configuration semantics, and forward-pass logic of upstream vLLM's Qwen4Exp model,
> and implement the equivalent execution path using the JAX/TPU architecture of
> `tpu-inference`. Correctness first (`vLLM CUDA/Triton → understand → JAX → XLA → TPU`).

## Layout

```
tpu_inference/models/jax/qwen4_exp/
  __init__.py        registration (register_model) + exports
  config.py          HF config compat + validation (mirrors Qwen4ExpTextConfig)
  hyperconnection.py GroupedGemmaRMSNorm + GatedResidual (mix/combine/combine_and_mix)
  attention.py       dense full-attention branch (QK-norm + partial RoPE + sigmoid gate)
  qsa.py             QSA indexer + top-k select + sparse GQA (exact JAX math)
  gdn.py             Gated Delta Net branch (Pallas kernel path + reference fallback)
  moe.py             routed MoE + shared expert + dense MLP
  ngram.py           PLE / n-gram embedding hash + gate + dilated short-conv
  layers.py          decoder layer with delayed HC combine
  model.py           full model + Qwen4ExpForCausalLM + Qwen4ExpMTP stub
  cache.py           KV / QSA side-cache / GDN / PLE / ngram-context specs
  weight_loader.py   checkpoint → JAX name mapping (stacked/fused, QSA scales, PLE shards)
  quant.py           Q4 packed + JAX dequant (Phase-1 target)
tests/models/jax/test_qwen4_exp.py   CPU tests (config, mapping, components, Q4)
docs/qwen4_exp.md                    Phase-1 architecture mapping (vLLM → JAX) + gaps
examples/qwen4_exp.py                registration + serve example
benchmarks/bench_qwen4_exp.py        compile/prefill/decode/memory/context reporter
scripts/setup_kaggle_tpu_v5e.sh      Kaggle TPU v5e-8 setup
scripts/install_clean.sh             clean-env install
patches/model_loader.patch           in-tree registration patch
```

## Install (clean env)

See `scripts/install_clean.sh` and `scripts/setup_kaggle_tpu_v5e.sh`.
Never silently falls back to CPU; missing `tpu_inference`/JAX raises loudly.

## Serve (normal vLLM path)

```bash
TPU_BACKEND_TYPE=jax python -m vllm.entrypoints.cli.main serve \
    <QWEN_MODEL> --tensor-parallel-size 8 --max-model-len <context>
```

Out-of-tree (preferred, keeps fork clean):

```python
from tpu_inference.models.jax.qwen4_exp import register
register()  # Qwen4ExpForCausalLM / ...ConditionalGeneration / Qwen4ExpMTP
```

In-tree: apply `patches/model_loader.patch` to `tpu_inference`.

## Quantization (v1: Q4)

`quant.py::dequantize_q4_packed` handles packed int4 + per-channel/per-group
scale/zero (+ compressed-tensors W4A16 shape). HC linears and QSA cache scales
are skipped per upstream (`without_modelopt_fp4`). Pin the exact official Q4
export before claiming parity (see `docs/qwen4_exp.md`).

## Testing

```bash
pytest tests/models/jax/test_qwen4_exp.py -q   # CPU, no TPU needed
QWEN4EXP_E2E=1 pytest tests/models/jax/test_qwen4_exp.py -q  # TPU v5e-8 + ckpt
```

Module construction requires an active device mesh (flax eager sharding;
same rule as upstream `tpu-inference` models) — the test suite provides one
via fixture, and serving builds under the runner mesh. TPU qualification
results: see `docs/qwen4_exp.md` (Kaggle v5e-8, 15/15 pass, on-TPU execution
proof). Reproduce with `python3 scripts/kaggle_qualify.py --all`.

## Unsupported in v1 (explicit)

Vision tower, full MTP draft acceptance/index-share, GDN FLA-exact CPU fallback
(production uses Pallas kernel), QSA side-cache optimization (correctness via
recompute), MRoPE vision branches, exotic NVFP4/FP8 paths. See `docs/qwen4_exp.md`.
