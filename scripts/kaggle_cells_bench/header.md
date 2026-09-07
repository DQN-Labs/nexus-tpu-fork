# Qwen3-8B serving throughput on TPU v5e-8 (tpu-inference JAX backend)

> **Do not cancel mid-run.** Cold setup downloads ~3GB of wheels plus ~16GB
> of weights and compiles XLA for an 8B model; the full run takes 1-3h.
> Each cell writes stage marker files and streams progress — silence during
> `pip install` / download / compile is normal.

Method: vLLM OpenAI-compatible server (`TPU_BACKEND_TYPE=jax`, TP=8) plus a
streaming client that measures time-to-first-token and token rates.

Sweep: prefill ≈1k / 4k / 16k tokens × 128 decode tokens (batch 1), plus one
batch-4 point. A warmup request (excluded from numbers) absorbs XLA
compilation. Raw numbers land in `/kaggle/working/sweep_results.json`.
