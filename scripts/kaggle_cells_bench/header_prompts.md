# Qwen3-8B prompt tests on TPU v5e-8 (tpu-inference JAX backend)

Same server as the throughput sweep (`TPU_BACKEND_TYPE=jax`, TP=8,
`--max-model-len 20480`, `--max-num-seqs 8`), but instead of synthetic
token-rate points it runs representative prompts end to end: a short
factual answer, a multi-step reasoning puzzle, a long (~10k-token) code
review with a planted concurrency bug, and a greenfield code-generation
task. Responses plus per-request timings land in
`/kaggle/working/prompts_results.json`.

Note: every notebook version runs on a fresh VM, so setup (wheels) and the
16GB weight download rerun each time — installs cannot be skipped, only
reused via attached datasets (not wired up yet).
