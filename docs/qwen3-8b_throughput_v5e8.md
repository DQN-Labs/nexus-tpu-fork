# Qwen3-8B throughput on TPU v5e-8 (tpu-inference JAX backend)

Date: 2026-09-05. Notebook: `hemanthvattikuti/nexus-tpu-qwen3-8flashnext`
(v28, `SaveAndRunAll`). Raw artifacts: `sweep_results.json`,
`vllm_server.log` (Kaggle session outputs).

> Note: precision is **BF16** (stock `Qwen/Qwen3-8B` safetensors), not FP16.
> The two are the same width (2 bytes/param); nothing below changes if you
> say "16-bit".

## Setup

| Item | Value |
|---|---|
| Hardware | TPU v5e-8 (8 chips, single host) |
| Model | Qwen/Qwen3-8B (8.19B params, BF16 = 16.4 GB weights) |
| Stack | jax 0.11.0, tpu-inference 0.28.0, vllm 0.28.0 (`--no-deps` wheel + explicit runtime deps), torch 2.10 |
| Server | `vllm serve`, `TPU_BACKEND_TYPE=jax`, `--tensor-parallel-size 8`, `--max-model-len 20480`, `--max-num-seqs 8` |
| Arch path | `Qwen3ForCausalLM` (JAX-native, dense, BF16, no quantization) |
| Client | OpenAI `/v1/completions`, `stream=true`, greedy (`temperature 0`) |

## Method

Per point: one warmup request (excluded, absorbs XLA compile), then measured
request. Time-to-first-token (TTFT) from first stream chunk;
`prefill_tok/s = prompt_tokens / TTFT`;
`decode_tok/s = completion_tokens / (total - TTFT)`. Token counts from the
server-reported `usage` block. Batch-1 unless noted. Single run per point
(no repetitions — treat last digits as approximate).

## Measured results (batch-1, 128 decode tokens)

| Prefill (prompt tokens, actual) | TTFT (s) | Total (s) | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|
| warmup: 1,121 | 0.186 | 0.296 | 6,026 | 145.2 |
| 2,261 (~1k target) | 0.066 | 1.040 | 34,487 | 131.4 |
| 9,101 (~4k target) | 0.226 | 1.190 | 40,359 | 132.7 |
| ~36,000 (~16k target) | — | — | — | — (see note) |

## Notes

- **Decode is flat (~131–145 tok/s)** across prefill lengths: classic
  memory-bandwidth-bound decoding. Implied effective bandwidth ≈ 135 tok/s
  × 16.4 GB ≈ **2.2 TB/s** aggregate (~34% of ~6.5 TB/s peak — healthy
  end-to-end including sampling/scheduling/collectives).
- **Prefill rises with length** (34k → 40k tok/s): compute-bound regime
  where the TPU gets more efficient — exactly the expected shape.
- **16k point missing (harness bug, not server):** prompt-size calibration
  was ~2.2× off, so the "~16k" prompt was really ~36k tokens, exceeding
  `--max-model-len 20480` → HTTP 400. Rerun with corrected sizing (or a
  larger `--max-model-len`) to fill it.
- Warmup TTFT (0.186 s) is already post-compile; first-compile cost was
  absorbed during server startup/model load, not measured here.

## What this is (and isn't)

- A **backend health + roofline baseline** for dense 8B BF16 serving on
  v5e-8 — not a Flash-Next measurement. Used to derive the Flash-Next
  estimate (6B-active Q4 ⇒ ~3 GB/token ⇒ ~400–550 tok/s batch-1 single
  stream after a 30–45% reality haircut; see project notes).
- Single-run numbers; production claims want repetitions, a batch sweep,
  and the 16k/32k long-context points.
