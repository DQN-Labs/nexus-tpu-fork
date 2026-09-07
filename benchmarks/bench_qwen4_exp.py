#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Benchmark: Qwen4Exp on TPU v5e-8 — compile/prefill/decode/memory/context.

Reports (task deliverable 10): compile time, prefill tok/s, decode tok/s,
memory usage, context length. Requires a TPU VM + checkpoint.

Usage:
    python benchmarks/bench_qwen4_exp.py --model <HF_OR_PATH> --tp 8 \\
        --max-model-len 32768 --prompt "Hello, world"

The script starts from a cold server so the first request includes XLA
compilation; it then issues a short-prefill and a long-prefill request and
derives prefill/decode rates from the OpenAI `usage` block. Read HBM usage
and max context from the server logs (`tpu_info` / allocator stats) — the
script prints exactly what to look for.
"""
import argparse
import time

try:
    import requests
except ImportError:
    requests = None


def _post(url, model, prompt, max_tokens, timeout=600):
    t0 = time.time()
    r = requests.post(f"{url}/v1/completions",
                      json={"model": model, "prompt": prompt,
                            "max_tokens": max_tokens, "temperature": 0.0},
                      timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    return r.json(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--prompt", default="Explain quantum computing in one sentence.")
    ap.add_argument("--serve-url", default="http://localhost:8000")
    ap.add_argument("--decode-tokens", type=int, default=64)
    args = ap.parse_args()

    print(f"model={args.model} tp={args.tp} max_len={args.max_model_len}")
    print("1) Start a COLD server (first request below includes XLA compile):")
    print(f"   TPU_BACKEND_TYPE=jax python -m vllm.entrypoints.cli.main serve "
          f"{args.model} --tensor-parallel-size {args.tp} "
          f"--max-model-len {args.max_model_len}")
    if requests is None:
        print("(install requests to run the client probe)")
        return

    # Cold request incl. compile.
    out, dt = _post(args.serve_url, args.model, args.prompt, args.decode_tokens)
    usage = out.get("usage", {})
    pt, ct = usage.get("prompt_tokens", -1), usage.get("completion_tokens", -1)
    text = out["choices"][0]["text"] if out.get("choices") else ""
    print(f"\n[COLD incl. compile] latency_s={dt:.2f} "
          f"prompt_tok={pt} completion_tok={ct}")
    print(f"  generated: {text[:300]!r}")

    # Warm request: same shape, compile cached -> serving latency.
    out2, dt2 = _post(args.serve_url, args.model, args.prompt, args.decode_tokens)
    usage2 = out2.get("usage", {})
    ct2 = usage2.get("completion_tokens", 0) or 1
    print(f"\n[WARM] latency_s={dt2:.2f} "
          f"approx_serve_tok_per_s={ct2 / max(dt2, 1e-6):.1f} "
          f"(includes one prefill + {ct2} decodes)")

    # Long-prefill request to expose prefill scaling.
    long_prompt = (args.prompt + " ") * 200
    out3, dt3 = _post(args.serve_url, args.model, long_prompt, 8)
    usage3 = out3.get("usage", {})
    pt3 = usage3.get("prompt_tokens", 0) or 1
    print(f"\n[LONG PREFILL] latency_s={dt3:.2f} prompt_tok={pt3} "
          f"approx_prefill_tok_per_s={pt3 / max(dt3, 1e-6):.1f}")

    print("\n2) From the SERVER log, record:")
    print("   - compile time: XLA compilation span on the cold request")
    print("   - memory usage: HBM per device (allocator / tpu_info output)")
    print("   - maximum context: raise --max-model-len until the KV + QSA "
          "side-cache + GDN/PLE state reservation fails, then back off")
    print("   - batch size: sweep --max-num-seqs and concurrent clients")


if __name__ == "__main__":
    main()
