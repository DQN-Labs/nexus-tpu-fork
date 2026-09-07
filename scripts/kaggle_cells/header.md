# Qwen4Exp (Qwen3.8-Flash-Next) JAX fork - TPU v5e-8 qualification

Tests the **`tpu-inference-qwen4exp`** fork (Qwen4Exp JAX implementation) on Kaggle TPU v5e-8.
The fork source is embedded in the setup cell (no GitHub dependency).

Proves: TPU visible (no silent CPU fallback) -> fork unit tests pass -> QSA decoder-layer math executes **on TPU** with CPU-identical numerics -> chunked prefill == full forward, with compile/prefill/decode timings.