# Production run WITH the fork applied (Qwen3.8-Flash-Next GPTQ on v5e-8).

Same server shape as the stock runs, except cell 2 clones
`DQN-Labs/nexus-tpu-fork` and registers its `Qwen4ExpForCausalLM` JAX
implementation into the tpu-inference loader, and cell 4 inspects the real
checkpoint with the fork's config + weight-name mapping (index/headers only,
no values loaded) before serving is attempted.
