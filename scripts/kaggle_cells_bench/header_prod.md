# Production prompt tests on TPU v5e-8 (owner-provided GPTQ model).

Same server shape as the throughput sweep (TPU backend, TP=8), but this
notebook exists for the production export: model path resolves from
`NEXUS_MODEL_PATH`, the `MODEL_PATH` constant, or a `/kaggle/input` scan.
With no export provided it reports NO PATH FOUND and stops cleanly.
