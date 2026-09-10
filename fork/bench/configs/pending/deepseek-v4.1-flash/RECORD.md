# DeepSeek V4.1 Flash development measurements

Status: campaign in progress; candidate configurations are not release-certified.

Question: measure memory and speed on one warm host with Engram tables on CPU
versus GPU, and compare tensor, pipeline, expert parallelism and DSpark.

- Anonymous venue: `hopper-fabric-large`, eight H100 SXM GPUs.
- Dedicated upstream image: `vllm/vllm-openai:deepseekv41-flash-0909`.
- AMD64 image digest: `sha256:4f3c8bcf6328305b8cb6f61146dbfa125b0b04ae98eef3d091019ec30f564a9b`.
- Source inspected: upstream PR #56214, `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
  The installed image version and source must be captured independently.
- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
  `dba1be0a40aa45a94ad051997016db3960a90277`, staged once at
  `/workspace/models/v41` and reused for every arm.
- Development cap: $50, external watchdog at 6,900 seconds from arming.
  Listed hourly cost: $18.7207 including the requested disk allocation.
  Listed ingress: $0.0133333/GB; initial checkpoint cache is cold.
- Model Runner V2, Python API frontend, DeepGEMM eager warmup skipped;
  exact environment and configuration SHA-256 accompany each result.
- Candidate matrix: TP8 Engram CPU/GPU, each with and without EP;
  TP4 CPU with and without EP; TP4/PP2 CPU/GPU; TP2/PP4 CPU;
  TP8 CPU/GPU with static/adaptive DSpark at five speculative tokens.
- Adaptive DSpark with PP is excluded by the implementation's explicit check.
- Baseline context: 32,768 tokens, 32 sequences, 4,096 batched tokens,
  prefix caching disabled, total configured KV budget 16 GiB across ranks.
- Measurements: boot time, memory telemetry, arithmetic correctness,
  1K-input/256-output at concurrency 1/8/32, 8K-input at concurrency 1,
  and natural coding/prose requests with speculative counter deltas.
- Performance runs are serialized. Each attempt gets new immutable config/log
  artifacts; failed engines are stopped without destroying the rental/cache.

Raw provider records and logs stay under ignored `runs/deepseek-v41-20260910/`.
No hardware results have been claimed by this preparation record.
