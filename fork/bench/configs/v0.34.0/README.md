# v0.34.0 qualified configuration

The recurring release uses `openimage/vllm`, without installing the audio extra.
The current gate is DeepSeek V4.1 Flash on an anonymous four-GPU Hopper profile.
The [hardware gate](results/20261117-gate.md) passed on the exact candidate digest.

The engine file preserves the pinned checkpoint, TP4, data-parallel vision
encoding, tool/reasoning parsers, and DSpark sampling. FP8 KV format, the
checkpoint's five-token draft width, tokenizer mode, and Engram CPU offload
are detected/defaulted upstream. The model's native
context is retained. There is no fork-imposed image count or thinking policy.

Default-valued dtype, executor, parallel sizes, MoE backend, prefix caching,
chunked prefill, processor cache, and disabled features are omitted. Fixed KV
bytes and GPU utilization are omitted to retain extensible KV sizing. The sole
scheduler override, `max-num-seqs: 512`, is required to leave memory for that
cache; default 1024 failed startup. DSpark improved decode throughput, and
data-parallel vision improved the nine-image probe. See the gate for measured
tradeoffs. Legacy collective overrides, expert parallelism, and adaptive
verification remain omitted.

Extensible KV is a per-process allocator, not coordination between independent
engines. Shared GPUs still require explicit memory budgets that account for
all residents, including processes started later. Supplying fixed KV bytes
disables extensible sizing. A dedicated-engine result does not certify sharing.

Sources: [release notes](https://github.com/vllm-project/vllm/releases/tag/v0.34.0),
[cache configuration](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/cache.py),
[Engram defaults](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/engram.py).
