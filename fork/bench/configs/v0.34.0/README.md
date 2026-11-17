# v0.34.0 candidate configuration

The recurring release uses `openimage/vllm`, without installing the audio extra.
The current gate is DeepSeek V4.1 Flash on an anonymous four-GPU Hopper profile.
Hardware validation is pending; historical results do not certify this release.

The engine file preserves the pinned checkpoint, TP4, FP8 KV, data-parallel
vision encoding, tool/reasoning parsers, and five-token DSpark. The tokenizer
mode and Engram CPU offload are detected/defaulted upstream. The model's native
context is retained. There is no fork-imposed image count or thinking policy.

Default-valued dtype, executor, parallel sizes, MoE backend, prefix caching,
chunked prefill, processor cache, and disabled features are omitted. Fixed KV
bytes and GPU utilization are omitted so the release can exercise extensible
KV sizing. Legacy collective overrides, expert parallelism, scheduler caps,
and adaptive verification require a measured benefit before reinstatement.

Extensible KV is a per-process allocator, not coordination between independent
engines. Shared GPUs still require explicit memory budgets that account for
all residents, including processes started later. Supplying fixed KV bytes
disables extensible sizing. A dedicated-engine result does not certify sharing.

Sources: [release notes](https://github.com/vllm-project/vllm/releases/tag/v0.34.0),
[cache configuration](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/cache.py),
[Engram defaults](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/engram.py).
