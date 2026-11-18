# v0.34.0 qualified configuration

The recurring release uses `openimage/vllm`, without installing the audio extra.
The current gate is DeepSeek V4.1 Flash on an anonymous four-GPU Hopper profile.
The [hardware gate](results/20261117-gate.md) passed on the exact candidate digest.
Subsequent [H200 tuning and host KV checks](results/20261118-tuning.md) support
adaptive verification and the CPU KV tier in the current profile.

The engine file preserves the pinned checkpoint, TP4, data-parallel vision
encoding, tool/reasoning parsers, and DSpark sampling. FP8 KV format, the
checkpoint's five-token draft width, tokenizer mode, and Engram CPU offload
are detected/defaulted upstream. The model's native
context is retained. There is no fork-imposed image count or thinking policy.

`enable-prompt-tokens-details: true` preserves usage metadata, including
`usage.prompt_tokens_details.cached_tokens` when available. This required API
behavior defaults to false upstream. Real cache-hit and cache-creation counts
were checked during the subsequent GPU campaign. The historical release gate
certifies the immutable configuration linked in its record.

Default-valued dtype, executor, parallel sizes, MoE backend, prefix caching,
chunked prefill, processor cache, and disabled features are omitted. Fixed KV
bytes and GPU utilization are omitted to retain extensible KV sizing. The sole
scheduler override, `max-num-seqs: 512`, is required to leave memory for that
cache; default 1024 failed startup. DSpark improved decode throughput, and
data-parallel vision improved the nine-image probe. See the gate for measured
tradeoffs. Adaptive verification improves higher-concurrency throughput, with
a measured low-concurrency tradeoff. Expert parallelism and forced Marlin lost
throughput in the comparisons. Larger graph capture added little beyond
adaptive verification while reducing GPU KV capacity. Collective selection
remains automatic; removing an old workaround is a behavior change.

## Host KV memory requirements

`kv-offloading-size: 256` enables the default native CPU backend with **256 GiB
total across TP ranks**, not per GPU. Budget host RAM for that tier, roughly
188.8 GiB of Engram table data, allocator overhead, loading, and runtime memory.
Validation used 768 GiB RAM. The native connector also needs adequate
`/dev/shm`; for this budget, use an appropriate shared IPC namespace or, in
Docker, `--shm-size 320g`. Container memory limits must cover both allocations
and overhead. Adjust the budget to the deployment's resources.

The CPU tier extends prefix reuse; active attention still needs GPU KV memory.

## Migrating an existing deployment

This file is a qualified inference baseline, not a complete replacement for
deployment-specific requirements. Review omitted settings before replacing an
existing configuration:

| omitted setting | v0.34.0 behavior in this profile |
| --- | --- |
| `enable-expert-parallel: true` | Defaults to false. EP saved about 7.7 GiB of model allocation per GPU, but reduced C1/C8/C32 throughput and did not improve C128 in the same-hardware comparison. |
| `engram-config` | Automatic Engram CPU offload was confirmed in the gate. The old block's `cpu_offload: true`, `embedding_across_dp: false`, and `dp_shared_memory: false` resolve identically for this TP4/DP1 profile. Preserve any other intentional overrides. |
| `kv-cache-dtype: fp8` | Both `auto` and `fp8` resolve to `fp8_ds_mla` with the tested packed KV backend. This equivalence is backend-specific. |
| `enable-prefix-caching: true` | Prefix caching is enabled by default for this model; cache hits were observed in the gate. |
| `kv-offloading-backend` | Defaults to native. The explicit size now enables a 256 GiB CPU tier; reload after GPU-cache reset was verified with DSpark and packed FP8 KV. |
| `mm-processor-cache-gb: 4` | The upstream default remains 4 GiB. Other configured sizes are not equivalent. |
| `disable-custom-all-reduce: true` | Defaults to false, permitting custom all-reduce. The gate exercised upstream collective selection on NVLink-connected GPUs; it does not retire a PCIe-only workaround by proof. |

Generic `cpu-offload-gb` is separate model-weight offload, disabled by default.
Removing an explicit zero is equivalent; removing a positive value is not.

Extensible KV is a per-process allocator, not coordination between independent
engines. Shared GPUs still require explicit memory budgets that account for
all residents, including processes started later. Supplying fixed KV bytes
disables extensible sizing. A dedicated-engine result does not certify sharing.

Sources: [release notes](https://github.com/vllm-project/vllm/releases/tag/v0.34.0),
[cache configuration](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/cache.py),
[Engram defaults](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/engram.py).
