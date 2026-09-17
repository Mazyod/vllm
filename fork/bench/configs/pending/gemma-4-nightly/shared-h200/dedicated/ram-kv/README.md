# Opt-in RAM prefix-cache candidates

The smallest additional cache architecture is vLLM's native `OffloadingConnector`:
completed prompt KV blocks are copied to local host RAM and can be retrieved when
their GPU copy is no longer available. It needs no LMCache service, router change,
or model-weight offload. The connector is present in the pinned image.

**These are statically checked candidates, not GPU-validated configurations.**
Hybrid attention, FP8, speculative decoding and images must pass retrieval checks
for each exact model before promoting this beyond a trial. The dedicated baseline
remains unchanged. Start with DeepSeek, where expensive long prefills and a smaller
GPU cache make reuse particularly worth measuring.

| Dedicated node | Existing GPU KV across TP4 | Candidate CPU KV, total engine | `/dev/shm` ceiling |
| --- | --- | --- | --- |
| DeepSeek | 144 GiB | 256 GiB | 320 GiB |
| Gemma | 448 GiB | 768 GiB | 800 GiB |

These RAM budgets are starting choices for 2-TB hosts, not measured optima.
The earlier 128-GiB Gemma suggestion was a small compatibility experiment; it
is smaller than the dedicated GPU allocation and may offer little extra retention.
The new Gemma tier is intentionally larger. GPU and CPU allocations are not
additive unique-token capacities: cached content overlaps, and hybrid/replicated
layouts may have different packing and deduplication behavior.

The CPU budget is **across the engine's workers, not per GPU**. The native CUDA
allocator uses shared mmap under `/dev/shm`, so each overlay enlarges that service's
mount. The ceiling is not a second allocation on top of the RAM cache. Preserve
room for DeepSeek's ~188.8-GiB Engram allocation, process/model-loading memory,
multimodal caches and the OS; check actual host/cgroup availability first.

Only the KV connector and `/dev/shm` ceiling change. The pinned weights, FP8 GPU
KV budgets, context, speculation, NCCL environment, model cache mounts, and router
remain the measured dedicated settings. Offload is prompt-only; eviction uses
the default LRU policy. The external token block size is deliberately not forced
because these models have heterogeneous KV groups.

## Enable one candidate

Run from the parent `deepseek-gemma4-dedicated` directory after the baseline works.
This restarts the selected model, with the other model's configuration unchanged.
For DeepSeek first:

```bash
docker stack config -c stack.yaml -c stack.ram-kv-deepseek.yaml > /dev/null
docker stack deploy -c stack.yaml -c stack.ram-kv-deepseek.yaml inference-dedicated
docker service logs -f inference-dedicated_deepseek
```

For a Gemma-only trial, use `stack.ram-kv-gemma4.yaml` instead. To retain both
candidates after validating them individually, include both overlays:

```bash
docker stack deploy -c stack.yaml \
  -c stack.ram-kv-deepseek.yaml \
  -c stack.ram-kv-gemma4.yaml inference-dedicated
```

Keep every active overlay in subsequent deploy commands. Redeploying only
`stack.yaml` restores the measured baseline and removes both RAM tiers. Reusing
an immutable Swarm config name after editing its YAML will fail; increment
`DEEPSEEK_RAM_KV_CONFIG_REV` or `GEMMA4_RAM_KV_CONFIG_REV` when changing a candidate.

## What would make this a win

1. Compare a cold request with repeat requests that share real long prefixes.
   Keep system text, tools, document order and image identity stable; put variable
   request text after the reusable content.
2. Confirm a retrieval from the RAM tier after GPU cache eviction. An immediate
   repeat that hits GPU APC proves nothing about the CPU tier. Check changes in
   `vllm:external_prefix_cache_hits_total` and
   `vllm:external_prefix_cache_queries_total` on the **engine** metrics endpoint.
   Do not restart the engine to force eviction: that discards this RAM cache too.
3. Check answer correctness, speculative acceptance, changed-image/changed-prefix
   misses, and C4 requests. Compare TTFT p50/p95, decode latency, throughput and
   memory against the baseline, including unique prompts that get no reuse.
4. Warm a few high-traffic prefixes after restart with representative short-output
   requests using the same templates. CPU KV is reclaimed when its workers exit;
   it is not persistent storage or a failover mechanism.

The benefit is less recomputation **after GPU eviction**. Already-resident GPU
hits cannot get faster by going through RAM. Unique inputs and ordinary decode
steps gain no reusable-prefix shortcut; transfer overhead still has to be measured.
With this much RAM, NVMe and cross-node cache services can wait until measured
reuse exceeds the chosen local RAM tier.

Sources: [pinned offloading guide](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/docs/features/kv_offloading_usage.md),
[pinned shared allocator](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/v1/kv_offload/cpu/shared_offload_region.py).
