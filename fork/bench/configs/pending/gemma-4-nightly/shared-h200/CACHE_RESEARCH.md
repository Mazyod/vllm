# KV cache research — 17 September 2026

Research only; no new GPU run or external-cache certification. Audited engine source: `af1c01499b289be555c475669ba50a88e96d846e` (the pinned image).

The router fork is now identified: **`openimage/production-stack-router:v0.1.12`**, upstream baseline `66b60661aa3052810859a417559e9e830772a091` plus 29 fork patches. Its published manifest is `sha256:7144e56c84abcb3e3f42a5eb37dcee16a46945f980a04d3d51e4134b51768733`, checked against Docker Hub. The successful release source is `5cca5e4414dd07192ff37f7e88da0e6513bb69d3`; current fork head `6970618c496817b085444d2a2e64949fcd05d9d6` adds only release documentation. Source comparison shows no fork changes to `routing_logic.py` or `hashtrie.py` relative to upstream 0.1.12. This identifies the published release, not the digest of an already running production container. [Fork release provenance](https://github.com/Mazyod/production-stack/blob/6970618c496817b085444d2a2e64949fcd05d9d6/docs/fork-maintenance.md), [successful release run](https://github.com/Mazyod/production-stack/actions/runs/34873842221)

The earlier audit also inspected newer upstream `ebbb8624010514c2d4fa7edf3de95a6440a755b2`. Its newer `loadaware` mode is **not in this fork release**. The selected deployment is now one dedicated node per model; [the canonical package](../../../../../deploy/deepseek-gemma4-dedicated/README.md) keeps a single backend per model and native APC.

## Decision

Keep native automatic prefix caching (APC), which is already enabled. Start with stable prompt construction and session affinity. For the next experiment, test a bounded **node-local native CPU KV tier** before adding another service. LMCache's external multiprocess connector is a credible alternative with stronger published Gemma4-specific evidence. Add NVMe or cross-node sharing only after measuring actual reusable-prefix misses and transfer cost.

These mechanisms primarily avoid repeated **prefill**, improving TTFT and reducing competing prefill work. They do not make each decode step intrinsically faster, guarantee a 20 tok/s floor, or turn host RAM into extra active HBM. Independent active contexts still need their GPU KV allocation. Reusing identical prefixes can reduce duplication, but capacity planning should retain the measured distinct-prefix worst case. [APC limits](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/#limits)

## Routing: what the Python router actually does

- `session`: consistent-hash affinity using a configured session identifier; without it, fallback is QPS routing. This is the simplest useful choice for multi-turn conversations across replicas.
- `prefixaware`: remembers previous routing decisions in a process-local trie. It explicitly assumes no eviction; this is not an authoritative inventory of GPU or CPU KV blocks. For chat it concatenates message text, omitting image identity, roles and template structure; it chooses randomly among the endpoints with the longest remembered match. A minimum-match threshold falls back to QPS.
- `kvaware`: queries LMCache's controller for longest cached token coverage. In the fork release it still tokenizes only `request_json.get("prompt", "")`, with a TODO for chat completions. The published slim image does not include LMCache, confirmed by inspection inside that exact image; enabling this mode is not a drop-in configuration change.
- `loadaware`: absent from the 0.1.12 release. It exists only in the newer upstream code inspected earlier and must not be recommended as a shipped fork option.

Therefore, do not promise exact longest-prefix cache routing for this chat+images stack by setting a router flag. The release-level chat/template/multimodal limitations remain after accounting for the fork patches. Multiple router replicas also have independent heuristic histories. [Audited routing implementation](https://github.com/Mazyod/production-stack/blob/5cca5e4414dd07192ff37f7e88da0e6513bb69d3/src/vllm_router/routers/routing_logic.py)

A heuristic routing match is not itself KV reuse: the engine still validates its own token/image cache keys. A stale router estimate can choose a backend with little reusable cache; it should not be interpreted as proof of an engine cache hit.

The heuristic trie hashes **128-character chunks**, not engine token blocks. [HashTrie implementation](https://github.com/Mazyod/production-stack/blob/5cca5e4414dd07192ff37f7e88da0e6513bb69d3/src/vllm_router/prefix/hashtrie.py)

## Dedicated-router configuration and timeout correction

The canonical dedicated package uses `routing_logic: roundrobin` with one explicitly mapped backend per model. No replica-routing strategy can improve locality within a single backend; native engine APC still reuses prefixes. Keep existing model routes if merging this into a router serving other models.

The fork defaults to a **300-second backend read-silence timeout**, with a separate 10-second connect timeout. The measured fourth cold 1M request reached 608 seconds to its first content token. The dedicated router YAML therefore uses `backend_read_timeout: 1800.0` and retains `backend_connect_timeout: 10.0`. These values are startup-only; restart after changing them. Read/entry timeouts do not rotate backends, and an interrupted stream is not resumed on another model instance. Streaming avoids requiring an entire long response to finish before headers/body, but a silent prefill can still exceed a read bound. [Fork timeout contract](https://github.com/Mazyod/production-stack/blob/5cca5e4414dd07192ff37f7e88da0e6513bb69d3/README.md#backend-socket-timeouts)

The exact published router image passed local CPU-mock checks for health, two-model discovery, four text/image-body routing checks, and parsing the 1,800-second timeout. A derived 0.25-second YAML bound also passed pre-header 504, terminal SSE stall handling, healthy streaming beyond that bound, and request-counter cleanup. This does not constitute GPU integration or a timed 30-minute request test. The fork's default slim build omits LMCache/transformers/vLLM extras; an external-cache router build requires deliberate dependency/version validation.

## Native cache and prompt design

APC uses full token blocks chained to preceding content, with extra identities such as image hashes and optional request cache salt. Changing an early token invalidates the later prefix. Keep stable system text/tools/documents before variable request text; avoid timestamps/nonces near the start when unnecessary. Caches are engine-local, evictable, and lost on restart unless an external tier persists them. Models cannot reuse one another's KV. Cache salts scope reuse to a trust group when required. An image processor cache is separate from the attention KV cache. [APC design](https://docs.vllm.ai/en/latest/design/prefix_caching/)

## Native RAM / NVMe: already present in the pinned engine

The exact image source has `OffloadingConnector`, `CPUOffloadingSpec`, and `TieringOffloadingSpec` with filesystem and P2P tiers. GPU↔secondary-tier data passes through the CPU tier. `cpu_bytes_to_use` is a total engine budget across workers, not per GPU. Begin with RAM because it is already much larger than the reusable GPU pool. [Pinned usage guide](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/docs/features/kv_offloading_usage.md)

The connector advertises hybrid-cache support. Its translator handles multiple KV groups, but setting an explicit external `block_size` requires equal token block sizes across groups. These models have heterogeneous geometry; keep the default or use `blocks_per_chunk` until inspected. Generic source capability is not model correctness certification. [Pinned group handling](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py)

**Swarm adjustment required before a RAM experiment:** on CUDA this native CPU tier uses a shared mmap under `/dev/shm`. The existing 8-GiB Gemma / 64-GiB DeepSeek tmpfs mounts cannot contain, for example, a 768-GiB / 256-GiB CPU tier. Increase each mount to its budget plus overhead and retain host/cgroup/memlock headroom. [Pinned allocator](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/v1/kv_offload/cpu/shared_offload_region.py)

That native shared region is engine-owned: its backing file is unlinked after mapping and the memory is reclaimed when the last worker exits. It is not a durable cache across engine or node restarts. NVMe persistence and rediscovery, or an external cache service that outlives the engine, require separate validation. [Region lifecycle](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/v1/kv_offload/cpu/shared_offload_region.py)

For the selected dedicated nodes, the prepared candidates use **768 GiB Gemma / 256 GiB DeepSeek per engine**, not a measured optimum. The earlier 128-GiB Gemma suggestion was a small compatibility experiment; dedicated Gemma already has 448 GiB of GPU KV across TP4, so a smaller CPU tier may add little retention. The new budget is deliberately larger. See the [opt-in candidate overlays](dedicated/ram-kv/README.md). Reserve the measured ~188.8 GiB Engram allocation plus process/model-loading/OS memory. Do not preallocate all 2 TB. Enlarge after hit-rate and transfer measurements. Checkpoint payloads occupy approximately **544.58 GB shared**, **510.31 GB DeepSeek-only**, or **34.27 GB Gemma plus assistant**. A nominal 1-TB NVMe is not a free 1-TB KV tier. Reserve checkpoint, image, compilation and logging space; use a separate bounded KV allocation.

## Compatibility matrix

| Path | Exact pinned source / published evidence | Status for our exact configurations |
| --- | --- | --- |
| Native GPU APC | Already enabled in the measured runs | Serving exercised; cache-hit performance not separately benchmarked |
| Native CPU/FS OffloadingConnector | Pinned source advertises hybrid support and accepts grouped layouts | Unverified for both models with current speculation and images |
| Bundled LMCache MP | Pinned connector rejects multiple KV groups | Do not use as a drop-in for these hybrid configurations |
| External LMCache MP + Gemma4 | Published Google 31B TP2/MTP1/Triton validation | FP8-block/FP8 KV/TP4/MTP4/images on this image unverified |
| External LMCache MP + DeepSeek | Published V4 Flash fp8_ds_mla/TP4/MTP1 validation | V4.1/Engram/DSpark5/images on this image unverified |
| Native P2P / shared storage | Generic tier/protocol exists | Cross-node model compatibility and Python-router integration unverified |

The pinned V2 model runner registers the engine KV tensors with the connector. The inspected config, V2 runner integration, and model files contain no blanket rejection of this speculation-plus-connector combination, but this does **not** establish correct draft-state restoration or support for each custom sparse/FP8 layout. One explicit config restriction is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: a connector is rejected unless the CuMem allocator is enabled. The canonical stack currently omits that allocator flag; do not restore it from the older DeepSeek draft while adding a connector. [Pinned config validation](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/config/vllm.py#L1091-L1132), [V2 connector registration](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/v1/worker/gpu/kv_connector.py)

## Candidate native CPU experiment, not a production config

Append this fragment to a **copy** of the Gemma engine YAML; retain its tested GPU KV budget:

```yaml
kv-transfer-config:
  kv_connector: OffloadingConnector
  kv_role: kv_both
  kv_connector_extra_config:
    cpu_bytes_to_use: 824633720832  # 768 GiB total across this engine's workers
    offload_prompt_only: true
```

For a later DeepSeek experiment, 256 GiB is `274877906944` bytes. These are bounded starting allocations, not measured optimums. The prepared overlays use 800 GiB Gemma / 320 GiB DeepSeek `/dev/shm` ceilings. These are not separate RAM allocations on top of the tier. Increase each service's tmpfs before startup. Do not change both services at once during initial correctness testing. The aggregate budget semantics are verified in the pinned CPU allocator. [Pinned CPU spec](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/v1/kv_offload/cpu/spec.py#L93-L120)

## LMCache compatibility is encouraging but conditional

Official Gemma4 documentation validates Google Gemma4 31B + assistant, TP2, MTP one token, Triton, including hybrid groups and draft KV retrieval. It reports 3.7× cold-to-warm TTFT improvement for that setup. It does not certify RedHatAI FP8-block + FP8 KV + TP4 + MTP4 + images on our exact image. [Gemma4 recipe](https://docs.lmcache.ai/recipes/gemma4.html)

The DeepSeek recipe validates **V4 Flash**, sparse MLA / `fp8_ds_mla`, and TP4 MTP one token. It does not validate **V4.1 + Engram + DSpark5**, nor this image/multimodal combination. Treat it as evidence of feasibility, not a drop-in guarantee. [DeepSeek V4 recipe](https://docs.lmcache.ai/recipes/deepseek_v4_flash.html)

Crucial: the **bundled** MP connector in our pinned vLLM rejects multiple KV groups. Do not solve this by disabling the hybrid cache manager: it changes the measured memory behavior. [Pinned bundled connector](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L70-L82)

Modern LMCache hybrid support requires the external connector module `lmcache.integration.vllm.lmcache_mp_connector`, compatible LMCache server/client and matching Torch/CUDA native ABI. Build/test a derivative image rather than mutate the validated baseline. [Compatibility guide](https://docs.lmcache.ai/getting_started/compatibility.html)

## Dedicated versus replicated deployment

Dedicated model/node: there is one destination/model, so replica routing gains disappear. Local APC plus a RAM tier still helps revisited chats/documents. DeepSeek and Gemma must have separate cache namespaces; their KV is not interchangeable.

Both models on both nodes: use model-specific backend pools and session affinity; each node maintains its own cache. Failover can recompute a missing prefix. Cross-node sharing can reduce this cold penalty only for compatible instances and adds network/controller dependencies; it does not migrate an interrupted generation or restore model availability by itself. The native P2P tier also needs orchestration to select peers and supply transfer metadata—the current Python router does not automatically gain that protocol simply by enabling the tier. [Native P2P protocol](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/#orchestration-layer-protocol)

## What to test before promoting external KV caching

For each exact model/YAML/speculation combination: cold store → same-engine warm reuse → force GPU eviction/reset → RAM retrieve → restart/cross-process retrieve. Include long text, repeated images, changed images, changed early prefix, distinct requests at C4, and output/acceptance-rate checks. Measure token hit coverage by tier, TTFT p50/p95, decode/ITL p50/p95, throughput, bytes moved, CPU pinned memory, GPU peaks/preemptions, plus cache miss fallback. Only then compare RAM versus NVMe or remote sharing. No cache-specific speedup has been measured on this deployment.

## Ranked next steps

1. Keep measured engine settings; stabilize prompts and measure APC token hit rate and TTFT on real workloads.
2. For replicas, establish model-specific session affinity and retain a release with the audited behavior; avoid a blind switch to prompt-only KV-aware lookup for chat/images. Dedicated nodes need no replica affinity.
3. Validate native RAM reuse one model at a time, including eviction/retrieval and images, then retest shared C4 latency and memory.
4. If native correctness or performance is inadequate, evaluate an explicitly pinned external LMCache MP build; repeat the same evidence gates.
5. Add NVMe only for a reusable working set that exceeds the chosen RAM tier. Add cross-node KV only when measured failover/cache-miss cost justifies transport and controller complexity.
