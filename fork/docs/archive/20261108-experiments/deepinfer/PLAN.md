# DeepInfer versus vLLM: October 17 checkpoint

> Closed on 2026-11-08. Historical preparation only; no GPU comparison was run.
> See the [closeout notes](../README.md).

Status on 2026-10-20: **preparation only; no GPU measurement or paid rental**.
The requested H200 comparison has a hardware compatibility blocker.
Do not interpret this document as a completed benchmark.

## Feasibility finding

Released DeepInfer cannot currently supply the requested H200 comparison.
Its [README at the inspected commit][deepinfer] supports NVIDIA SM100 and
Ascend 950, and exposes offline inference. Its pinned FlashMLA dependency
[explicitly removed Hopper support][flashmla]. The [open SM90 proposal][sm90]
contains a design document, not a working implementation. Porting it would
require attention and quantized GEMM work; changing an architecture flag or
installing an older FlashMLA is not an established solution.

A comparison on four B200s could measure the engines on the same supported
hardware. It would not establish a speedup for four H200s, nor the effect of
PCIe-only collectives. H200 vLLM measurements would be a separate hardware arm.

DeepInfer also does not include the complete distributed service:
external KV storage, prefill/decode disaggregation integration, and replica
load balancing are missing. Its internal scheduler does have live-request KV
management. An instrumented offline run can establish engine latency and
throughput; it cannot certify HTTP serving, production routing, cancellation,
or multi-user reliability. A serving adapter would be an explicit additional
experimental component, with its overhead measured separately.

The published DeepInfer curves use four SM100 GPUs for prefill and 64 for
decode. They do not establish four-GPU decode performance. The [deployment
Q&A][qa] recommends wide expert parallelism for decode. That recommendation
is not a claim that smaller offline deployments cannot run.

## Pinned inputs

| Component | Revision |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Model commit | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| Latest stable vLLM checked | `v0.32.0`, released 2026-10-18 |
| vLLM source tag commit | `5df63a7203` (full SHA in `provenance.json`) |
| vLLM image index | `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34` |
| vLLM image, Linux amd64 | `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` |

The model index reports **510,286,023,000 bytes (475.24 GiB)** of tensor
payload across 48 safetensors shards; headers, tokenizer files, images,
packages, build artifacts and runtime caches require additional disk.
Only metadata and tokenizer assets were downloaded during preparation.

The new checkpoint's `config.json` and `tokenizer_config.json` are
byte-identical to the existing deployment's pinned
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`.
This establishes configuration compatibility, not output equivalence or a
successful engine boot.

DeepInfer's build currently pins Python 3.12 requirements including
PyTorch `2.13.0+cu132` and NCCL `2.32.3`. Its pinned FlashMLA requires
CUDA 13.1 or newer and recommends 13.2 or newer. Resolve, build and archive
the dependencies before a full-model timed campaign; record the resulting
environment rather than relying on mutable package names.

## Existing baseline and its limits

Reuse [the dedicated DeepSeek recipe][recipe] and its [environment][environment].
It uses TP4 plus expert parallelism, CPU-offloaded Engram, FP8 KV, a 36 GiB
KV budget per GPU, a 1,048,576-token context window, 32 scheduler slots,
4,096-token prefill chunks, and DSpark with five draft tokens and adaptive
verification. Its collective workarounds belong to the baseline.

The [previous measured record][baseline] reports 491.5 output tokens/s for
four active requests over 121.4 seconds on H200/NVLink, using approximately
1K input / 256 output tokens. It reports 129.76 GiB peak GPU memory and a
17.90x startup KV-capacity estimate. That throughput includes prefill and
drain; it is not isolated decode speed.

The user's approximate 170–200 tokens/s at one stream and 100 tokens/s per
stream at eight users are hypotheses to investigate, not benchmark samples.
The 17.90x startup number estimates full-context cache capacity. It is neither
the scheduler's sequence limit nor evidence that 17–18 users meet a latency
objective continuously. There is no valid measured DeepInfer/vLLM ratio yet.

## Measurement contract

Use the same physical GPU allocation sequentially for the primary engine
comparison. Pin the same checkpoint, token IDs, sampling parameters and output
limits. Record native precision and actual KV layouts; do not attribute a
precision change solely to scheduling. Retain each engine's supported
parallelism and identify it explicitly: vLLM TP/EP and DeepInfer DP/EP are
different layouts.

Use these separately labelled workloads:

| Workload | Input / output tokens | Purpose |
| --- | --- | --- |
| Short interactive | 1,024 / 1,024 | Single-stream and concurrent decode |
| Chat/code | 8,192 / 1,024 | Representative prefill plus generation |
| Retrieval | 32,768 / 512 | Longer prefill under concurrency |
| Long context | 131,072 / 256 | Memory and TTFT sensitivity |
| Maximum context | Actual prompt + output <= 1,048,576 | Capacity proof, separate from throughput |

Run a fixed-length synthetic workload to control token counts and a fixed
natural-language/code corpus to measure realistic speculation. Archive corpus
bytes, their license and digest, tokenizer revision, encoded token lengths and
request seeds before renting. Do not use repetitive filler or random tokens
as the only evidence of draft acceptance.

Start concurrency at 1, 2, 4, 8, 12, 16, 18, 24 and 32. Extend to 48 and
64 only if memory and measured latency justify it. Give each completed request
an immediate replacement: a single batch that drains is not sustained load.
Do not distribute one request to every DP rank and call that C1; concurrency
is the global number of requests across all replicas.

Warm up model execution and compilation before timing. Record cold startup
and cold first-request latency separately. Use short exploratory sweeps to
locate useful points, then at least ten minutes at each finalist with three
repeats. Record the observed duration and completed-request count. Do not label
shorter runs as certified sustained capacity.

Measure:

- TTFT from request submission to first generated token, with p50/p95/p99.
  HTTP results include server queueing and transport; native-engine timings
  must be labelled separately. Reasoning tokens count as generated tokens.
- Per-request TPOT as `(last-token time - first-token time) / (output tokens - 1)`.
  Report its distribution and per-request reciprocal decode speed, including
  p5 speed. Exclude one-token responses from decode-speed statistics.
- Stream delivery gaps separately from TPOT. A speculative burst can contain
  several tokens; one SSE event is not necessarily one token.
- Aggregate emitted output tokens/s over the measured wall-clock interval,
  input tokens/s, requests/s, failed and incomplete requests, and token counts.
  Distinguish the steady-state window from startup and final drain.
- Natural draft acceptance and actual emitted output, memory by GPU, host RAM,
  utilization, power where available, cache hits, preemption and queue growth.
- Goodput and highest sustained concurrency for a matrix of proposed SLOs:
  50 and 100 output tokens/s per request, short-input TTFT <= 2 seconds,
  and 8K-input TTFT <= 5 seconds. These are experimental thresholds, not
  user-specified service requirements. Show tail results and errors alongside
  the qualifying fraction; do not report a universal maximum-user count.
- Compute dollars per million emitted output tokens at each measured load,
  plus elapsed setup/download cost and transfer charges separately.

Use bounded open-loop arrival-rate tests around the closed-loop knee to detect
queue growth. Capture timeouts, OOMs, rejection, client-side queue time,
server-side waiting, and cancellations rather than dropping those samples.

Run uncached and controlled-prefix-hit profiles separately. A warm compilation
cache is desirable; accidental prompt-prefix reuse changes the workload.
Keep errors and partial outputs in the evidence and prohibit overlapping
benchmark drivers on the same allocation.

## Reuse and experiment controls

vLLM `bench serve` in v0.32.0 already provides concurrency limits, request rates,
warmups, detailed TTFT/TPOT/ITL/end-to-end results and goodput thresholds.
Reuse it for HTTP measurements. Choose enough requests to cover the declared
duration, verify actual elapsed time, and use its detailed request timestamps
for the steady-state window. The short release-gate probes in `fork/bench/perf.py`
do not on their own establish this campaign's sustained-user capacity.

DeepInfer's [provided benchmark][microbench] is useful as a kernel/engine
diagnostic, not as the headline service measurement:

- It sets `DEEPINFER_FORCE_LOAD_BALANCE=1`, uses random token inputs, and
  separates prefill from a short decode measurement.
- Its reported decode token rate multiplies step rate by `1 + nextn`;
  the source labels this as a full-acceptance convention. Record actual
  `RequestOutput` token emissions for a user-facing throughput result.
- Its default request count is per DP rank, so global concurrency needs explicit
  accounting.

For an adapted Blackwell experiment, first run upstream offline correctness
and structured-output checks, then instrument actual request output timestamps
and repeated admissions. Validate any adapter against deterministic CPU traces
that include multi-token bursts, empty output, errors, early EOS and truncated
requests before paying for a full-model run.

Keep the first comparison small: matched baseline, speculation off/on, then
only the parameters implicated by the measured bottleneck. Candidate controls
include prefill chunk size, DSpark draft length/adaptive verification, KV
allocation, and supported attention/MoE backends. Separate the inherited
deployment configuration from a topology-specific tuned configuration.

vLLM v0.32.0 includes relevant Engram CPU lookup, decoder replay and DSpark
changes in its [release notes][vllm-release]. Their presence motivates a new
measurement; upstream speedup claims are not results on this allocation.

## Cost and rental gates

Campaign type, if resumed: **development/tuning**. Keep one watched rental and
its checkpoint/compiler caches across configuration failures. A failed model
process does not trigger a new checkpoint download on a new host.

The property requiring expensive hardware is real-checkpoint memory fit and
native kernel performance. Token construction, metric arithmetic, configuration
handling and evidence collection remain CPU work. An inexpensive Hopper or
consumer Blackwell GPU does not validate an SM100-only kernel path.

Read-only Vast offer checks on 2026-10-20 returned these indicative totals:

| Shape | Reserved disk | Advertised total / hour |
| --- | --- | --- |
| 4 x H200 | 800 GB | $20.44 |
| 2 x B200 | 800 GB | $22.29 |
| 4 x B200 | 1,000 GB | $43.80 |

These are quotes, not spend, and include different disk reservations. Availability,
storage, transfer billing units, driver compatibility and interconnect must be
rechecked before create. The 2-B200 offer is a possible memory-fit development
venue, not a validated fit or substitute for the four-GPU comparison. Do not
reserve it until a source-level memory estimate and build plan justify it.

Before any create:

1. Resolve whether the deliverable is a Blackwell engine comparison or an
   H200 service improvement; no released DeepInfer H200 arm is available.
2. Finish the executable probes, corpus and cheap correctness checks; validate
   configurations against the actual engine parser/build.
3. Record an explicit hourly ceiling, hard duration cap, total cost envelope,
   expected transfer bytes, transfer price units and measured/advertised rate.
4. Verify adequate host RAM, disk, GPU memory, drivers and topology.
5. Update the shared coordination notice and arm the existing external watchdog
   for the exact label `fork-deepinfer-v41-dev-20261020` before create.
6. Observe cache reuse and transfer bytes/rate; collect evidence after each
   iteration. At session end destroy only this label's rentals and verify
   provider state. Never sweep unrelated V100 work.

There is no cross-session messaging channel available in this thread.
Coordination notices were left in the main, V100 and experiment worktrees'
ignored `runs/` directories. Delivery/acknowledgment by the other agent is
unconfirmed. Only an exact-label cleanup is safe.

## Preparation evidence

- Isolated worktree and branch `fork/deepinfer-v41-benchmark`.
- DeepInfer source and dependency pins inspected without modifying upstream.
- Official model metadata downloaded and compared to the deployment pin.
- vLLM latest stable release and registry manifest digests resolved.
- Existing dedicated recipe, environment and measured limits identified.
- `bash fork/bench/preflight.sh`: **PREFLIGHT GREEN**, 589 passed, 8 skipped,
  followed by the full mock gate.
- Rental searches and account observations were read-only. This session has
  created no paid resource; the unrelated V100 rental was not changed.
- No new TTFT, throughput, decode, concurrency or GPU-memory results exist.

[deepinfer]: https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
[flashmla]: https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/README.md
[sm90]: https://github.com/deepseek-ai/DeepInfer/pull/47
[qa]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md
[recipe]: https://github.com/Mazyod/vllm/blob/6ff072552aeb80d72cae8ec924f3874be408a544/fork/deploy/deepseek-gemma4-dedicated/deepseek.yaml
[environment]: https://github.com/Mazyod/vllm/blob/6ff072552aeb80d72cae8ec924f3874be408a544/fork/deploy/deepseek-gemma4-dedicated/stack.yaml
[baseline]: https://github.com/Mazyod/vllm/blob/6ff072552aeb80d72cae8ec924f3874be408a544/fork/deploy/deepseek-gemma4-dedicated/README.md
[microbench]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py
[vllm-release]: https://github.com/vllm-project/vllm/releases/tag/v0.32.0
