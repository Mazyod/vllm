# Initial Whisper scaling on two V100 16 GB GPUs

**Highest throughput in this initial comparison:** two independent Whisper
workers, one per GPU, using [whisper-throughput.yaml](../whisper-throughput.yaml).
Run one BGE service beside the first worker and one Snowflake service beside the
second. The new profile changes only `max-num-seqs` from 4 to 8. The frozen image
and model revisions are unchanged.

The four-sequence native-DP configuration measured here is preserved at
[its original commit](https://github.com/Mazyod/vllm/blob/df6504d193b02106b85c7e18f897f6ce6c130a93/fork/deploy/volta/whisper-dp2.yaml).
The [current native DP2 profile](../whisper-dp2.yaml) uses eight sequences
following the [paired batching comparison](20261118-whisper-dp2-batching.md).

[Structured measurements](20261118-whisper-scaling.json) bind the configurations,
image, software, quality checks, memory observations and cleanup receipt.

## Same-host comparison

| layout | Whisper sequences per worker | short clips, C16 (audio seconds/s) | long uploads, C4 (audio seconds/s) | peak GPU memory (MiB, GPU 0 / 1) |
| --- | ---: | ---: | ---: | --- |
| One Whisper on GPU 0; BGE + Snowflake on GPU 1 | 4 | 21.95 | 26.85 | 6097 / 3891 |
| Two independent Whisper workers; BGE on GPU 0, Snowflake on GPU 1 | 4 | 41.43 | 51.24 | 8029 / 8007 |
| Native vLLM DP2; BGE on GPU 0, Snowflake on GPU 1 | 4 | 32.48 | 51.33 | 8029 / 8025 |
| **Two independent Whisper workers; balanced pooling placement** | **8** | **49.41** | **73.11** | **8047 / 8051** |

The selected configuration achieved **2.25x** the baseline short-clip throughput
and **2.72x** its long-upload throughput. Short-clip C16 p95 latency fell from
6.96 to 3.09 seconds in this sample. A single long upload took 3.50 seconds for
the English fixture and 5.78 seconds for Arabic, versus 5.77 and 11.00 seconds
in the original layout.

Native DP2 also passed. Its metrics recorded 103 successful engine requests on
each rank, confirming that both GPUs handled work. It was tested with four
sequences per rank; native DP2 with eight was evaluated in the linked follow-up,
not this initial comparison. These results do
not establish an architectural throughput ceiling for either approach.

## Placement of the throughput-oriented alternative

| process/container | instances | visible GPU(s) | engine config |
| --- | ---: | --- | --- |
| Whisper A | 1 | 0 | `whisper-throughput.yaml` |
| Whisper B | 1 | 1 | `whisper-throughput.yaml` |
| BGE | 1 | 0 | `bge.yaml` |
| Snowflake | 1 | 1 | `snowflake.yaml` |

Both Whisper endpoints serve the same `whisper` model alias. Balance requests
between them through the deployment's router. The experiment used round-robin
request assignment with balanced language ordering.

Container/service replica counts are separate from vLLM engine parallelism.
For explicit GPU binding, use one instance of each Whisper service above. A
single service with two replicas is equivalent only if its deployment mechanism
assigns each replica a different GPU. Two replicas with the same fixed GPU
binding both use that GPU.

The independent workers use default TP1 and DP1. Do not additionally set native
DP2 on each copy. For the alternative native-DP setup, use one Whisper service
with both GPUs visible and `--config whisper-dp2.yaml`.
v0.32.0 defaults that internal-DP launch to two API server processes. See the
[versioned upstream DP guide](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/).

## Workload and validation

- Same immutable image: `openimage/vllm-audio@sha256:ddbb02258092f1888e19654e0757744e70d52be2c36c01584025a5ab5106dc5e`.
- Two Tesla V100-SXM2-16GB GPUs, SM70, driver 580.173.02. The observed
  interconnect was `NODE`; no NVLink was reported.
- vLLM 0.32.0+cu126, PyTorch 2.13.0+cu126, CUDA 12.6, TorchAudio 2.11.0+cu126,
  Transformers 5.19.0, Triton 3.7.1. No runtime package was changed.
- Reused 32 short English/Arabic FLEURS clips and two approximately 190-second
  long fixtures, with pinned audio hashes and saved independent FP32 references
  from the [initial qualification](20261117-gate.md).
- Short sweeps used concurrency 1, 4, 8 and 16 with 32 measured requests per
  cell. The final batching check repeated only C8 and C16. Long tests used two
  measured requests at C1 and four at C4. Warmup cohorts were excluded.
- Throughput sums completed audio duration over barrier-synchronized HTTP
  cohort spans; response validation is outside those spans. Requests used
  explicit language, JSON, and the full native decoder budget.
- BGE and Snowflake received alternating arrivals at 12 requests/minute combined,
  cycling through 64/4096/8192-token inputs and larger batches. Pooling shapes
  were validated and warmed before timing. All 94 background requests passed
  their saved FP32 vector/ranking checks.
- All 589 ASR calls, including warmups, passed the declared reference-fidelity
  checks. Native DP and independent copies both retained correct responses.
- GPU memory was sampled once per second through startup and traffic. Both GPUs
  returned to 0 MiB after final model shutdown.

This is a small warm-fixture, loopback-HTTP comparison. Long-upload throughput
has only two distinct source recordings; it is not a production latency or
sustained-load guarantee. Pooling correctness passed, but pooling latency rose
under Whisper contention. The background arrival rate does not certify
saturation of BGE and Snowflake simultaneously. The original fine-tune's
accuracy limitations and explicit-language JSON scope still apply.

## Cost and cleanup

One watched development rental retained model and compiler caches through all
four layouts. The initial pinned-model staging transferred 6,836,307,522 guest
bytes in 35.0 seconds, approximately 1563.6 Mbit/s. Image-pull traffic is outside
these counters. The rate was $0.3941/hour and $0.00390625/GB inbound.

Elapsed time through verified teardown was 44.87 minutes, a full-rate
compute/storage bound of **$0.295**, excluding transfer. The provider and
external label watchdog both confirmed destruction. No rental remains for
this comparison. Private identities and raw transcripts are excluded.
