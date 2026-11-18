# Native Whisper DP2: four versus eight sequences per GPU

The [single-service DP2 profile](../whisper-dp2.yaml) now uses
`max-num-seqs: 8`. This allows up to sixteen Whisper sequences across two GPUs.
The image, model revision, two-GiB KV budget per worker, and other engine settings
are unchanged. Deployment remains one Whisper service with both GPUs visible,
plus BGE on GPU 0 and Snowflake on GPU 1, one instance each.

## Paired result

Both settings ran on the same warm two-V100 16 GB rental with identical inputs
and background pooling traffic. Only the per-rank sequence cap changed.

| workload | HTTP concurrency | 4 sequences/rank (audio seconds/s) | 8 sequences/rank (audio seconds/s) |
| --- | ---: | ---: | ---: |
| Short clips | 1 | 11.00 | 11.10 |
| Short clips | 4 | 24.11 | 24.65 |
| Short clips | 8 | 39.72 | 42.01 |
| Short clips | 16 | 46.41 | 69.13 |
| Long uploads | 1 | 57.68 | 56.27 |
| Long uploads | 4 | 63.71 | 115.14 |

Under load, eight improved short-clip throughput by **49%** and long-upload
throughput by **81%**. Short-clip C16 p95 latency improved from 2.73 to 2.28 s.
There was little benefit at low concurrency; a larger cap is not a universal
latency improvement or a promise of linear scaling.

Observed memory peaks increased from 8031/8023 MiB to **8075/8047 MiB** across
the two GPUs. Both settings recorded zero cache preemptions. Each rank handled
103 successful engine requests per trial; these include audio chunks and are
not the count of HTTP uploads.

## Verification and scope

[Structured evidence](20261118-whisper-dp2-batching.json) records the exact
config hashes, measurements, per-rank counters, memory samples and cleanup.
All **328 ASR calls** including warmups and **34 background BGE/Snowflake calls**
passed the same saved independent FP32 fidelity checks as the previous study.
Pooling shapes were also validated before each timed run.

The unchanged harness used 32 English/Arabic short clips and two roughly
190-second long fixtures. Short cells measured 32 requests each; long cells
measured two uploads at C1 and four at C4. A warmup cohort was excluded at each
concurrency. Timings sum synchronized HTTP cohort spans, with reference
validation outside the timed intervals. Background traffic alternated BGE and
Snowflake at 12 requests/minute combined, including full 8192-token inputs and
larger batches.

This is a new rental, so compare four and eight within this run; absolute
comparisons with the earlier independent-worker measurements are not controlled.
Warm reused fixtures and loopback HTTP do not establish a production SLA.
The original fine-tune accuracy limitations and explicit-language JSON contract
still apply. The experiment stopped after this two-setting comparison.

## Artifact and venue

- Image: `openimage/vllm-audio@sha256:ddbb02258092f1888e19654e0757744e70d52be2c36c01584025a5ab5106dc5e`.
- Current config SHA-256: `0b1afedb639a5300189e27f823b7550c51ab28120fec18fb57dae74fc0a2e5a6`.
- Control config SHA-256: `910cce3f28d9aa3d37eb97928f39a958a0415b2bc73e0a1bed66ca00d04e2240`.
- Two Tesla V100-SXM2-16GB GPUs, SM70; driver 580.173.02; observed link `PHB`,
  with no NVLink reported.
- vLLM 0.32.0+cu126, PyTorch 2.13.0+cu126, CUDA 12.6, TorchAudio 2.11.0+cu126,
  Triton 3.7.1, Transformers 5.19.0. No serving dependency was changed.

The watched development rental retained checkpoints and compiler caches across
both launches. Final model shutdown left both GPUs at 0 MiB usage. Provider and
external watchdog independently confirmed destruction. No rental remains.
Elapsed compute/storage through verified teardown was 21.36 minutes at
$0.3274/hour, a full-rate bound of approximately **$0.117**, excluding transfer.
Measured guest staging bytes/rate and inbound price are in the structured
record; provider-side image-pull traffic is outside those counters.
