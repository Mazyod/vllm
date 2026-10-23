# Two 16 GB V100 full-model campaign

The candidate image reuses the audited [SM70 runtime](../v100-modern/README.md)
and changes Whisper to native vLLM with the pinned `mlmlaziz/bbyn-v2` checkpoint.
All model revisions are in [models.json](models.json). GPU 0 holds Snowflake,
BGE, FP32 EmbeddingGemma and FP16 Whisper. GPU 1 holds Jina v4 retrieval
multivectors. Every process has exactly one visible GPU and TP1; no GPU link is
used by inference. Full contexts, sliding attention, image resolution and the
128-dimensional per-token Jina outputs are preserved.

The image under test has digest
`sha256:5a1adb39d075eff377372c877a36bef30ecc3f82896633a15036405673d7e77b`,
built from source commit `4b1bc89f03` and base digest
`sha256:0151cd685a671f16083a29c894dfa6a9f0b6eddeef4c9772d3ede172dd22cd44`.
An authenticated, read-only temporary registry delivers it to Vast; checkpoints
and account credentials are not included in that image. Model workers run as
UID 10001. The provider's SSH bootstrap runs as root. This tests the actual
image and model processes, but not Docker Compose isolation inside the rental.

## Evidence and quality gates

`prepare_cases.py` binds 295 requests to existing independent FP32 oracles:
12 text embedding/reranking cases, nine Jina cases and 274 English/Arabic ASR
cases. Supplemental WAV/MP3/AAC probes cover short and three-minute audio in
both languages, using independent FP32 references for each decoded waveform.
The campaign fingerprints requests, reference arrays, audio and the reference
environment. Only reference calculations are reused: every serving response
is generated again. Image requests receive fresh UUIDs to force vision work.

- Embeddings: correct shape, finite unit vectors and minimum cosine 0.999.
- BGE: score error at most 0.01 and matching top-ranked document.
- Jina: correct token-vector dimensions and unit norms; mean cosine at least
  0.995 for text and 0.99 for images. Three MaxSim ranking comparisons must agree.
  Minimum and first-percentile cosines are retained; mean agreement is not a
  guarantee for every token or an unseen relevance dataset.
- ASR: explicit language and ordinary JSON. Short-case errors may exceed the
  FP32 reference by at most one word; long cases by at most two percentage
  points. Short-corpus WER regression must not exceed 0.5 percentage points per
  language. Missing speech when the reference is nonempty and silence
  hallucinations fail. Absolute error counts and transcripts remain visible.

These are reference-fidelity gates, not a claim that the reference is infallible
or that a public corpus certifies production-domain accuracy. ASR long fixtures
are held-out utterance composites, with boundary-shift copies. No context or
audio is shortened to pass a gate.

## Measurements

`benchmark.py` provides three modes:

- `quality`: all selected requests plus independent-reference comparisons.
- `sweep`: synchronized concurrent HTTP cohorts at 1/2/4/8, adding 16 for Jina.
  Short, medium, full-context and batched requests are measured separately;
  Jina includes one/two/eight high-resolution images and mixed cohorts.
- `mixed`: scheduled arrivals across all five services, measuring offered and
  completed rate, client launch delay and drain time. Planned rates are 10, 30
  and 60 total requests/minute; the 30-RPM acceptance window is 20 minutes.

Every response is validated. The report exporter also requires server token
accounting to match the complete requested input for every successful pooling
and reranking response. Each sweep cell first runs one excluded, validated
warmup cohort to keep shape-specific JIT outside steady measurements. Cohort
throughput excludes the validation gaps;
HTTP latency includes body reception but excludes JSON/vector validation.
Validation time is recorded separately. Interval measurements use monotonic
time; wall-clock timestamps align them with resource samples. p95 is emitted
only with at least 20 samples and p99 only with at least 100. Low-sample tails
must not be presented as reliable SLA estimates.

Metrics include successful requests/s, items/s, input tokens/s, audio seconds/s,
ASR real-time factor, images/s, response bandwidth, latency, errors and numerical
quality. One-second GPU samples retain VRAM, utilization, power, temperature and
clocks. Host samples retain serving-process PSS/RSS and CPU use. These are sampled
footprints, not guaranteed maxima. Each benchmark records the active launch
receipt and verifies the configuration digest before issuing requests.

Start with the full residency/quality gate, then measure the GPU 0 services
one at a time while Jina runs its sweep on GPU 1. Host CPU and memory samples
cover both GPUs' services. Repeat with all five models under mixed contention, and test Jina client concurrency separately from
the engine's active-sequence limit. Any sequence-limit tuning keeps the same
32,768-token batch budget and full input contracts. Failures retain their logs;
only model processes stop during tuning. The final fixed configurations must
pass a fresh process start and acceptance replay on the warm development rental.

## Tools

`services.py` owns distinct process groups, nonroot launch, health checks,
full-shape warmup and shutdown. `check_lifecycle.py` verifies that a worker
ignoring TERM is killed even if its parent exits first. `hardware_check.py`
checks actual 16 GB SM70 devices, a 12 GiB allocation, matrix math, the native
vLLM RMSNorm extension and efficient SDPA before model staging.

Benchmark-only dependencies live outside `/opt/vllm`; the launcher removes
their `PYTHONPATH` before spawning any model. Keep one watched rental/cache
through configuration failures. An external label watchdog must be armed before
create, and provider-confirmed teardown is mandatory at the end.

## Deployment candidate

Use the candidate image with the existing five-service Compose layout and the
Jina cache override. Compose resolves paths relative to the first file:

```bash
cd fork/deploy/v100-campaign
export V100_IMAGE=local/vllm-v100:campaign-20261023
export MODEL_DIR=/absolute/path/to/pinned/models
export VLLM_API_KEY='replace-with-your-secret'
docker compose -f ../v100-modern/compose.yaml -f compose.override.yaml config --quiet
```

Stage the exact manifest on a trusted machine with Hugging Face access to the
gated checkpoints, then mount that directory read-only:

```bash
uv run --no-project --with huggingface-hub python ../v100-modern/stage-models.py \
  --root "$MODEL_DIR" --manifest models.json \
  --names snowflake bge gemma whisper jina
```

The manifest selects native `mlmlaziz/bbyn-v2` for Whisper. Send `language=en` or `language=ar` and
`response_format=json` explicitly for ASR. The override caps Jina's cache at
1.5 GiB while retaining its full context and image limits. Deployment acceptance
requires the campaign results and the final digest's Prisma policy review;
Compose container isolation is not exercised inside the Vast rental.

Before any development rental, require the offer's absolute `end_date` to be
later than the watchdog deadline plus an hour. Provider expiry can stop a
non-interruptible offer too. Keep receipts under ignored `runs/`, collect
incrementally, and check actual transferred files rather than trusting a copy
command's exit status. Provider copies into an image's lower-layer directories
can leave the running container with a stale overlay view; use a newly created
workspace destination for migrations.

## Reproduce the reports

From the repository root, with the retained fixtures and collected responses:

```bash
CAMPAIGN_RUN=runs/v100-full
PYTHONPATH=. uv run --no-project --with jiwer==4.0.0 python \
  fork/deploy/v100-campaign/summarize.py \
  --collected "$CAMPAIGN_RUN/dev3/collected" \
  --out fork/deploy/v100-campaign/results \
  --cases "$CAMPAIGN_RUN/oracles/cases.json" \
  "$CAMPAIGN_RUN/dev3/formats-short/cases.json" \
  "$CAMPAIGN_RUN/dev3/formats-full/cases.json"
uv run --no-project --with matplotlib==3.11.2 python \
  fork/deploy/v100-campaign/plot.py \
  --benchmarks fork/deploy/v100-campaign/results/benchmarks.csv \
  --out fork/deploy/v100-campaign/results/jina-concurrency.png
```

The exporter binds corpus statistics to the fixture fingerprint in each run,
keeps repeated quality runs separate, and excludes composite recordings from
the short-clip corpus. The plot is byte-reproducible with the pinned renderer.

The exact tested image is also exported locally under
`runs/v100-full/image/vllm-v100-campaign-20261023.tar.zst`, with its adjacent
SHA-256 checksum and receipt. Load that archive on the target host with
`docker image load -i vllm-v100-campaign-20261023.tar.zst`. The loaded image ID
must match the digest above. After staging the pinned models and setting the
Compose variables, launch from this directory:

```bash
docker compose -f ../v100-modern/compose.yaml -f compose.override.yaml up -d
```
