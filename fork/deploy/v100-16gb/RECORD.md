# V100 offload hardware record — 2026-10-20

**Development smoke passed** on `volta-2`: two Tesla V100-PCIE-16GB GPUs,
SM70, driver 580.126.09, PCIe `NODE` connectivity (no NVLink). This is a
capability record and contains no private host or account identifiers.
[Machine-readable evidence](evidence.json) includes the actual kernel checks,
latency samples, reference comparisons, and the Jina import failure.

## Result

| payload | result | warm p50 / p95 |
| --- | --- | --- |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | pass, FP16, V1/FlexAttention; normalized 1,024-dimensional vectors | 228 / 301 ms |
| `BAAI/bge-reranker-v2-m3` | pass, FP16, V1/FlexAttention; correct ranking and score parity | 231 / 246 ms |
| `google/embeddinggemma-300m` | pass **only up to 256 tokens**, FP32, V1/FlexAttention; normalized 768-dimensional vectors | 235 / 298 ms |
| `openai/whisper-large-v3` | pass, FP16, V0/XFormers; exact normalized JFK transcript | 757 / 782 ms |
| Jina plugin 0.1.1 | incompatible with vLLM 0.10.2; missing `pooler.tokwise` API | not measured |

All four passing services stayed loaded together: three text services on GPU 0,
Whisper on GPU 1. Ten rounds submitted one HTTP request to each service
concurrently; each embedding request contained four texts and each rerank
request three documents. Peak sampled memory was **4,688 MiB / 9,346 MiB**.
The latency percentiles exclude each service's first request and describe only
nine warm samples, not production SLAs. Initial JIT/request warmup took several
seconds; the final run reused compiler and checkpoint caches.

The independent reference uses Transformers 4.57.6, FP32 CPU inference, the
original checkpoints, their actual pooling/projection, and identical inputs.
Embedding acceptance requires per-vector cosine >=0.999; rerank scores require
absolute error <=0.01. Observed Snowflake cosine exceeded 0.999998, Gemma was
approximately 1.0, and rerank error was below 0.000001. Gemma also passed an
exactly 256-token comparison and rejected 257 tokens with HTTP 400.

## Limits established by failures

- V0 cannot run the pooling workloads in v0.10.2: the Snowflake launch failed
  because `RobertaEmbeddingModel` has no `compute_logits`. V1/FlexAttention
  resolved this without a source patch.
- Gemma's full 2,048-token configuration fails because this FlexAttention
  backend does not implement sliding-window attention. Disabling sliding with
  a 512-token cap is **not equivalent**: boundary cosine fell to about 0.994.
  The reference config converts its bidirectional window to radius 256, so the
  accepted cap is 256, including BOS/EOS and query/document prefixes.
- The serving image's Transformers 4.56.1 implements causal Gemma attention
  regardless of the checkpoint's newer bidirectional flag. It gave an invalid
  reference (cosines around 0.74–0.80); references use a separate 4.57.6 venv.
  The serving configuration explicitly sets `is_causal: false`.
- Jina 0.1.1 failed an actual import on the V100 rental. No Jina checkpoint was
  downloaded and no text/image inference or FP16 parity is claimed.

No upstream code was patched. The tested image stack is the pinned v0.10.2
amd64 base plus the nine packages in `requirements-audio.txt`. The locally
built `local/vllm-v100-audio:0.10.2` has exactly the same 189 package versions
and passes `uv pip check`. It was not published or rebooted as a new rental;
GPU execution used the base extended in place with the same packages.

## Scope and cost

This proves a small concurrent API workload, not production throughput,
multilingual ASR, long audio, Jina support, or 2,048-token Snowflake/BGE parity.
The JFK fixture is public and its checksum is recorded in the run instructions.
Exact engine YAMLs, model/tokenizer revisions, and runnable checks accompany
this record. Raw logs and full vectors remain under ignored
`runs/v100-16gb-dev/`.

One watched Vast development rental was used. The watchdog was armed before
create; the envelope was $0.50/hour maximum, a two-hour hard cap, and 30 GB
expected transfer. The selected rate was $0.262222/hour including 80 GB disk;
ingress was $0.002604167/GB and egress $0.00390625/GB. Checkpoints were staged
once and reused through all configuration failures. Final committed-byte replay,
elapsed time, transfer accounting, and verified teardown will be appended when
the development session closes.
