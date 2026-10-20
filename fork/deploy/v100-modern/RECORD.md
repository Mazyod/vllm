# V100 full-payload validation — 2026-10-20 UTC

This record concerns the anonymous `volta-2` profile: two Tesla V100-SXM2-16GB
GPUs, SM70, 16,384 MiB reported per device, driver 580.173.02, and a NODE PCIe
interconnect without NVLink. It is not a statement about a private deployment's
ownership or location. The image is Linux/amd64.

## What is deployed

Four services use the vLLM v0.32.0 source build with PyTorch 2.13.0+cu126 and
CUDA 12.6. Only vLLM's native extensions were compiled; the PyTorch, CUDA and
Triton components are published packages. The native wheel and assembler were
executed on SM70 and fingerprinted against the built image.

GPU 0 holds Snowflake Arctic Embed L v2.0, BGE reranker v2 m3, FP32
EmbeddingGemma 300M, and FP16 Whisper large v3. GPU 1 holds Jina v4 retrieval
multivectors. Whisper uses faster-whisper 1.2.1 / CTranslate2 4.8.3 in the same
image. The exact model revisions, service configurations, dependency locks,
API adapter and attention plugin are committed beside this record.

## Quality gate

- Snowflake and BGE: actual 8192-token inputs; EmbeddingGemma: actual 2048-token
  inputs with native sliding attention. Independent FP32 references checked
  dimensions, normalization, scores and ordering. Mixed embedding batches and
  32-candidate reranking were also checked. Text inputs are real NIST content.
- Jina: actual 32768-token text, `[32768,128]` output, and one/two/eight full
  resolution images. Eight images produced `[32277,128]` output. Each page
  used approximately 3.16 million processed pixels under `max_pixels=3211264`.
  Text token IDs matched the serving tokenizer; image preprocessing tensors
  matched the canonical reference exactly. All three tested MaxSim top results
  agreed with the independent FP32 model.
- Jina comparisons use mean row-cosine floors of 0.995 for text and 0.99 for
  images, alongside finite/unit-norm checks. The measurements retain minimum
  and first-percentile cosine too: outputs are not bit-identical, and the mean
  must not be mistaken for a per-token lower bound. Three ranking queries are
  a regression check, not certification of an unseen production corpus.
- Whisper: 74 English/Arabic cases, including four distinct recordings longer
  than three minutes, two using held-out utterances. WAV/FLAC, MP3, AAC/M4A and
  automatic language detection were exercised. Corpus WER must remain within
  two percentage points of the FP32 reference; long recordings have their own
  gate and format variants permit at most one additional word error.
- The API contract checks JSON/verbose/text consistency, bearer authentication,
  invalid inputs, upload limits, and rejection rather than truncation above the
  configured duration limit. The RTSP boundary probe is described in
  [SECURITY.md](SECURITY.md).

The canonical Jina reference uses pinned original weights and adapters in
FP32. Its dense block-diagonal vision mask was evaluated as independent SDPA
blocks, preserving every attention boundary. A full one-image comparison to
the original reference was bit-for-bit identical. The checkpoint's missing
multimodal template was restored from its pinned Qwen base model. Reference
caches bind to input hashes/token IDs; serving requests are always rerun.

## Corrections found by actual workloads

The v0.10.2 experiment is historical and does not meet this requirement. Modern
vLLM works with an SM70 wheel and matching CUDA 12.6 packages; its official
modern wheel cannot be made SM70-capable merely by changing a runtime tag.

The native vLLM Whisper decoder lost speech inside long recordings. A wider
silence-search window improved some examples but still failed the held-out
Arabic recording (16.96% WER versus 12.01% for the reference). The dedicated
Whisper decoder passed the same case with the original large-v3 weights.
Conversion with the modern serving environment produced the same model binary
as the independent converter environment.

Jina's full-context request initially took about 87 seconds with FlexAttention.
The scoped Torch SDPA pooling backend reduced it to about 16 seconds while
preserving causal attention, complete inputs, and the numerical/ranking gates.
A PIL preprocessing experiment did not improve parity; the fast processor was
retained after direct pixel equality was established.

An initial warm workload passed 396 checked requests, but a subsequent cold
restart exposed Jina allocator fragmentation during eight-image requests:
14.06 GiB allocated and 1.32 GiB reserved-but-unused could not satisfy another
2 MiB allocation. `PYTORCH_ALLOC_CONF=expandable_segments:True` is therefore
part of the deployment, followed by a new cold-start gate and sustained-load
replay. The first ten-minute replay had no request/quality failures, but Jina
completed 5.625 requests/minute inside the measurement window against a 5.7
gate. The final run extends the same workload to twenty minutes; neither the
offered rate, payload mix nor thresholds were reduced. No text context, image count, resolution or attention window was reduced.

## Capacity and artifact scope

The final allocator configuration passed **700 checked load requests**: 50 cold
burst requests, then 600 steady requests and 50 further burst requests. There
were no request or correctness failures. Offered load was 30 requests/minute;
completed throughput after the two-minute warmup window was **30.06/minute**.
All five services passed the per-service 95% throughput gate.

| service | steady requests | p50 seconds | p95 seconds |
| --- | ---: | ---: | ---: |
| snowflake | 120 | 2.13 | 8.39 |
| bge | 120 | 0.60 | 5.49 |
| gemma | 120 | 0.15 | 1.70 |
| whisper | 120 | 0.81 | 6.53 |
| jina | 120 | 4.86 | 33.84 |

Sampled VRAM peaks were **9,781 MiB** on GPU 0 and
**16,083 MiB** on GPU 1; both reached 100% utilization.
Serving-process PSS peaked at **17.20 GiB** and summed RSS at
**20.53 GiB**. Five-second CPU samples peaked near
4.0 cores. These are sampled footprints, not minimum RAM or
CPU guarantees. Jina needs its dedicated 16 GB card.

The final measurements in [results/measurements.json](results/measurements.json)
identify the tested workload and sampled resource peaks. The steady workload is
30 **total** requests/minute, evenly divided across five services, with varied
real payloads including full contexts. Each service must achieve at least 95%
of its offered rate after the warmup window. This is not 30 full-context calls
per minute per service.

Mixed 5/10/20-request bursts use full-context BGE pairs, embedding batches
containing full contexts, long audio and full-context Jina text. Separate Jina
bursts exercise two/four eight-image requests and mixed image sizes. Image
requests use fresh UUIDs; both processor and encoder identities incorporate
those IDs. The processor cache is disabled, while model/checkpoint and compiler
caches stay warm. These capacity numbers do not rely on repeatedly cached
vision embeddings.

GPU memory includes allocator reserves. Host PSS is sampled across the serving
UID, excluding the sampler; RSS additionally counts shared pages multiple times.
Latency is measured over loopback from client-worker start through response
validation, including HTTP serialization and server queuing. Generator lateness
and client-worker queuing are reported separately. No private production corpus, exact CPU specification or latency
SLA was supplied. Plan host-memory headroom above the measured process footprint.

The shipping image is `local/vllm-v100:0.32.0`, manifest digest
`sha256:0151cd685a671f16083a29c894dfa6a9f0b6eddeef4c9772d3ede172dd22cd44`,
labeled with source commit `c0ad1612cf697c678afd25a141dcf21b7f1770c5`.
A portable Docker archive and checksum are retained under the ignored run
artifacts. The SBOM supplements Trivy's package inventory with the measured
native FFmpeg 9.0.2 component, also identified by Grype's binary catalogue.

The built image is checked as nonroot with a read-only root, no network and no
GPU, including native file hashes and actual audio decoding. Real GPU execution
used the same wheel, packages and configurations in a watched development
container. A separate clean boot of the final image on Vast remains unverified:
no authorized registry publication route was supplied and the rental disallowed
nested containers. Prisma policy approval is also outstanding; public scanner
results and their applicability limits are documented separately.

## Cost and lifecycle

The development rental kept model and compiler caches across model/configuration
failures, under an external label watchdog and a six-hour hard cap. Raw provider
records, endpoints, logs, checkpoints, transcripts and vector arrays stay under
ignored `runs/`; committed measurements contain no provider identifiers.

One earlier modern-development rental was mistakenly destroyed by an agent-added
idle-heartbeat cutoff while local work continued. That cutoff was removed. The
replacement rental retained its checkpoints through the rest of the work. Its
duplicate transfer is an operational overhead, not an acceptable tuning loop.
The final sanitized lifecycle receipt records elapsed time, rates, measured
transfer scope and verified teardown.

Teardown is confirmed by the provider and the external watchdog. The two modern
development sessions totaled approximately **$2.95** in compute/storage; the
replacement session's observed interface traffic adds approximately **$0.12**.
These are estimates, not invoices, and exclude unmeasured first-session traffic
and provider-host image pulls. See [results/lifecycle.json](results/lifecycle.json).
