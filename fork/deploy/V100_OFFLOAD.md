# V100 auxiliary inference requirements

The deployment target is **two 16 GB Tesla V100s**, anonymous profile `volta-2`.
Primary inference remains on Hopper. Offload Snowflake embeddings,
EmbeddingGemma, BGE reranking, Whisper large v3, and **Jina v4 multivector**.
The implementation and reproducible build are in
[`v100-modern/README.md`](v100-modern/README.md).

The earlier [v0.10.2 experiment](v100-16gb/RECORD.md) is historical only. Its
256-token Gemma workaround and incompatible Jina APIs are not acceptable
production configurations. Full advertised contexts, native attention
semantics, image resolution, and complete recordings are required.

## Image requirements

Use modern vLLM, current patched dependencies, an auditable lock, and an image
SBOM/security report suitable for submission to Prisma Cloud. Do not assume a
new CUDA driver makes an incompatible binary work. V100 is SM70; modern
upstream vLLM images omit that target, and CUDA 13 cannot compile it. Simply
installing a modern PyPI wheel alongside CUDA 11/12 cannot add missing kernels.

The selected lane builds vLLM **v0.32.0** with CUDA **12.6**, using published
SM70-capable PyTorch binaries. It does not compile the driver or PyTorch.
Preserve the CUDA 12.6 `ptxas` selection for Triton runtime compilation. This
exception is isolated from the Hopper release pointer and image workflow.
Whisper uses CTranslate2/faster-whisper in that same image: its long-form
algorithm passed recordings that native vLLM transcribed incompletely. The
OpenAI transcription route, original large-v3 checkpoint and full recordings
are preserved.

Sources: [NVIDIA legacy GPU capabilities](https://developer.nvidia.com/cuda/gpus/legacy),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
[vLLM v0.32 CMake targets](https://github.com/vllm-project/vllm/blob/v0.32.0/CMakeLists.txt),
[PyTorch's Volta binary policy](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290).

## Payload and acceptance contracts

Pinned checkpoints and revisions are in
[`models.json`](v100-modern/models.json). “Snowflake Embedding Gemma” is treated
as two distinct models, Snowflake Arctic Embed L v2.0 and Google EmbeddingGemma
300M. Different production IDs require separate validation.

| workload | required full payload | correctness evidence |
| --- | --- | --- |
| Snowflake Arctic Embed L v2.0 | 8192 tokens, mixed-length batches | FP32 reference cosine, dimension, normalization, token count and input/output mapping |
| EmbeddingGemma 300M | 2048 tokens, native sliding attention, FP32 on V100 | FP32 reference parity, prefixes, mixed batches, finite normalized 768-dimensional output |
| BGE reranker v2 m3 | 8192 combined query/document tokens, 32 candidates | Reference scores and ordering, document/index mapping, finite output |
| Whisper large v3 | English/Arabic, short and >3-minute audio, FLAC/MP3/AAC, automatic language | Ground-truth and independent FP32 WER, timestamps covering the complete recording, real-time factor |
| Jina v4 retrieval multivector | 32768 text tokens, full-resolution images and multiple images | `[tokens,128]`, normalization, FP32 projector/reference parity, batching and MaxSim retrieval ranking |

Jina's contract is **`/pooling` with `task=token_embed`**, not ordinary pooled
`/v1/embeddings`. Pin the plugin, baked retrieval adapter, projector and image
processor together. Test the model's full advertised 32768-token context;
the backbone's larger configuration field does not expand that advertised
contract. Preserve `max_pixels=3211264` for this deployment's baked checkpoint,
and validate one, two and eight images within the total context budget.

Model sources: [Snowflake](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0),
[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m),
[BGE](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[Whisper](https://huggingface.co/openai/whisper-large-v3),
[Jina canonical model](https://huggingface.co/jinaai/jina-embeddings-v4),
[Jina multivector checkpoint](https://huggingface.co/Mazyod/jina-embeddings-v4-vllm-mv),
[Jina plugin](https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin).

## Combined deployment gate

All five services must remain resident on the two cards throughout the mixed
workload. Use independent TP1 services and explicit memory/sequence limits;
two V100s do not provide a single 32 GB memory pool. Final placement follows
measured full-payload memory usage, not model-weight size alone.

Interpret the supplied “tens per minute” as **30 total requests/minute**,
evenly divided across five services, plus simultaneous 5/10/20-request bursts.
The quality suites' real full-context/media payloads must appear in this load.
Report per-service p50/p95/max latency, errors, achieved throughput, queueing,
ASR WER/real-time factor, and sampled peak VRAM/utilization. Numerical checks
must continue under load. No exact production latency SLA or traffic mix was
provided; retain that limitation explicitly rather than inventing one.

Test the exact image's packages/native artifacts, nonroot operation, readiness
warmup and authentication. Record which checks ran in the built image versus
the development environment. A successful public vulnerability scan is not a
Prisma policy approval; the deployable digest must be scanned under that
installation's policy before promotion.

Hardware testing follows [HARDWARE_PROFILES.md](HARDWARE_PROFILES.md) and
[RUNBOOK.md](../bench/RUNBOOK.md): cheapest matching real V100 venue, external
watchdog before paid creation, one warm development rental across failures,
no repeated checkpoint downloads as a tuning strategy, anonymous committed
records, measured transfer/time/rate/cache reuse, and verified final teardown.
Run `bash fork/bench/preflight.sh` and the alignment gate before submission.
