# V100 auxiliary inference requirements

Status: **bounded hardware smoke passed** (2026-10-20). The
[tested legacy image, configurations, and evidence](v100-16gb/README.md)
cover Snowflake, BGE, Whisper, and EmbeddingGemma **up to 256 tokens** together.
Jina is incompatible with this image. Production-capacity certification remains
outside that small smoke workload.

## Scope

Keep primary inference on Hopper and evaluate embedding, reranking, ASR, and
Jina multimodal embedding services on `volta-2`. Use anonymous profiles under
[HARDWARE_PROFILES.md](HARDWARE_PROFILES.md); record no private host details.
The target is two **16 GB** V100s. Confirm CPU architecture, driver, interconnect,
exact production model revisions, plugin version, request sizes, and concurrency
before choosing final deployment placement. Two V100s are two independent
memory budgets.

Start each model alone with TP1 on `volta-1`, then prove the selected services
remain loaded and serve concurrently on `volta-2`. A possible first placement
is text embedding/reranking on one card and ASR on the other. Jina placement
depends on measured headroom and image resolution; it is not assumed to fit
alongside those services. Sequential successes do not prove offload capacity.

## Image decision

**Do not deploy `openimage/vllm-openai-audio:v0.31.0` on V100.** NVIDIA lists
V100 as compute capability 7.0. The pinned upstream release documents 7.5 as
its minimum; its Dockerfile defaults to CUDA 13.0.3 and omits SM70 from its
build targets. CUDA 13 removed Volta compilation/library support. Adding audio
extras, changing `--dtype`, setting `TORCH_CUDA_ARCH_LIST` at runtime, or
upgrading a driver cannot supply the missing kernels.

Sources: [NVIDIA legacy GPUs](https://developer.nvidia.com/cuda/gpus/legacy),
[v0.31.0 requirements](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/getting_started/installation/gpu.cuda.inc.md),
[v0.31.0 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.31.0/docker/Dockerfile),
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

| candidate | assessment | next gate |
| --- | --- | --- |
| Official `vllm/vllm-openai:v0.10.2`, with matching audio extras for ASR | Hardware-smoke-tested with PyTorch 2.8.0+cu128. Pooling uses V1/FlexAttention; Whisper uses V0/XFormers. See the separate [Dockerfile](v100-16gb/Dockerfile). | Production context/concurrency and service-quality targets; Gemma is capped at 256 tokens and Jina is excluded. |
| A newer vLLM source build with CUDA 12.6 and matching SM70 PyTorch binaries | Research option if newer model/plugin APIs are necessary. The v0.31.0 CMake branch for CUDA below 12.8 still lists SM70; that alone proves neither a successful build nor working runtime backends. | Reproducible build, dependency ABI checks, and complete SM70 kernel/model proof. |
| Existing Jina plugin image | Its documented parity is on A10G with vLLM 0.22.0, not V100. | Confirm exact deployed plugin/checkpoint, then audit its complete stack before hardware testing. |

The legacy candidate's source is
[`01efc7ef781391e744ed08c3292817a773d654e6`](https://github.com/vllm-project/vllm/tree/01efc7ef781391e744ed08c3292817a773d654e6).
Registry inspection on 2026-10-20 resolved its Linux/amd64 manifest to
`vllm/vllm-openai@sha256:df2607b26bdda2875de4832f4d08da0055b4b6e3570347f3a849bcc652771dd6`.
Manifest availability is not execution evidence. Confirm host architecture
before using this platform-specific digest.

The tested legacy stack uses `VLLM_USE_V1=1` and
`VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` for pooling. Whisper uses
`VLLM_USE_V1=0` and `VLLM_ATTENTION_BACKEND=XFORMERS`. All services use TP1,
eager execution, conservative sequence limits, and an explicit dtype.
The initial V0 pooling experiment failed: its runner expects `compute_logits`.
These are model-specific settings, not universal V100 flags. In particular,
V1-only plugin APIs do not become V0 compatible through an environment variable.
See the [v0.10.2 backend selector](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/platforms/cuda.py)
and [model registry](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/registry.py).

An ASR derivative must install `vllm[audio]==0.10.2` without replacing the
candidate's vLLM, torch, or CUDA packages; record the resulting package lock
and image digest. Do not repoint `Dockerfile.audio` or apply the current
release's patches to this older image. A source-built image would be a new
exception to this fork's prebuilt-image policy, with its own maintenance and
retirement decision; it is not part of the existing release lane.

For a newer source-build experiment, verify the exact PyTorch wheel's SM70
coverage. CUDA toolkit support and wheel architecture coverage are separate:
PyTorch 2.11 dropped Volta from its CUDA 12.8 binaries while retaining it in
CUDA 12.6 binaries.
See [PyTorch's release notice](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290).

## Payload requirements

“Snowflake Embedding Gemma” does not identify a single checkpoint. Treat
Snowflake Arctic Embed and Google EmbeddingGemma as separate possibilities
until exact IDs are confirmed. BGE reranker variants also have different
architectures and memory requirements.

| workload / illustrative ID | initial precision and compatibility constraint | required correctness evidence |
| --- | --- | --- |
| Snowflake, e.g. `Snowflake/snowflake-arctic-embed-l-v2.0` | FP16 candidate; this example is XLM-RoBERTa. Do not substitute another Arctic variant silently. | Exact prompts, pooling, dimensions, normalization, and retrieval ranking against the existing service/reference. |
| `google/embeddinggemma-300m` | **FP32 on V100; 256-token cap in the tested image.** Full context needs unsupported sliding-window attention. Google excludes FP16 activations; V100 lacks native BF16 support. | Finite embeddings, correct query/document prefixes, requested dimensions, and retrieval parity; 256-token parity and 257-token rejection passed. |
| BGE, e.g. `BAAI/bge-reranker-v2-m3` | FP16 candidate, XLM-RoBERTa sequence classification. Gemma-based BGE variants require a separate assessment. | Finite scores, correct document/index mapping, relevant-vs-irrelevant ordering, reference score/rank comparison. |
| `openai/whisper-large-v3` | FP16 candidate with audio extras; test the exact model, not `large-v3-turbo` or a small-model substitute. | `/v1/audio/transcriptions`, known transcripts, intended languages, short and production-length audio, and real-time factor. |
| Jina v4 with custom plugin | Pin plugin, checkpoint, projector, and processor settings. FP16 backbone is only a numerical experiment; preserve FP32 projection when required. | Real text **and image** requests, finite normalized per-token vectors, projector parity, late-interaction ranking, and production image resolution. |

Sources: [Snowflake model](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0),
[EmbeddingGemma precision requirements](https://huggingface.co/google/embeddinggemma-300m),
[BGE model](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[Whisper model](https://huggingface.co/openai/whisper-large-v3).

The public
[`Mazyod/jina-embeddings-v4-vllm-plugin`](https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin/tree/f972e376d11acb918cebfc8a7e594f7869763f9b)
was inspected as a possible match, not assumed to be the deployed version.
At that commit its package is 0.1.1; its
[compatibility record](https://github.com/Mazyod/jina-embeddings-v4-vllm-plugin/blob/f972e376d11acb918cebfc8a7e594f7869763f9b/research/docs/COMPAT.md)
certifies plugin 0.1.0 / vLLM 0.22.0 / A10G. Its model imports
`pooler_for_token_embed` from `pooler.tokwise` and `SupportsLateInteraction`;
v0.10.2 has neither that module path nor that interface. Installing the current
plugin into the legacy candidate is therefore not a working solution.
The plugin's served contract is `/pooling` with `[n,128]` multivectors, not
an ordinary single-vector `/v1/embeddings` response.

## Validation gates

1. **Static / free:** confirm the missing hardware and model details; pin
   image, model, plugin, and projector identities; inspect native architecture
   targets and plugin imports; run `bash fork/bench/preflight.sh`.
   The existing Hopper release gate does not certify these payloads.
2. **Runtime before downloads:** on the selected V100, record sanitized
   memory/topology, driver, CUDA runtime, torch/vLLM versions, and
   `torch.cuda.get_arch_list()`. Require SM70 coverage, execute FP16 and FP32
   CUDA operations, then actual vLLM and selected attention kernels. Import
   success or `torch.cuda.is_available()` alone is insufficient.
3. **One service at a time:** validate its exact engine arguments with the
   installed release; record the chosen backend and dtype from boot logs.
   Stage each pinned checkpoint once. Run the table's real API requests and
   compare outputs with a trusted reference at identical preprocessing.
   Set numerical tolerances and task-quality thresholds before measurement;
   an HTTP 200 or finite vector alone does not establish correctness.
4. **Two-card proof:** load all services selected for offload together with
   explicit per-process memory limits. Exercise production text lengths,
   rerank batch sizes, audio duration, and image pixel bounds concurrently.
   Record peak VRAM, errors, cold/warm latency, p50/p95 latency, throughput,
   and ASR real-time factor against agreed service targets. Do not use the
   default 90% memory budget for several processes sharing one GPU.
5. **Certification:** commit exact engine YAMLs and placement only after the
   configuration works; rerun those bytes and publish a sanitized evidence
   record with per-model pass/fail, reference comparison, and image digest.
   Keep failures distinct: unsupported kernel/API, non-finite output, OOM,
   quality regression, and unacceptable latency require different remedies.

Use an explicitly selected existing test host when available. Any rental is
development until these configurations pass: arm the external label watchdog
**before create**, record the price/time/transfer envelope, and preserve the
warm model cache across model failures, as required by
[RUNBOOK.md](../bench/RUNBOOK.md). A newer GPU can check API wiring but cannot
certify V100 execution.

## Assessment boundary

Completed: real SM70 kernel execution on two 16 GB V100s, concurrent real API
requests for four services, independent FP32 references, Gemma's 256-token
boundary and 257-token rejection, and the Jina plugin's runtime import failure.
A separate legacy audio image was built locally and its package versions
matched the tested runtime. One watched Vast development rental reused its
checkpoints through all configuration attempts.

Outstanding: exact production payload identities and service targets, broader
quality/context/concurrency tests, full-context Gemma, and a Jina-compatible
SM70 stack. The locally built derivative was not published or rebooted as a
separate rental; execution used the same base and package additions in place.
See the [hardware record](v100-16gb/RECORD.md) for evidence, cost, and teardown.
