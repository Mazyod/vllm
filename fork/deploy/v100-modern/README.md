# vLLM 0.32 on two 16 GB V100s

This is the separate SM70 build lane. It does not change the Hopper image or
release pointer. The earlier v0.10.2 experiment is historical evidence, **not a
production recommendation**: its shortened Gemma context and incompatible Jina
APIs do not meet these requirements. See the [validation record](RECORD.md)
and [security review](SECURITY.md) before promotion.

## Stack and maintenance boundary

Build only the vLLM native wheel from upstream v0.32.0 commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. Use published CUDA 12.6 PyTorch
binaries and NVIDIA's packaged 12.6 assembler; do not compile drivers, PyTorch,
or CUDA. Modern official vLLM wheels lack the SM70 kernels, so installing the
Python package with an older CUDA runtime alone cannot work.

The runtime and ASR locks pin all external Python dependencies with hashes. Platform wheels are
PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, torchaudio 2.11.0+cu126,
TorchCodec 0.17.0+cpu, and Triton 3.7.1. CPU TorchCodec avoids its CUDA 13 NVRTC
dependency; model inference remains on GPU. `TRITON_PTXAS_PATH` selects CUDA
12.6's assembler because Triton's bundled CUDA 13 assembler cannot target SM70.

The wheel build makes two explicit **runtime dependency metadata** changes:
setuptools is permitted at the patched 83.0.0 release (CVE-2026-59890), and the
unused `PyNvVideoCodec==2.0.4` and `opencv-python-headless` video-I/O dependencies
are omitted. Both bundled FFmpeg 8.1.1; none of these services uses them. Audio decoding and
all image inputs remain supported. Engine/model source is unchanged. The
builder retains upstream's build-time setuptools constraint, and the exact
metadata diff accompanies the wheel. Retire the overrides when upstream allows
the patched setuptools and makes the video-I/O dependencies optional.
Retire this separate build lane if upstream ships a validated SM70 wheel again,
or when Volta offload is retired. Updating any pinned version requires repeating
the native-kernel, full-payload, quality, load, and security checks.

## Full payload contracts

| service | precision | full text context | placement | API |
| --- | --- | ---: | --- | --- |
| Snowflake Arctic Embed L v2.0 | FP16 | 8192 | GPU 0 | `/v1/embeddings`, 1024 dimensions |
| BGE reranker v2 m3 | FP16 | 8192 combined pair tokens | GPU 0 | `/rerank` |
| EmbeddingGemma 300M | FP32 | 2048 | GPU 0 | `/v1/embeddings`, 768 dimensions |
| Whisper large v3 | FP16 | 448 decoder tokens per chunk | GPU 0 | `/v1/audio/transcriptions` |
| Jina v4 retrieval multivector | FP16 backbone, FP32 projection | 32768 | GPU 1 | `/pooling`, `[tokens,128]` |

EmbeddingGemma keeps its native sliding attention and full context. FP32 avoids
unsupported BF16 and the model's unsuitable FP16 activations.

Whisper uses **faster-whisper 1.2.1 / CTranslate2 4.8.3** in the same image.
The native vLLM decoder lost speech on held-out long recordings; changing split
boundaries did not fix that reliably. The dedicated decoder passed those same
recordings. It uses the original pinned large-v3 weights converted to FP16,
beam size 5, full 30-second model windows and timestamp-based long-form seeking,
without VAD or previous-text conditioning. No recording tail is discarded.
Retire this API adapter when vLLM's native Whisper implementation passes the
same long-recording quality gate.

The transcription API supports non-streaming `json`, `verbose_json`, and `text`,
with segment timestamps. It retains vLLM's upload/decode limits: by default
25 MiB compressed, 600 seconds, and 256 MiB decoded. Oversized inputs are
**rejected, never truncated**. Configure the corresponding
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `VLLM_MAX_AUDIO_DECODE_DURATION_S`, and
`VLLM_MAX_AUDIO_DECODE_BYTES` environment variables if a larger admission limit
is needed, then validate that workload. Native decoder context remains 448.

Jina uses plugin 0.1.1 and the pinned **retrieval adapter baked into the
multivector checkpoint**, including its projector. It is not the single-vector
checkpoint. Full-resolution image processing keeps `max_pixels=3211264`; the
configuration accepts up to eight images within the total context budget.
The bundled `vllm-volta-runtime` plugin uses Torch's efficient SDPA kernel for
Jina text attention through vLLM's `CUSTOM` backend. It preserves the model's
causal mask and request boundaries; no context or attention feature is removed.
It rejects unsupported windows, custom masks, quantized outputs and cached
partial prefills. Gemma continues to use FlexAttention with native sliding
attention; Jina vision uses Torch SDPA. The four vLLM services use TP1 and eager
execution. ASR has one GPU worker to bound memory. The two devices remain
separate memory pools and need no NVLink.
Model repositories and immutable revisions are in `models.json`.

Keep **`PYTORCH_ALLOC_CONF=expandable_segments:True`**. A cold restart followed
by long text and eight-image requests exposed allocator fragmentation despite
1.32 GiB of reserved, unused memory; the full image geometry remains unchanged.
The processor still normalizes images, on CPU, with the reference's fast image
backend. Its cache is disabled to bound host memory for distinct document jobs.
Benchmarks supply fresh image UUIDs so both preprocessing and vision encoding
are exercised instead of receiving a capacity boost from repeated images.

## Build

The CUDA builder is Linux/amd64. A CPU build host can compile the wheel; GPU
hardware is required for validation. Allow adequate disk and RAM. Never put
account tokens or model weights into the image build context.

```bash
mkdir -p "$PWD/wheelhouse"
docker run --rm \
  -v "$PWD/fork/deploy/v100-modern:/recipe:ro" \
  -v "$PWD/wheelhouse:/workspace/build-artifacts" \
  nvidia/cuda@sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade \
  bash /recipe/build-wheel.sh

docker build \
  --build-context wheelhouse=./wheelhouse \
  --build-arg OVERLAY_REVISION="$(git rev-parse HEAD)" \
  -t local/vllm-v100:0.32.0 \
  -f fork/deploy/v100-modern/Dockerfile fork/deploy/v100-modern
```

The runtime is a patched Ubuntu 24.04 base, Python 3.12, nonroot UID 10001.
The small local plugin and transcription API are installed from the reviewed
source tree. The final stage omits the distro FFmpeg packages; TorchCodec uses
PyAV 19.0.1's bundled FFmpeg 9.0.2 through ordinary SONAME aliases. Both decoders
share those pinned libraries. The dependency builder's older FFmpeg installation
is not copied into the runtime image.
Compiler headers and GCC remain because TorchInductor compiles kernels at
runtime. The native wheel's checksum is verified separately from the hashed
Python dependency lock. OS packages are refreshed at build time; the final
image digest and SBOM, rather than the base tag alone, identify the deployed
artifact. Rebuilt native wheels may have different binary hashes; record and
validate the exact artifact that will ship.

## Stage and deploy

Stage on a trusted host using its existing Hugging Face authentication for the
gated Gemma repository. Do not send an account-wide HF token to a rental.

```bash
uv run --no-project --with huggingface-hub -- \
  python fork/deploy/v100-modern/stage-models.py \
  --root /srv/v100-models --names snowflake bge gemma whisper jina

export MODEL_DIR=/srv/v100-models
export V100_IMAGE=local/vllm-v100:0.32.0
# One-time CPU conversion, using the same pinned image; no network or GPU needed.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv -v "$MODEL_DIR:/models" \
  --entrypoint /usr/local/bin/uv "$V100_IMAGE" \
  run --no-project --python /opt/vllm/bin/python ct2-transformers-converter \
  --model /models/whisper --output_dir /models/whisper-ct2 \
  --copy_files tokenizer.json preprocessor_config.json --quantization float16

# Pin V100_IMAGE to your published registry digest for production rollout.
# Set VLLM_API_KEY through your secret manager or protected environment.
docker compose -f fork/deploy/v100-modern/compose.yaml up -d
```

Use a patched NVIDIA R580 driver and a compatible Container Toolkit; the
measured driver is 580.173.02. NVIDIA lists R580 as the last driver branch for
Volta in its [architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html). Ensure staged files are
readable by UID 10001. Compose mounts models read-only, drops capabilities,
uses a read-only root filesystem, and keeps writable compiler caches in a
volume. API keys are required. Ports 8001–8005 bind to localhost by default;
place the authenticated services behind your existing TLS proxy for remote
clients. `BIND_ADDRESS` explicitly changes the bind address.

Health checks warm full-length shapes before declaring readiness. Cold kernel
compilation can take minutes; preserve the cache volume across restarts. Keep
cache contents separate between incompatible stack versions. Avoid routing
traffic solely on `/health` before the container warmup finishes.

Jina request contract:

```json
{"model":"jina","task":"token_embed","input":["Query: account safeguards","Passage: Account access requires approval."]}
```

Send this to `/pooling` with bearer authentication. Each response item contains
one 128-dimensional vector per token. Rank documents with late interaction:
sum, over query tokens, the maximum query/document token dot product. Do not
average these vectors into a single embedding. For images use `messages` with
`image_url` content, `task=token_embed`, and `add_generation_prompt=false`.
For a single image, append `{"type":"text","text":"Describe the image."}`
to the content array to match the canonical single-image processor. For a
multi-image document, use the image parts alone, matching its nested-image
processor. The reference template revision is recorded in `models.json`.

## Reproduce validation

`prepare-fixtures.py` creates real document and English/Arabic speech fixtures
from the pinned public NIST/FLEURS inputs. `media_variants.py` adds MP3, AAC and
automatic language cases. Two additional >3-minute recordings use held-out
utterances, for 74 speech cases in total. Fixture manifests include hashes and
provenance; large datasets and model outputs stay outside Git.

`text_quality.py` compares full contexts, mixed batches and 32-document reranks
against FP32 Transformers. `asr_quality.py` compares transcripts and corpus WER,
including recordings longer than three minutes, against independent FP32
Whisper. `jina_quality.py` checks multivector shape, finiteness, normalization,
FP32 numerical agreement, text/image batching and MaxSim ranking. Its canonical
reference requires a separate Transformers 4.57.6 / peft 0.17.1 environment;
serving uses the modern locked environment. The original reference's forced
BF16 `encode_*` wrappers are bypassed by its own FP32 forward/projector.
Its block-diagonal vision SDPA is evaluated as independent blocks to avoid
quadratic masked-out work across images. A direct original-forward check runs
before that optimization; the complete one-image FP32 output also matched
bit-for-bit. The canonical checkpoint omits a usable multimodal template;
`stage-models.py --include-jina-reference` restores the pinned Qwen base-model
template expected by its multi-image processor. Text token IDs are checked
against the serving tokenizer as well as the reference.
Reference caches are bound to request/audio hashes and token IDs; every serving
request is rerun even when an independent reference is reused.

After those quality suites pass, `load_test.py` replays their payloads and
checks correctness during the final twenty-minute run at 30 total requests/minute, followed by
simultaneous 5/10/20-request full-payload bursts and dedicated concurrent
high-resolution image/mixed Jina bursts. `asr_contract.py` checks response
formats, authentication and rejection boundaries. `telemetry.py` samples
serving-process PSS/RSS and CPU; collect one-second NVIDIA telemetry alongside it. This interprets “tens per
minute” as total traffic, evenly split across services; it is not a claim of
30 full-context requests/minute **per service**. Record GPU memory/utilization
throughout. No deployment-specific latency SLA or private production corpus
was supplied, so retain the measured latency distribution and repeat with the
real traffic mix before increasing admission limits.

Run an actual image scan and SBOM export, then submit the **same image digest**
to Prisma Cloud under the deployment's policy. A clean public advisory scan is
not a Prisma policy verdict. Do not substitute import tests or an HTTP 200 for
the quality and capacity checks above.
