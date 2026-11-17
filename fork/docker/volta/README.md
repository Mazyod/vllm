# Frozen CUDA 12.6 / SM70 audio image

Candidate for a single `openimage/vllm-audio:0.32.0-cu126-sm70` release.
Publication requires a new hardware gate; the archived 2026-10-23 results do
not certify a rebuilt digest. No `latest` tag or periodic rebuild is intended.

This builds pristine vLLM v0.32.0 native extensions for SM70, retaining the
previously verified PyTorch 2.13.0+cu126, TorchAudio 2.11.0+cu126, Triton 3.7.1,
and CUDA 12.6 assembler. Installing an older CUDA runtime around a modern
CUDA 13 wheel cannot restore its absent SM70 kernels.

The recipe omits the old campaign's Jina plugin, attention plugin, CTranslate2,
ASR adapter, model weights, and service supervisor. Whisper uses native vLLM.
Packaging edits only relax the runtime setuptools ceiling and omit unused
video-I/O packages; the build exports that diff and the wheel hash into the
image. No model implementation or audio chunking code is patched.

```bash
docker build -f fork/docker/volta/Dockerfile \
  --build-arg MAX_JOBS=2 --build-arg OVERLAY_REVISION="$(git rev-parse HEAD)" \
  -t local/vllm-audio:0.32.0-cu126-sm70 fork/docker/volta
docker run --rm -i -e VLLM_TARGET_DEVICE=cpu --entrypoint python \
  local/vllm-audio:0.32.0-cu126-sm70 < fork/docker/volta/check_audio.py
```

`requirements-runtime.lock` retains the previous hash-locked runtime, excluding
the unused Jina model plugin. Builder dependencies are separately pinned. The
final published digest is the immutable artifact; distro package updates mean a
later rebuild must be treated as a new candidate, never silently substituted.

[Engine files](../../deploy/volta/) retain full 8192-token pooling/reranking
contexts and Whisper's 448-token decoder window. The Whisper profile pins the
same `mlmlaziz/bbyn-v2` Whisper large-v3 fine-tune as the historical native-vLLM
campaign. Its verified API contract uses explicit language and JSON; timestamp
formats require separate qualification. The frozen runtime needs V1 and the
CUDA 12.6 `ptxas`, set inside the image. Do not copy that environment into the
recurring v0.34.0 image.

The final gate must execute native kernels on actual Tesla V100 **16 GB** GPUs,
compare full-input Snowflake/BGE results against independent FP32 references,
exercise Whisper with audio decoding and TorchAudio, and measure concurrent
co-residency after a cold restart. Preserve actual token counts and memory
peaks. A successful import or a larger-memory GPU is not capacity proof.
