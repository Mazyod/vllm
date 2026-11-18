# Frozen V100 audio runtime

Use `openimage/vllm-audio:0.32.0-cu126-sm70` (also `v0.32.0-cu126-sm70`),
qualified by the [hardware gate](results/20261117-gate.md). The immutable digest is
`sha256:ddbb02258092f1888e19654e0757744e70d52be2c36c01584025a5ab5106dc5e`.
This is a one-time release with no `latest` promotion or scheduled rebuilds.

The [source recipe](../../docker/volta/) compiles vLLM 0.32.0 native extensions
for SM70 with CUDA 12.6. It includes the tested TorchAudio/audio stack and
preserves the necessary legacy model-runner and CUDA assembler environment.
Do not copy that environment into vLLM 0.34.0.

| configuration | API contract | full input/decoder window |
| --- | --- | --- |
| [Whisper](whisper.yaml) | native `mlmlaziz/bbyn-v2`; explicit language and JSON | 448 decoder tokens; long audio uses native chunking |
| [BGE reranker](bge.yaml) | `/rerank` | 8192 tokens |
| [Snowflake](snowflake.yaml) | `/v1/embeddings`, 1024 dimensions | 8192 tokens |

Stage the exact pinned checkpoints in a mounted Hugging Face cache. Supply a
configuration with `vllm serve --config /path/to/model.yaml`; the image contains
no model weights or supervisor. Start co-resident services serially, then test
them together. The gate records its tested GPU placement and memory peaks.

Whisper's gate compares native FP16 inference with FP32 using the same pinned
fine-tune, audio, and chunk boundaries. Matching that reference does not
establish domain accuracy: Arabic read-speech WER was approximately 50% in both
paths. Timestamp formats and automatic language detection are not qualified.

## Higher Whisper throughput on two GPUs

The [scaling comparison](results/20261118-whisper-scaling.md) recommends two
independent Whisper workers, one per GPU, using
[whisper-throughput.yaml](whisper-throughput.yaml). Run BGE on GPU 0 and
Snowflake on GPU 1. Each Whisper worker allows eight sequences; the original
four-sequence configuration remains available unchanged.

Use one container/service instance per explicitly bound GPU and balance the
two Whisper endpoints through your router. Raising a deployment's replica
count does not itself assign distinct GPUs. Native vLLM DP2 also works, but
the tested two-worker configuration had the best measured aggregate throughput.

Fetch the exact configuration snapshot into a new directory:

```bash
(
  set -eu
  config_base="https://raw.githubusercontent.com/Mazyod/vllm/5d668ca5855b9ec430530c2db75475f2d7b0b229"
  mkdir -p vllm-configs-20261118
  for model in whisper whisper-throughput bge snowflake; do
    curl -fL --retry 3 "$config_base/fork/deploy/volta/$model.yaml" \
      -o "vllm-configs-20261118/$model.yaml"
  done
)
```

The throughput profile's SHA-256 is
`27ab20dd25dc50d91800a54161cd48b83e371250d9563b7644f388209ee4d242`.
Mount the selected YAML into the container and pass `--config /path/to/model.yaml`
to the image's existing `vllm serve` entrypoint. Ports, replica counts, GPU
bindings, cache mounts and request routing belong in the deployment definition.
