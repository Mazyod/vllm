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
| [Whisper, two GPUs](whisper-dp2.yaml) | one service/API with two native DP workers; explicit language and JSON | 448 decoder tokens; long audio uses native chunking |
| [Whisper, one GPU](whisper.yaml) | native `mlmlaziz/bbyn-v2`; explicit language and JSON | 448 decoder tokens |
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

## One Whisper service on two GPUs

Use [whisper-dp2.yaml](whisper-dp2.yaml) for a single Docker service and API
endpoint. Set the deployment's `REPLICAS=1` and expose both GPUs to that
container. vLLM's `data-parallel-size: 2` starts one full Whisper worker per GPU
and handles request distribution internally.

| service | deployment replicas | visible GPU(s) | engine config |
| --- | ---: | --- | --- |
| Whisper | 1 | 0, 1 | `whisper-dp2.yaml` |
| BGE | 1 | 0 | `bge.yaml` |
| Snowflake | 1 | 1 | `snowflake.yaml` |

This profile uses eight sequences per rank, up to sixteen across both GPUs.
The [paired native-DP test](results/20261118-whisper-dp2-batching.md) measured
49% more short-clip throughput at concurrency 16 and 81% more long-upload
throughput at concurrency 4 than four sequences per rank. Both ranks processed
requests with BGE and Snowflake active, with no cache preemptions.

The optional [whisper-throughput.yaml](whisper-throughput.yaml) profile uses
eight sequences with independent one-GPU workers and external request
balancing. Its [earlier measurements](results/20261118-whisper-scaling.md)
remain available; the native DP2 profile provides one service and endpoint.

Fetch the exact configuration snapshot into a new directory:

```bash
(
  set -eu
  config_base="https://raw.githubusercontent.com/Mazyod/vllm/2b32ed0e0f3f03d392e540abec5bb3e92acb97d0"
  mkdir -p vllm-configs-dp2
  for model in whisper-dp2 bge snowflake; do
    curl -fL --retry 3 "$config_base/fork/deploy/volta/$model.yaml" \
      -o "vllm-configs-dp2/$model.yaml"
  done
)
```

The DP2 profile's SHA-256 is
`0b1afedb639a5300189e27f823b7550c51ab28120fec18fb57dae74fc0a2e5a6`.
Mount the selected YAML into the container and pass `--config /path/to/model.yaml`
to the image's existing `vllm serve` entrypoint. Ports, replica counts, GPU
bindings, cache mounts and request routing belong in the deployment definition.
