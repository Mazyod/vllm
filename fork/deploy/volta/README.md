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

This is the exact native-DP configuration that passed the
[two-V100 comparison](results/20261118-whisper-scaling.md), including inference
on both ranks with BGE and Snowflake active. It retains four sequences per
rank; eight sequences with native DP2 were not tested.

The optional [whisper-throughput.yaml](whisper-throughput.yaml) profile uses
eight sequences with independent one-GPU workers and had the highest measured
aggregate throughput. It requires two externally balanced workers. The DP2
profile is the simpler deployment choice.

Fetch the exact configuration snapshot into a new directory:

```bash
(
  set -eu
  config_base="https://raw.githubusercontent.com/Mazyod/vllm/df6504d193b02106b85c7e18f897f6ce6c130a93"
  mkdir -p vllm-configs-dp2
  for model in whisper-dp2 bge snowflake; do
    curl -fL --retry 3 "$config_base/fork/deploy/volta/$model.yaml" \
      -o "vllm-configs-dp2/$model.yaml"
  done
)
```

The DP2 profile's SHA-256 is
`910cce3f28d9aa3d37eb97928f39a958a0415b2bc73e0a1bed66ca00d04e2240`.
Mount the selected YAML into the container and pass `--config /path/to/model.yaml`
to the image's existing `vllm serve` entrypoint. Ports, replica counts, GPU
bindings, cache mounts and request routing belong in the deployment definition.
