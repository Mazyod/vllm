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
