# Legacy image for two 16 GB V100s

This is a separate deployment image based on **vLLM 0.10.2**, not a downgrade of
this fork's Hopper release. It adds only the exact audio dependencies tested on
that base. No upstream source patch, CUDA rebuild, or Jina plugin is included.
The [hardware record](RECORD.md) defines the tested scope and limitations.
Retire this legacy lane when the V100 profile is retired or a replacement
serving stack passes the same hardware and payload checks.

```bash
docker build -t local/vllm-v100-audio:0.10.2 fork/deploy/v100-16gb
```

The base is pinned to its Linux/amd64 manifest. Both vLLM and PyTorch contain
SM70 kernels; the real V100 checks execute FP16/FP32 matrix multiplication,
vLLM RMSNorm, and XFormers attention. The manually extended rental and the
locally built derivative have identical Python package names and versions.
The derivative itself has not been booted as a new Vast instance or published.

## Placement and engine settings

| service | configuration | GPU | engine / attention | precision | maximum input |
| --- | --- | --- | --- | --- | --- |
| Snowflake Arctic Embed L v2 | `snowflake.yaml` | 0 | `VLLM_USE_V1=1`, `FLEX_ATTENTION` | FP16 | 2,048 tokens |
| BGE reranker v2-m3 | `bge.yaml` | 0 | `VLLM_USE_V1=1`, `FLEX_ATTENTION` | FP16 | 2,048 tokens per query/document pair |
| EmbeddingGemma 300M | `gemma-short.yaml` | 0 | `VLLM_USE_V1=1`, `FLEX_ATTENTION` | FP32 | **256 tokens, including special tokens and prefixes** |
| Whisper large-v3 | `whisper.yaml` | 1 | `VLLM_USE_V1=0`, `XFORMERS` | FP16 | 448-token decoder window; only the short audio fixture was tested |

Set `VLLM_ATTENTION_BACKEND` to the table's attention value. Do not force V0
for pooling: its model runner fails with a missing `compute_logits` method.
Do not copy Hopper BF16/FP8 settings to these services.

EmbeddingGemma requires `runner: pooling`, `convert: embed`, and explicit
`is_causal: false` for this old vLLM implementation. FlexAttention in this
release cannot implement its sliding window. The diagnostic 512-token cap
failed reference parity: Transformers 4.57.6 interprets the bidirectional
window as radius 256 (exclusive bound 257). The **256-token** configuration
preserves that mask without sliding, and rejects larger inputs. It does not
provide the model's normal 2,048-token service. Keep the query/document prefixes
in `probe.py`; the cap includes those prefixes.

**Jina v4 is not supported by this image.** Plugin 0.1.1 fails to import because
`vllm.model_executor.layers.pooler.tokwise` does not exist in v0.10.2. Its newer
vLLM requirement needs a separate SM70 build/port and fresh text/image parity.
Do not install it into this working legacy stack.

## Reproduce inside the image

Stage the exact revisions from the YAML files into `HF_HOME=/workspace/hf`.
For example, the image's Hugging Face client can download only inference files:

```bash
export HF_HOME=/workspace/hf
hf download Snowflake/snowflake-arctic-embed-l-v2.0 \
  --revision ac6544c8a46e00af67e330e85a9028c66b8cfd9a \
  --include '*.json' '*.safetensors' '*.model' '*.txt'
```

Repeat for BGE and Whisper with their committed revisions. Stage
`google/embeddinggemma-300m` revision
`57c266a740f537b4dc058e1b0cda161fd15afa75` at `/workspace/models/gemma`, including
its tokenizer, `modules.json`, pooling configuration, and both Dense layers.
Use an authorized local download for the gated model; this campaign transferred
its large weight file using a file-scoped signed URL, without giving the rental
an account-wide Hugging Face token.

Place OpenAI Whisper's public [JFK fixture](https://github.com/openai/whisper/blob/main/tests/jfk.flac)
at `/workspace/jfk.flac` and verify SHA-256
`63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
Copy this directory to `/workspace/v100`. Prepare an isolated **reference-only**
environment; the serving image's Transformers 4.56.1 ignores EmbeddingGemma's
bidirectional setting and is not a valid reference for that model:

```bash
uv venv --system-site-packages --python /usr/bin/python3 /workspace/reference-env
uv pip install --python /workspace/reference-env/bin/python --no-deps \
  -r /workspace/v100/requirements-reference.txt
bash /workspace/v100/run.sh /workspace/results/new-attempt
```

`run.sh` requires two visible, idle V100s. It starts all four services, checks
health, submits ten concurrent rounds of real API requests, compares embeddings
and rerank scores with FP32 Transformers, and checks Gemma at 256 tokens plus
rejection at 257. It stops its model process groups on completion or failure;
it does not create or destroy a rental. Raw logs belong in ignored `runs/`.
A Vast campaign still needs the external label watchdog and cost policy in
[HARDWARE_PROFILES.md](../HARDWARE_PROFILES.md).

These are smoke checks: one request per service per round, embedding batches of
four texts, three rerank documents, and one English audio clip. They do not
certify production concurrency, multilingual ASR quality, long recordings, or
the full 2,048-token Snowflake/BGE input envelope.
