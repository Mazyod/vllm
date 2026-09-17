# Gemma 4: TP4, FP8 KV, MTP

For the combined DeepSeek + Gemma deployment on the same four H200s, use the
[canonical Swarm configuration](../deepseek-gemma4-tp4/README.md). This directory
retains the earlier standalone Compose example.

A TP4 adaptation of the [tested Hopper nightly configuration](../../bench/configs/pending/gemma-4-nightly/RECORD.md).
**TP1 and TP2 were hardware-tested; TP4 and this Compose deployment were not.**
The pinned target and assistant each have 32 query heads, 16 sliding-attention
KV heads, and four global-attention KV heads, all divisible by four.

Place `compose.yaml` and `vllm.yaml` together, then run:

```bash
docker compose up -d
docker compose logs -f gemma4
```

The API is available on port **8001**, with model name **`gemma-4-31b`**:

```bash
curl --fail http://localhost:8001/health
curl http://localhost:8001/v1/models
```

The first launch downloads the pinned target and assistant revisions. Both
model weights and compiler caches persist in named volumes across restarts.
No custom Dockerfile is needed. The host must have Docker's NVIDIA GPU support
configured; [device reservations select GPUs 0–3](https://docs.docker.com/compose/how-tos/gpu-support/).

`TRITON_ATTN` is the tested fix for FP8 + MTP. The separate all-reduce settings
carry the fork's PCIe workarounds; this exact combination was not measured in
the nightly NVLink experiment. They do not disable Triton attention or MTP.

The 1.25 GiB-per-GPU KV budget retains the tested TP2 configuration's **5 GiB
aggregate cache allocation** while moving to four GPUs. This scaling is an
adaptation, not a measured TP4 capacity or total VRAM figure. The window is
32K, with eight admitted sequences; eight simultaneous full-length 32K
requests are not promised. The 25% utilization setting controls the startup
free-memory check; explicit KV bytes control cache allocation.

The full checkpoint remains enabled. To use the separately tested text-only
mode, add `language-model-only: true`; this disables image/video inputs.
