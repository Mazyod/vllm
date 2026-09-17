# DeepSeek V4.1 Flash + Gemma 4 31B on shared TP4 H200

For the selected one-model-per-node layout, use the [dedicated deployment](../deepseek-gemma4-dedicated/README.md). This shared layout remains a measured alternative.

**Docker Swarm** configuration for two text-and-image services sharing
GPUs 0–3 on one x86-64, four-H200 node with 2 TB host RAM. This supersedes the
standalone Gemma Compose example for this deployment.

**Hardware-tested engine configurations.** Both TP4 engines have passed text, image,
structured-output, tool-call, and long-context probes on the pinned nightly.
The current allocation reports 6.71 full 32K contexts for Gemma and 7.95 full
1M contexts for DeepSeek. Four independent near-full prompts per model were verified generating together:
four Gemma prompts of 31,168–31,174 tokens and four DeepSeek prompts of
1,029,585–1,029,593 tokens. All eight streams advanced during a 5.1-second
overlap, retrieved the planted code, and left both engines healthy after
intentional cancellation. Memory was 130.9 GiB/GPU, with 9.5 GiB/GPU free.
This proves residency and overlapping generation, not eight completed
maximum-output requests or a production soak test.
See the [live hardware report](../../bench/configs/pending/gemma-4-nightly/shared-h200/index.html).
The test venue has NVLink; PCIe-only performance is not established. Swarm
packaging is statically validated; the engines were run inside the pinned image,
without a Swarm daemon on the test venue.

| Service | Engine file | API port / model | Context limit | KV bytes per GPU |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | [deepseek.yaml](deepseek.yaml) | 8000 / `deepseek-v4.1-flash` | 1,048,576 | 17,179,869,184 (16 GiB) |
| Gemma 4 31B FP8 block | [gemma4.yaml](gemma4.yaml) | 8001 / `gemma-4-31b` | 32,768 | 8,589,934,592 (8 GiB) |

Both use [stack.yaml](stack.yaml), which pins the tested AMD64 nightly manifest:

```text
vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0
```

Its image/config ID begins `sha256:0f9e42bf94b6`; source is
[`af1c01499b289be555c475669ba50a88e96d846e`](https://github.com/vllm-project/vllm/tree/af1c01499b289be555c475669ba50a88e96d846e),
vLLM `0.29.1rc1.dev187+gaf1c01499`. The source contains native DeepSeek V4.1
and Engram pinned-host-memory allocation and asynchronous prefetch. The recipe's
`0.30.0+` label describes the upcoming release; the necessary code is already
in this pinned nightly. All three model revisions are pinned in the engine YAMLs.

## Node preparation

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and a driver compatible with this CUDA 13 image. Swarm does not use Compose's
`deploy.resources.reservations.devices`. This stack requires the NVIDIA runtime
as Docker's default on the selected GPU node. If not already configured, run
these on that node during a maintenance window; restarting Docker can interrupt
existing workloads:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
docker info --format '{{.DefaultRuntime}}'
sudo mkdir -p /opt/huggingface /opt/vllm/cache/deepseek /opt/vllm/cache/gemma4
```

The runtime must honor `NVIDIA_VISIBLE_DEVICES=0,1,2,3` through its environment
GPU-selection path. See [NVIDIA's GPU enumeration documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration).
On the GPU node, verify that path without adding `--gpus` or `--runtime`:

```bash
docker run --rm -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --entrypoint nvidia-smi \
  vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0 -L
```

Expect four H200s. Use the existing model cache at `/opt/huggingface`, or change
both cache bind sources to its actual location. Prestage all three pinned checkpoints before starting the services. The measured
cold transfer took 75 minutes, longer than the engine initialization deadline.
Subsequent starts reuse this cache. Allow roughly
550 GB for the model payloads plus room for the image and compiler caches.
If Hugging Face access is gated, accept the model terms and authenticate the
shared HF cache before launch; do not put a token directly into this stack.

From this directory on the GPU node, stage the revisions from the YAMLs into
the same persistent cache used by both services:

```bash
docker run --rm -i --entrypoint uv \
  -v /opt/huggingface:/root/.cache/huggingface \
  -v "$PWD":/configs:ro \
  vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0 \
  run --no-project python - <<'PY'
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
for name in ("gemma4", "deepseek"):
    config = yaml.safe_load(Path(f"/configs/{name}.yaml").read_text())
    checkpoints = [config]
    if "model" in config.get("speculative-config", {}):
        checkpoints.append(config["speculative-config"])
    for checkpoint in checkpoints:
        snapshot_download(
            checkpoint["model"], revision=checkpoint["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
        )
PY
```

From the Swarm manager, label **exactly one** suitable node:

```bash
docker node update --label-add inference-pair=h200-tp4 <NODE>
```

Both services intentionally see the same four devices. Do not add
`NVIDIA-GPU=4` generic reservations to each service: on a node advertising four
exclusive GPU resources, the second service would remain pending. Swarm is not
accounting for HBM or shared GPU compute in this arrangement; dedicate the
selected GPUs to this pair. The placement label is a capability selector, not
a record of a private host.

## Start and update

Place all three YAML files together on the Swarm manager. Swarm distributes
the engine files as immutable Docker configs; only the cache directories need
to exist on the GPU node. The stack explicitly launches `vllm serve --config`;
do not prepend another `serve` to the image's default entrypoint.

For the first start, bring up Gemma, verify health on the GPU node, then enable
DeepSeek. This separates the first-load peaks and makes failures easier to
identify. Swarm does not provide Compose-style `depends_on` startup ordering.

```bash
docker stack config -c stack.yaml > /dev/null
DEEPSEEK_REPLICAS=0 docker stack deploy -c stack.yaml inference
docker service logs -f inference_gemma4
```

Once `curl --fail http://localhost:8001/health` succeeds **on the GPU node**,
run on the manager:

```bash
docker stack deploy -c stack.yaml inference
docker service logs -f inference_deepseek
docker stack services inference
```

The second deploy defaults DeepSeek to one replica. If `DEEPSEEK_REPLICAS` was
previously exported as zero in your shell, unset it first. Published ports use
host mode, so a remote client must address the GPU node, not an arbitrary Swarm
manager. APIs are `http://<GPU-node>:8000/v1` and `http://<GPU-node>:8001/v1`.

When editing an engine YAML, change its corresponding `DEEPSEEK_CONFIG_REV` or
`GEMMA_CONFIG_REV` before redeploying, for example:

```bash
export GEMMA_CONFIG_REV=20260917b
docker stack deploy -c stack.yaml inference
```

Keep those revision values in your deployment environment for subsequent
deploys. Change one engine at a time. Old configs remain available for rollback;
the stack uses `stop-first` updates to avoid overlapping old and new replicas.
Automatic restarts after a node reboot can still overlap startup, so the staged
first-start procedure is not a permanent ordering guarantee.

## Two redundant shared stacks

Use a different placement value for each node, and deploy one stack per value.
This keeps each pair together; simply labelling both nodes identically would
allow Swarm to schedule the two models independently. Cache directories and
pinned checkpoints must exist on both nodes.

```bash
docker node update --label-add inference-pair=pair-a <NODE_A>
docker node update --label-add inference-pair=pair-b <NODE_B>
INFERENCE_PAIR_NODE=pair-a DEEPSEEK_REPLICAS=0 docker stack deploy -c stack.yaml inference-a
INFERENCE_PAIR_NODE=pair-b DEEPSEEK_REPLICAS=0 docker stack deploy -c stack.yaml inference-b
# After each Gemma endpoint is healthy:
INFERENCE_PAIR_NODE=pair-a docker stack deploy -c stack.yaml inference-a
INFERENCE_PAIR_NODE=pair-b docker stack deploy -c stack.yaml inference-b
```

The selector also separates immutable config names. Preserve it on every update.
Route each model's API to its two healthy endpoints. Budget four long requests
per model across the pair if all must survive a node failure without extra
queueing; eight total assumes both nodes are available. This routing/failover
policy was not exercised by the single-node hardware experiment.

## Memory and model choices

DeepSeek retains its native mixed weight quantization, TP4 + expert parallelism,
Engram CPU offload, and automatic Hopper kernel selection. `dtype: auto` chooses
the nonquantized tensor dtype; it does not turn the checkpoint into BF16 weights.
DBO and microbatching are disabled, as required by this pinned Engram path.
Offloaded Engram weights and scales use pinned host RAM, not the `/dev/shm`
mount when `dp_shared_memory: false`. Unlimited memlock supports that allocation.
The 64 GiB tmpfs is a shared-memory ceiling, not a 64 GiB allocation at startup.

The original DeepSeek `gpu-memory-utilization: 0.95` automatic cache policy is
unsuitable for this shared deployment. Here both caches have fixed byte budgets.
Their combined KV allocation is **24 GiB per GPU**, excluding weights,
attention workspaces, CUDA graphs, vision, and runtime memory. The utilization
values (0.80 and 0.15) govern the initial free-memory threshold with explicit KV
bytes; they do not enforce an 80%/15% partition or a hard memory cap.

| Model | KV per GPU | Context | vLLM KV token capacity | Startup concurrency | Scheduler limit |
| --- | --- | --- | --- | --- | --- |
| Gemma MTP | 8 GiB | 32,768 | 219,744 | 6.71× | 8 |
| DeepSeek DSpark | 16 GiB | 1,048,576 | 8,339,905 | 7.95× | 32 |

At the same 8 GiB/GPU budget, Gemma reports **7.72× at 16,384** and
**9.29× at 8,192** tokens; both variants passed text, image, and near-window
retrieval probes while DeepSeek remained loaded. Halving context from 32K to
16K improves reported capacity by only 15%, so retain 32K for the four-request
target unless the extra capacity is more valuable than the larger window.

At a fixed 16 GiB/GPU, DeepSeek reports **14.29× at 524,288** and
**23.76× at 262,144** tokens. Both passed text, image, and near-window retrieval
probes with Gemma 32K still loaded. Those larger concurrency estimates were not
stress-tested to their advertised maximum; the explicit shared proof is four
requests per model at the original 32K/1M limits.

These startup estimates exceed the four-per-model target. They are not throughput
promises, and `max-num-seqs` does not guarantee that many full contexts. Context
limits include prompt and generated tokens. Hybrid cache allocation can change
with the context limit, so use measured fixed-budget comparisons in the report.

Gemma keeps FP8-block weights, FP8 KV, MTP with four draft tokens, and global
`TRITON_ATTN`. DeepSeek uses its native `fp8_ds_mla` cache, EP, Engram offload,
and DSpark with five draft tokens. Do not copy Gemma's attention override into
DeepSeek. Both use ordinary NCCL collectives: custom all-reduce, FlashInfer
all-reduce, symmetric-memory collectives, and fused all-reduce/RMS are disabled.
The Python frontend and skipped exhaustive DeepGEMM warmup are explicit for both.

MPS is excluded: two variants passed short checks but crashed under sustained
four-per-model traffic. Ordinary process sharing passed the five-minute C4/model test (220 requests),
but throughput varied materially: 92.5 Gemma + 82.9 DeepSeek output tokens/s
over 321 seconds including drain, versus 256.4 + 211.7 in an earlier one-minute
control. The runs use approximately 1K input / 256 output tokens. Shared-GPU
latency variability remains a limitation; capacity is not a throughput promise.
No extra generic weight offload or allocator override is required by this setup.

For more users, first distinguish capacity from latency. Larger caches admit
more resident tokens but do not add compute. Two shared nodes can retain both
services after one node fails; one dedicated node per model removes contention
but loses that redundancy. Route and queue long requests against a per-node
budget instead of treating the scheduler limit as a full-context guarantee.

Additional weight offload is a fallback, not a measured improvement here.
vLLM's [UVA offload documentation](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)
describes host-memory access during each forward pass and the need for a fast
CPU–GPU link. Its compatibility and speed with this exact DeepSeek configuration
have not been tested. Engram-only offload is the validated path in this report.

## Dedicated-node alternatives

The same stack can disable either service with `GEMMA_REPLICAS=0` or
`DEEPSEEK_REPLICAS=0`. Preserve the chosen replica counts on future deployments.
The report links the exact dedicated engine YAMLs; replace the corresponding
engine file and increment its config revision if selecting one of those budgets.
Do not use a dedicated cache budget while both models share the GPUs.

Dedicated DeepSeek at TP4, DSpark, 1M context, and 36 GiB KV/GPU reported
**17.90×** startup concurrency and passed the full probe suite. Its C4 control
completed 233 requests at **491.5 output tok/s** (about 123 per occupied slot),
versus **82.9 tok/s** for DeepSeek in the sustained shared run. KV budgets and
measurement durations differ, so this is a deployment comparison, not a
single-variable microbenchmark. Neither number is a full-context decode rate.

Dedicated Gemma at TP4, MTP, 32K context, and 112 GiB KV/GPU reported
**93.89×** startup concurrency with a scheduler limit of 128. Its 15 probes
passed; the isolated C4 control completed 52 requests at **635.9 output tok/s**
(about 159 per occupied slot). That last control ran only 20.9 seconds including
drain, versus 121.4 seconds for dedicated DeepSeek and 321.2 seconds for the
shared test. It is a short control, not a sustained-load guarantee. An earlier
Gemma C1/C4 pair overlapped due to an orphaned driver and is explicitly excluded.
None of the dedicated startup maxima were stress-tested to full capacity.

## Images, tools, and thinking

Both models retain their vision towers and admit up to eight images per request.
DeepSeek uses `mm-encoder-tp-mode: data`, supported by its pinned implementation
and recommended by its official recipe for image latency. Gemma's pinned class
does not advertise that mode, so it keeps the default encoder behavior.

Gemma's `max_soft_tokens: 280` is the recipe's ordinary image token budget. For
small text, OCR, or detailed documents, test 560 or 1120; higher values increase
vision compute and memory. This is a resolution tradeoff, not an image-count
limit. Video and audio are disabled in Gemma because this endpoint requires text
and images. The 31B model does not have the smaller variants' audio tower.

Both services enable their native reasoning and tool parsers, with thinking off
by default. To enable it per request, send:

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

For DeepSeek, optionally add numeric `reasoning_effort` inside that object.
Set `enable_thinking: true` explicitly when opting in; merely specifying an
effort does not override the configured false default in the pinned tokenizer.
Its unconfigured default effort is **50**, not 75. `drop_thinking: true` retains
the tokenizer's history policy. Gemma uses the pinned checkpoint's bundled chat
template, which already implements images, tools, and thinking; no external
template file or image rebuild is needed.

## DeepSeek speculation

DSpark is enabled in the canonical YAML and passed TP4 H200 text and image probes
on this exact image, including shared traffic with Gemma MTP. Keep the tested
scheduler and speculative limits. The earlier upstream issue was not a blanket
Hopper incompatibility. DSpark's benefit depends on workload and contention;
consult the simultaneous-load measurements rather than adding isolated model
throughputs. Gemma uses `mtp`; DeepSeek uses `dspark`.

## Verify on the target

Check both `/health` and `/v1/models` endpoints while both models remain loaded.
On the GPU node, set these for DeepSeek, then repeat with port 8001 and model
`gemma-4-31b`:

```bash
api_port=8000
api_model=deepseek-v4.1-flash
curl --fail "http://localhost:${api_port}/health"
curl --fail "http://localhost:${api_port}/v1/models"
curl --fail "http://localhost:${api_port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${api_model}\",\"max_tokens\":64,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*19? Return only the integer.\"}]}"
curl --fail "http://localhost:${api_port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${api_model}\",\"max_tokens\":128,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What word is printed on this traffic sign?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/stop_sign.jpg\"}}]}]}"
```

Expect `323` and identification of the STOP sign. A text-only check does not
exercise vision or Gemma's MTP after image prefill. Then exercise representative
multi-image requests, long prompts, tool calls, and concurrent traffic to both
APIs while observing `nvidia-smi` and the engine logs. Equivalent checks were run on H200. Repeat them on the deployment node to catch
runtime, driver, cache, and topology differences.

## Sources and local validation

Official recipes reviewed at recipes commit
`296ce72d28595433f5e5ac88fb46ade591664264` on 2026-09-17:

- [DeepSeek V4.1 Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- [Gemma 4 31B recipe](https://recipes.vllm.ai/Google/gemma-4-31B-it)
- [Gemma usage guide](https://github.com/vllm-project/recipes/blob/296ce72d28595433f5e5ac88fb46ade591664264/Google/Gemma4.md)
- [Pinned Engram implementation](https://github.com/vllm-project/vllm/blob/af1c01499b289be555c475669ba50a88e96d846e/vllm/models/deepseek_v41/nvidia/engram.py)
- [Docker service scheduling and updates](https://docs.docker.com/engine/swarm/services/)

`docker stack config`, YAML consistency checks, and the fork alignment gate pass.
All top-level engine/frontend options are present in the pinned source. The
fork preflight passed: **568 passed, 6 skipped; PREFLIGHT GREEN**.
Raw source snapshots and validation output stay in the ignored directory
`runs/deepseek-gemma4-config-review/`. The linked hardware report records the runtime evidence and its limits. These
smoke probes are not a model quality benchmark or a production soak test.
