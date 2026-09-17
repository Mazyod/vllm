# Dedicated DeepSeek and Gemma: one four-H200 node each

Canonical Docker Swarm deployment for **two separate nodes**, each with four
H200 GPUs and 2 TB host RAM. DeepSeek occupies one node; Gemma occupies the other.
Both support text, up to eight images per request, tool calls, and optional
reasoning. This is the selected performance-first layout; there is one replica
of each model and no model-level failover.

The engine YAMLs are **byte-for-byte copies of the dedicated hardware runs**.
They are the tested baseline, not a claim of a globally optimal configuration.
No new GPU run was performed while packaging this deployment.

| | DeepSeek V4.1 Flash | Gemma 4 31B FP8 block |
| --- | --- | --- |
| Engine YAML | [deepseek.yaml](deepseek.yaml) | [gemma4.yaml](gemma4.yaml) |
| GPUs | TP4 + expert parallelism | TP4 |
| Context, input + output | 1,048,576 | 32,768 |
| GPU KV budget | 36 GiB/GPU; 144 GiB/node | 112 GiB/GPU; 448 GiB/node |
| KV format | `fp8` → native `fp8_ds_mla` | `fp8`, global `TRITON_ATTN` |
| Speculation | DSpark, 5 draft tokens | MTP assistant, 4 draft tokens |
| Startup concurrency estimate | 17.90× full windows | 93.89× full windows |
| Scheduler limit | 32 sequences | 128 sequences |
| Observed GPU peak | 129.76 GiB/GPU | 126.25 GiB/GPU |
| C4 aggregate output throughput | 491.5 tok/s | 635.9 tok/s |
| C4 observation duration | 121.4 seconds | 20.9 seconds |
| C4 requests completed | 233 | 52 |
| Direct API | DeepSeek node port 8000 | Gemma node port 8001 |
| Served model | `deepseek-v4.1-flash` | `gemma-4-31b` |

Throughput used ~1K input / 256 output tokens, temperature zero, four active
requests to the single model. It includes prefill and drain, not decode-only
speed at maximum context or startup concurrency. Both models passed 15 probes,
including images, tools, structured output and near-window retrieval. Startup
maxima were not load-tested; Gemma's final throughput control was short. Two
previous overlapping Gemma drivers were excluded from the results.

The venue had NVLink and driver 595.71.05. PCIe-only throughput is unmeasured.
Physical capacity was 140.4 GiB/GPU. See the
[executive slides and detailed measurements](../index.html).
Swarm packaging and the optional router have separate validation below.

## Files and pinned images

Deploy the two engines with [stack.yaml](stack.yaml). Model revisions are pinned
inside their engine YAMLs. The exact engine image is:

```text
vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0
```

This is the established nightly whose image/config ID starts `sha256:0f9e42bf94b6`,
source `af1c01499b289be555c475669ba50a88e96d846e`, vLLM
`0.29.1rc1.dev187+gaf1c01499`. Do not substitute a moving nightly tag without
revalidation.

The optional [router-stack.yaml](router-stack.yaml) and [router.yaml](router.yaml)
use the confirmed fork release `openimage/production-stack-router:v0.1.12`, pinned
by its published manifest digest:

```text
openimage/production-stack-router@sha256:7144e56c84abcb3e3f42a5eb37dcee16a46945f980a04d3d51e4134b51768733
```

The tag, successful release run, and digest were checked against the registry
and fork records. This identifies the published release, not an inspection of
an existing production container. Its routing implementation matches upstream
0.1.12; the fork adds reliability, timeout, statistics, and other route fixes.

## Prepare each GPU node

Use Docker Swarm and a CUDA-13-compatible NVIDIA driver. The NVIDIA Container
Toolkit must be Docker's default runtime on each GPU node; Swarm does not apply
Compose GPU-device reservations. If needed, configure it during an appropriate
maintenance window:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
docker info --format '{{.DefaultRuntime}}'
sudo mkdir -p /opt/huggingface /opt/vllm/cache/deepseek /opt/vllm/cache/gemma4
```

Verify the same environment-based device enumeration used by the stack:

```bash
docker run --rm -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --entrypoint nvidia-smi \
  vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0 -L
```

Expect four H200s. Dedicate those GPUs to the assigned engine and stop any
previous shared-model services before starting this allocation. Labels control
placement; they do not prevent unrelated services from using the GPUs.

Prestage checkpoints once on **each model's assigned node**. Serving is offline
for Hugging Face downloads, as in the experiments; missing checkpoints must be
staged before launch. The measured initial download exceeded the engine's
one-hour startup deadline. Authenticate the HF cache/accept model terms first
if required. Reserve roughly 510.31 GB for DeepSeek, or 34.27 GB for Gemma plus
assistant, in addition to images and runtime caches.

From this configuration directory on the GPU node, select only that node's model:

```bash
engine_model=deepseek  # Set gemma4 on the Gemma node.
docker run --rm -i --entrypoint uv \
  -v /opt/huggingface:/root/.cache/huggingface \
  -v "$PWD":/configs:ro \
  vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0 \
  run --no-project python - "$engine_model" <<'PY'
from pathlib import Path
import sys
import yaml
from huggingface_hub import snapshot_download
assert sys.argv[1] in ("deepseek", "gemma4")
config = yaml.safe_load(Path(f"/configs/{sys.argv[1]}.yaml").read_text())
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

Subsequent launches reuse the model cache and the per-model runtime-cache bind.
The containers retain separate compiler/Triton caches. No LMCache or extra CPU
KV tier is required by this measured baseline.

## Deploy the engines

On the manager, label **exactly one different node per model**:

```bash
docker node update --label-add inference-model=deepseek <DEEPSEEK_NODE>
docker node update --label-add inference-model=gemma4 <GEMMA_NODE>
docker stack config -c stack.yaml > /dev/null
docker stack deploy -c stack.yaml inference-dedicated
docker stack services inference-dedicated
docker service logs -f inference-dedicated_deepseek
# In another terminal:
docker service logs -f inference-dedicated_gemma4
```

Each node starts its own model independently. GPU indices 0–3 are container-local
on the assigned node. Configs are distributed by Swarm; the model/runtime bind
directories must exist on the respective GPU node. The stack creates the
attachable overlay network `inference-dedicated`.

Direct health/API endpoints are `http://<DEEPSEEK_NODE>:8000/health` and
`http://<GEMMA_NODE>:8001/health`, with OpenAI APIs under `/v1`. Published engine
ports use host mode; use the assigned GPU node address.

For engine-file changes, increment `DEEPSEEK_CONFIG_REV` or `GEMMA_CONFIG_REV`
before redeploying and preserve that value on later deployments. Updates and
rollbacks use `stop-first`; this deployment cannot serve a model during its
restart. Keep the existing config for rollback until validation completes.

## Use the existing router, or deploy the optional router

There is one backend per model, so `roundrobin` is sufficient and preserves
engine-local prefix reuse. A session or prefix router cannot improve locality
within a single destination. Native GPU prefix caching stays enabled.

For an **existing Swarm router**, attach it to `inference-dedicated` and merge
the `static_models` entries from [router.yaml](router.yaml) into its current
configuration. Preserve unrelated routes. Restart to apply backend timeout
changes; those fork flags are startup-only. For example, network attachment is:

```bash
docker service update --network-add inference-dedicated <EXISTING_ROUTER_SERVICE>
```

If the router runs outside that overlay, replace the internal URLs with the
GPU-node addresses and published ports above. The provided internal service
names assume the engine stack is named `inference-dedicated`.

For a **new router instead**, after both engines are healthy:

```bash
docker stack config -c router-stack.yaml > /dev/null
docker stack deploy -c router-stack.yaml inference-router
curl --fail http://localhost:8080/health
curl --fail http://localhost:8080/v1/models
```

Use a Swarm node address instead of localhost when accessing remotely. The
router publishes port 8080 through the routing mesh and uses no GPU. Do not
launch it on a port already used by the existing router. It has one replica;
this package does not provide HA for the router or the models.

The router silence bound is **1,800 seconds**, versus the fork's 300-second
default. The measured four-request cold 1M prefill reached 608 seconds to first
content token. The new bound accommodates that observation with margin; it is
not an unlimited queueing or output-duration guarantee. Prefer streaming for
long responses. Align any external proxy/client timeout with the intended
workload. To edit the optional router YAML, bump `ROUTER_CONFIG_REV` and redeploy;
Swarm config objects are immutable.

## Why these flags remain

Both engines retain the measured V2 runner, Python frontend, skipped exhaustive
DeepGEMM warmup, and ordinary NCCL collectives. FlashInfer/custom/symmetric-memory
all-reduce and fused all-reduce/RMS are disabled. MPS is absent because both C4
MPS variants crashed. These are the settings behind the dedicated measurements,
not newly tuned flags.

Explicit KV bytes determine the GPU cache budget. `gpu-memory-utilization`
(0.80 DeepSeek, 0.15 Gemma) is a startup free-memory threshold in this mode, not
an allocation cap: Gemma still reserves 112 GiB KV/GPU. These values deliberately
match the tested YAML bytes. Do not run both dedicated allocations on one node.

DeepSeek offloads only Engram (~188.8 GiB across the node) into pinned host RAM.
The 64-GiB `/dev/shm` mount is a separate IPC ceiling, not where that Engram
allocation resides. Gemma uses an 8-GiB IPC mount. A future native CPU KV tier
would require larger mounts; it is a separate unvalidated experiment described
in the [cache research](../CACHE_RESEARCH.md).

Gemma retains the measured 280-soft-token image budget. Higher image resolutions
or OCR-oriented budgets require new quality/memory checks. Reasoning is disabled
by default for both; clients may request it using
`chat_template_kwargs: {enable_thinking: true}`. Context limits include input,
image tokens and generated output. Scheduler limits do not promise that many
simultaneous maximum-context requests.

## Optional next step: local RAM prefix retention

Prepared [native RAM KV candidates](ram-kv/README.md) add a CPU prefix-cache tier
without changing the measured GPU budgets or router. Use the model-specific
`stack.ram-kv-deepseek.yaml` or `stack.ram-kv-gemma4.yaml` overlay to try one model
at a time. They allocate 256 GiB DeepSeek / 768 GiB Gemma **across each engine**
and enlarge `/dev/shm`. These candidates are statically checked, not GPU-validated,
and are not enabled by the baseline `stack.yaml`.

## Validation and limits

- Engine YAML SHA-256 values match the hardware receipts exactly; environment
  values match the measured launch settings, with only cache-directory relocation.
- Both Swarm files render with `docker stack config`; the Swarm topology itself
  has not been hardware-deployed by this experiment.
- The pinned router image was checked locally against CPU mock backends for
  health, model discovery, text/image-body routing and timeout parsing. A derived
  0.25-second test bound also passed pre-header 504, terminal SSE stall errors,
  healthy streams beyond the read bound, and counter-cleanup checks. This checks
  routing/configuration, not image inference or a 30-minute timed request.
- The fork preflight and alignment gate pass. The hardware report remains the
  authority for measured engine performance and its limits.

Release provenance: [fork maintenance record](https://github.com/Mazyod/production-stack/blob/6970618c496817b085444d2a2e64949fcd05d9d6/docs/fork-maintenance.md),
[successful router release](https://github.com/Mazyod/production-stack/actions/runs/34873842221).
