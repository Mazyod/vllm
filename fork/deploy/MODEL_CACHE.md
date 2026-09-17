# Vast.ai H200 model warm-up optimization

**Status: proposed, not hardware-validated.** Provider and Hugging Face command
references checked on 2026-09-17. No rental, clone, throughput, or savings
measurement has validated this workflow. The existing rental controllers do not
automatically provision or attach these volumes.

## Goal

Stage a large model (for example, 500 GB) before renting a 4×H200 instance.
Treat persistent model storage as reusable infrastructure with its own budget
and retention deadline. Keep the [hardware policy](HARDWARE_PROFILES.md) and
[development/certification contract](../bench/RUNBOOK.md#before-spending-name-the-question-and-campaign-type).

```text
Hugging Face → cheap staging instance → persistent model volume
                                         ├─ same machine → GPU instance
                                         └─ clone to target machine → GPU instance
```

The intended order is **download/copy/verify, then rent expensive compute**.
Local weights remove the network download, not container startup, disk reads,
GPU weight loading, or compilation.

## Storage constraints

Vast's [volume guide](https://docs.vast.ai/guides/instances/storage/volumes)
documents local volumes tied to one physical machine, attachment to Docker
instances on that machine, and reuse after deleting the compute instance.
They cannot attach directly across machines or to VM instances. The volume's
allocated size cannot be changed in place.

For approximately 500 GB of required files, 650–700 GB is an initial estimate.
Inventory the pinned revision first: alternate weight formats, draft models,
temporary files, and extra revisions can exceed that allowance. Container disk
is separate from the model volume. Confirm that `/data` is the actual persistent
mount before downloading; an ordinary directory in the container is insufficient.

Before any paid create, run CPU preflight, choose development or certification,
and arm the external label watchdog for each compute rental. Record elapsed
time, hourly rates, transfer bytes and charges, and cache reuse. The current
watchdog destroys instances, **not persistent volumes**: separately budget and
track storage retention, including clones. Keep provider identifiers in ignored
`runs/` records; committed evidence uses anonymous capability profiles.

## Stage once on cheap compute

Search for storage on a machine that can supply the intended GPUs, or plan to
clone the cache later. Do not assume a cheap GPU elsewhere shares the target's
disk. An available storage offer does not establish future GPU availability;
recheck compute offers before committing to the transfer.

After the cost controls above are in place, the documented storage commands are:

```bash
vastai search volumes
vastai create volume <VOLUME_OFFER_ID> --size 700 --name model_cache
vastai show volumes
```

Use the cheapest compatible staging compute on that same machine. Vast's
[instance CLI](https://docs.vast.ai/cli/reference/create-instance) supports
`--link-volume <VOLUME_ID> --mount-path /data` at creation. Include those options
in the watched launch, with sufficient container disk and the intended image.

Prefer the native Hugging Face cache when the runtime uses repository IDs:

```bash
export HF_HOME=/data/huggingface-cache
export HF_HUB_CACHE=/data/huggingface-cache/hub
export HF_XET_HIGH_PERFORMANCE=1

uv run --no-project --with huggingface_hub -- \
  hf download <ORG>/<MODEL> --revision <COMMIT_SHA> --dry-run
uv run --no-project --with huggingface_hub -- \
  hf download <ORG>/<MODEL> --revision <COMMIT_SHA>
uv run --no-project --with huggingface_hub -- \
  hf cache verify <ORG>/<MODEL> --revision <COMMIT_SHA> --fail-on-missing-files
```

Run the dry run locally before choosing volume capacity; run download and
verification on the staging instance. Pin an immutable model revision and record
the resolved tooling versions. The example downloads the entire repository;
if selecting only required files, retain an explicit file/checksum manifest
and validate that set instead of ignoring missing-file failures.
The [HF CLI](https://huggingface.co/docs/huggingface_hub/main/package_reference/cli)
documents checksum verification and the missing-file failure option.

[Xet high-performance mode](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf_xet_high_performance)
tries to use available CPU, disk, and network capacity. Measure throughput on the
cheap host; the flag cannot overcome CPU, disk, or network bottlenecks. Prefer
the current `hf download` path over a hand-built serial downloader.

For explicit local paths, use `--local-dir /data/models/<MODEL>/<COMMIT_SHA>`
for both download and verification. This is a different layout from the shared
Hub cache: setting `HF_HOME` does not make a local-directory download reusable
by repository ID. Serve that local path, or use the native cache consistently;
do not stage both copies. See the [HF download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

After verification, stop writers and delete only the staging compute instance.
Confirm provider state and that the volume remains. Keep credentials out of
the reusable cache; use a scoped runtime token only for models that need it.

## Reuse locally or clone before GPU launch

For the same machine, attach the existing volume to the new GPU instance.
For another machine, select a destination **volume offer**, then use the
documented [clone command](https://docs.vast.ai/cli/reference/clone-volume):

```bash
vastai clone volume <SOURCE_VOLUME_ID> <DESTINATION_VOLUME_OFFER_ID> --size 700
```

The destination allocation must be at least the source volume's allocated size,
not merely its used bytes, and must fit the destination offer. Thus a 700 GB
source cannot be cloned into 650 GB. Budget both retained volumes and transfer.

Wait for confirmed completion before renting the full GPU instance. A successful
request submission is not evidence of a complete clone. The exact completion
signal, detached-volume cloning behavior, and destination integrity checks are
still validation tasks. If verification requires compute, use cheap compute
on the destination first. Keep the source unchanged during copying.

Use the same cache mount, model revision, and cache settings in the runtime.
For this repo, `run-on-box.sh` explicitly downloads to `/workspace/hf`; exporting
`HF_HOME` alone will not override its `--cache-dir`. Existing engine files may
also name snapshot paths. Align those paths with the mounted cache before use.
A small offline load check should prove cache reuse before attempting 500 GB.

If only a one-GPU slice on the target machine is available for staging, use it
with persistent storage and later create a separate four-GPU instance. Do not
assume in-place expansion or continued availability of the other GPUs. At equal
per-GPU rates and staging duration, this reduces staging GPU-hours by 75%; it
does not promise a 75% reduction in total cost.

## Compare total cost

For **500 decimal GB**, ideal transfer times are:

| sustained bandwidth | theoretical time |
| --- | --- |
| 1 Gbit/s | 66.7 minutes |
| 2 Gbit/s | 33.3 minutes |
| 4 Gbit/s | 16.7 minutes |
| 8 Gbit/s | 8.3 minutes |

Actual times include transfer overhead and contention. Prefer measured fast
networking (4 Gbit/s or better when available), NVMe, adequate CPU/RAM, capacity,
and reliable hosts. Advertised bandwidth alone is not a throughput measurement.

Compare avoided four-GPU download time against staging compute, storage for the
entire retention period, clone ingress/egress, duplicate storage, and verification
compute. Persistent storage is beneficial only when that total is lower. Record
the actual bill and time to first successful inference, including model loading.

## Validate before automating

First use a tiny pinned public model and the cheapest compatible venue to prove:

1. Volume mount, download, checksum verification, and survival after compute deletion.
2. Same-machine reattachment and offline model loading without downloading again.
3. Cross-machine clone with no expensive GPU rented, observable completion,
   destination checksums, and offline reuse.
4. Storage/transfer billing, watchdog teardown, and deliberate volume retention
   or cleanup under a separate storage budget.

Only then measure the large-model path. Future provisioning can select the
target offer, check its local cache, clone if needed, wait and verify, recheck
GPU availability, and launch vLLM with the verified volume. Automation remains
deferred until those provider behaviors and savings have evidence.
