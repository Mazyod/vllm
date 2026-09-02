# Multi-model deployment proofs

This directory answers a different question from `fork/bench`: not “does one
engine configuration work in isolation?”, but “do all services in a proposed
deployment fit and serve at the same time on one anonymous capability profile?”

Hardware names and privacy rules live in
[`HARDWARE_PROFILES.md`](HARDWARE_PROFILES.md). A profile is a test requirement,
not a statement about owned or privately deployed infrastructure.

A deployment proof has four parts:

1. `deployments/<name>.yaml` pins the container, model revisions, GPU placement,
   context policy, rental ceiling, and required evidence.
2. `engine/<name>/*.yaml` contains the exact `vllm serve --config` bytes used by
   each service.
3. `run-on-box.sh` stages the pinned snapshots, launches every service together,
   runs `probe.py`, and keeps raw logs and GPU telemetry.
4. `results/<date>-<name>/` is copied off the rental before teardown. A short
   reviewed record may be committed; raw artifacts stay under `runs/` unless
   they contain no sensitive or host-identifying data.

“Fits” means all services are healthy simultaneously and complete their policy
requests while the others remain loaded. A boot line or two separate rentals
is not a multi-model proof. Performance numbers must name the exact engine file
that produced them.

## Cost and campaign contract

Use development mode while any configuration, placement, runner, or probe may
change. Rent once, stage once, arm the detached watchdog, and restart only model
process groups after failures. Preserve each attempt's logs and reuse model and
compiler caches. Destroy at the planned end, hard cap, unsuitable venue, or
unrecoverable host/provider state.

Use `campaign.py` only for certification after the exact bytes and probes have
passed on the warm development rental. Certification is intentionally one-shot:
it collects and destroys on every outcome because changing anything would make
it a different object than the one being certified.

The first proof uses one verified on-demand eight-H200 offer, even though only
six GPUs are assigned. The unused pair is deliberate capacity evidence. The
rental policy is encoded in the deployment manifest:

- no interruptible bids;
- a hard price ceiling;
- a hard lifetime cap;
- the existing detached `fork/bench/watchdog.sh` armed by label;
- results collected before destroy;
- provider state read after destroy, with a retry if the instance remains.

Never rent a deployment with a bare `vastai create` and no watchdog. The
fork's release-gate lessons about SSH key ownership, mutable image tags, and
unconfirmed teardown apply unchanged.

The unattended **certification** command is:

```bash
set -m
uv run --no-project --with httpx --with pyyaml -- \
  python -m fork.deploy.campaign --out runs/glm53-gemma4-6xh200 &
driver=$!
nohup fork/bench/watchdog.sh fork-deploy-glm53-gemma4-6xh200 "$driver" 7200 \
  >>/tmp/fork-deploy-glm53-gemma4-watchdog.log 2>&1 &
wait "$driver"
```

Read the watchdog's final line. `done: ... is torn down` is the external
confirmation; `NOTHING WATCHED` does not prove anything.

## Historical diagnostic proof

[`glm53-gemma4-6xh200.yaml`](deployments/glm53-gemma4-6xh200.yaml) assigns:

- GPUs 0–3: GLM-5.3-Flash native FP8 weights, TP4, BF16 KV on Hopper, native
  MTP k=3, 131,072-token context;
- GPUs 4–5: Gemma-4-31B FP8 block weights, TP2, FP8 KV, Google assistant draft
  MTP k=4, 32,768-token context;
- GPUs 6–7: unassigned.

GLM expert parallel is deliberately off: the isolated H200 campaign measured a
5–12% loss plus extra activation and CUDA-graph memory. Gemma is dense and has
no routed experts, so expert parallel does not apply. Tensor parallelism is the
minimum that carries each checkpoint: four H200s for the roughly 306 GiB GLM
checkpoint and two for Gemma.

The Gemma file explicitly caps both sequence count and KV bytes. This is the
escape hatch for the observed failure in which automatic memory sizing fills
the card and leaves insufficient headroom for the external draft and runtime
workspace.

Gemma also runs with `VLLM_USE_V2_MODEL_RUNNER=0` scoped only to its process.
The GLM vendor image defaults Gemma4 to V2, whose FlashInfer draft backend
refuses sliding-window attention on SM90. The stable V1 path is required for
Gemma's external MTP; GLM stays on V2.

The first co-resident proof passed on 2026-09-02. See the
[`RECORD.md`](results/20260902-glm53-gemma4-6xh200/RECORD.md) and its small,
committed evidence files. Full raw logs remain in the ignored local run
directory named by the record.

It used disjoint GPU sets and is retained as evidence, not as the target packing
shape. The next primary matrix is four total GPUs:

- GLM TP4 plus Gemma TP4, both sharing all four GPUs;
- GLM TP4 plus Gemma TP2, with Gemma sharing two of the same four GPUs.

Run `hopper-pcie-4-large` first. Use `hopper-fabric-4-large` only when the PCIe
profile cannot be rented, and label its performance as a fallback/upper bound.
Try the overlapping placements and smaller explicit cache policies before
adding devices.
