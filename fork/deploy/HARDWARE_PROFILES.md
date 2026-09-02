# Anonymous hardware profiles and venue policy

These names describe test capabilities, not owned infrastructure. A result says
“measured on `hopper-pcie-4-large`”; it never says who owns such a machine, where
one is deployed, or that the profile mirrors a private topology.

## Profiles

| profile | capability | valid questions |
| --- | --- | --- |
| `cpu-mock` | no accelerator | schemas, token construction, controller state, collection, reports |
| `cuda-budget-1` | cheapest available CUDA GPU | process lifecycle, HTTP routing, two-service orchestration with small fixtures |
| `hopper-budget` | cheapest Hopper venue that fits the reduced workload | Hopper-only kernels, speculative-decoding control flow, backend selection |
| `hopper-pcie-2` | two Hopper GPUs without an NVLink-class GPU fabric | isolated TP2 behavior and PCIe collective workarounds |
| `hopper-pcie-4-large` | four high-memory Hopper GPUs without an NVLink-class GPU fabric | primary four-GPU multi-model packing and PCIe behavior |
| `hopper-fabric-4-large` | four high-memory Hopper GPUs with NVLink/SXM-class fabric | memory-fit fallback and throughput upper bound when the PCIe profile is unavailable |
| `hopper-fabric-large` | larger fabric-connected Hopper venue, count recorded only in the result | historical/disjoint controls; never the default packing target |

“Large” means the per-GPU memory requirement is part of the question. Record the
actual GPU model and memory in a result only when necessary to interpret that
result; do not turn it into an ownership statement.

## Cheapest-valid-venue rule

Before searching offers, write the single property the run must prove and use
the first row below that preserves it:

| property | first venue |
| --- | --- |
| configuration syntax, prompt sizing, token counts, retries, evidence format | `cpu-mock` |
| rental/SSH/collection lifecycle | `cuda-budget-1` with a tiny public model |
| speculative API and metrics, architecture independent | cheapest GPU supporting the model or a small fixture |
| SM90/Hopper backend selection or crash | `hopper-budget`; prefer H100-class when memory permits |
| exact long-context memory fit | the smallest count/memory profile that can carry the real checkpoints |
| PCIe collective behavior | matching `hopper-pcie-*`; a fabric venue is not equivalent |
| final four-GPU multi-model proof | `hopper-pcie-4-large`; `hopper-fabric-4-large` only as a labeled fallback |

Using a stronger GPU “to be safe” is not neutral: it costs more and can hide the
memory pressure or interconnect behavior being tested. Escalate only after the
cheaper venue is ruled out for a written reason.

## Placement policy

GPU sets are not exclusive territories. Multi-model packing asks whether
processes can share the same physical GPUs within a memory and performance
policy. Test overlapping placements before adding devices:

1. spread both models across all four GPUs when that balances memory;
2. compare a smaller TP size for the smaller model when it reduces communication;
3. explicitly cap KV bytes and sequence count for the required service policy;
4. add GPUs only when the four-GPU profiles cannot satisfy the stated request.

The primary GLM+Gemma matrix is therefore four total GPUs: GLM TP4 plus Gemma
TP4 on all four, and GLM TP4 plus Gemma TP2 overlapping two of them. A six-GPU
disjoint placement is a diagnostic control, not the target proof.

## Privacy rules

Committed deployment documents and results must not contain:

- ownership or private-location claims;
- provider host, machine, offer, or instance ids;
- public/private IP addresses, SSH endpoints, GPU UUIDs, or account identifiers;
- language that equates an anonymous profile with a private deployment.

Raw rental records belong under ignored `runs/`. A reviewed committed record may
retain GPU model, count, memory, interconnect class, software versions, elapsed
time, and price because those explain the measurement without identifying a
private system.

## Development versus certification

Choose one before renting:

### Development/tuning

- one rental, one model download, many process restarts;
- detached watchdog and hard cap remain armed throughout;
- configuration/probe failures are evidence: save logs, stop process groups,
  push the fix, and relaunch on the same instance;
- keep the model cache and compilation cache warm;
- collect incrementally and destroy only at the session end or an abort condition.

### Certification

- exact committed bytes and revisions are already known to work;
- one unattended launch and fixed probe set;
- collect results, then destroy and verify provider state;
- any failure ends that certification run because tuning would change the object
  being certified.

A one-shot certification controller is not a development loop. If it is being
run a second time because code or configuration changed, the first run was
premature.
