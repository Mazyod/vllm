# Gemma 4: nightly FP8 KV-cache experiments

[Open the HTML report](report.html). Each section is a full-viewport slide.
The development session completed on 2026-09-17.

**FP8 KV cache works with MTP and tensor parallelism on the tested H100
nightly when `attention-config.backend: TRITON_ATTN` is explicit.** TP1 and
TP2 passed. Default FP8 + MTP fails in FlashInfer's FA2 path. V2 works with
the override; V1 plus the same override also passed.

The [32K TP2 recipe with MTP](tp2-fp8-on-triton-long-small.yaml) used
23.4 GiB per GPU after the workload, with a 24.4 GiB observed peak. Its
2.5 GiB-per-GPU FP8 cache supported a 29,109-token retrieval prompt.
The [text-only MTP variant](tp2-fp8-on-triton-long-small-text.yaml) used
22.1 GiB per GPU. The HTML report compares lower-memory options and their
speed and context trade-offs.

A [TP4 Docker Compose adaptation](../../../../deploy/gemma4-tp4/README.md)
is available separately. TP4 was not part of this hardware experiment.

The ledger contains 33 runs: 28 passed, totaling 504 successful correctness
checks and 688 native benchmark requests. Five runs reproduced unsupported
backend selections.
Eight setup/client attempts are retained in the raw evidence but excluded
from these results. Native benchmarks generated 176,128 output tokens, with
every measured request completing its required 256 tokens.

The pinned AMD64 image is
`vllm/vllm-openai@sha256:f1491cb3abc84aca4e319c2b3f3f23644f0d9e19fdb9012c80818d3f30e5a0a0`.
Its image ID is
`sha256:0f9e42bf94b6abc20ef1f5949828b24ba5390430bd9a160bd6e93683d3a33b05`.

[identity.json](identity.json) records software, checkpoint revisions and the
anonymous hardware profile. [measurements.json](measurements.json) records
each configuration's digest, environment, memory, probes and native benchmark
results. Linked YAML files are the exact bytes used by those launches.
The model paths point to pinned snapshots staged under `/workspace/hf`.
Apply each record's environment as well as its YAML. The 35% memory-utilization
setting in the tighter recipes changes the startup free-memory check; explicit
KV bytes determine the cache allocation, and utilization is not a hard cap.

Rebuild the standalone HTML, without dependencies:

```bash
uv run --no-project -- python fork/bench/configs/pending/gemma-4-nightly/render_report.py
```

Raw engine logs, model outputs, telemetry, launch scripts and benchmark argv
remain in ignored `runs/gemma-nightly-20260916/`. One watched development
rental and one download of each checkpoint were reused throughout. Rental
destruction was confirmed after 133.6 minutes. Estimated cost was $10.18,
including compute, storage and a conservative full-image transfer allowance.

Validation: 568 tests passed, 6 skipped, `PREFLIGHT GREEN`; alignment passed.
The standalone report was checked in desktop and mobile browsers.
