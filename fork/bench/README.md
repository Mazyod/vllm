# fork/bench

A release gate for this fork: for a candidate upstream release, does the patch
series still earn its place, and does the engine still behave the way the image's
target configurations need?

- [DESIGN.md](DESIGN.md) — what it measures and why.
- [RUNBOOK.md](RUNBOOK.md) — the session protocol.
- [configs/README.md](configs/README.md) — the configuration contract.
- [configs/CATALOG.md](configs/CATALOG.md) — configurations and recorded runs.

## Quick start

One command answers "can this release ship", from no machine to results on
disk with nothing left running:

```bash
export HF_TOKEN=...
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag v0.27.1 --image <IMAGE> --out runs/v0.27.1 --phase 4 --rent
```

`--rent` spends money. Run the free checks below first — they prove the whole
orchestration against fixtures and a mock, so nothing about it is discovered on
a machine that bills by the second.

Everything except the GPU phases runs locally and free:

```bash
# whole suite plus a full dry run of the gate
bash fork/bench/preflight.sh

# phase 0: has upstream absorbed any patch yet?
uv run --no-project -- python -c \
  "from pathlib import Path; from fork.bench.static import brief; \
   print(brief('v0.27.1', Path('.')))"

# the gate itself, replaying fixtures against the mock
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag v0.27.1 --out runs/dry --dry-run
```

`uv run` needs `--no-project` here. Without it uv resolves the vLLM project
itself, which the local CPU path does not need and which cannot resolve on a
machine with no CUDA wheel. Gate commands need `--with pyyaml` to load the
tag-selected configuration store.

This directory is developer tooling. It is never copied into the image.

Extending the harness? Read [LESSONS.md](LESSONS.md) first. Ten defects
survived a full local suite and a green dry run before the first GPU session
found them, and four of those produced confident wrong answers rather than
visible failures.

## Not built yet

Known gaps, recorded so they are not mistaken for oversights:

- **Weight-quantization A/B.** The same-box controls cover the fp8 *kv cache*.
  Block-fp8 *weights* are a property of the checkpoint, so comparing them needs
  a second checkpoint rather than a flag.
- **Output quality.** Unchanged from [DESIGN.md](DESIGN.md) § Non-goals: nothing
  here catches an accuracy regression that serves normally.
