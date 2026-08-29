# Release gate runbook

The session protocol. Phases run in order; each has a precondition that must
hold before the next begins. Deviate only to triage a failure, and record what
was done. See [DESIGN.md](DESIGN.md) for why each probe exists and
[LESSONS.md](LESSONS.md) for what the first hardware session cost to learn.

Every local Python command uses `uv run --no-project`. Without the flag `uv`
tries to resolve vLLM itself, which this tooling never needs and which fails
outright on a machine with no CUDA wheel. Gate commands also include `--with
pyyaml` because selecting `--tag` loads that release's YAML configuration
store; commands that issue requests add `--with httpx` as well.

## Phase 0 — static (free, local)

1. `git fetch upstream --tags`
2. Render the brief:

   ```bash
   uv run --no-project -- python -c \
     "from pathlib import Path; from fork.bench.static import brief; \
      print(brief('<TAG>', Path('.')))"
   ```

3. Read the upstream release notes and run `scan_release_notes` over them.
4. **Gate:** decide whether this release warrants GPU spend. A release that
   touches none of the flagged areas and absorbs no patch may not.

## Phase 0.5 — CPU preflight (free, local)

1. `bash fork/bench/preflight.sh`
2. **Gate:** `PREFLIGHT GREEN`. Do not provision otherwise. This includes a full
   dry run of the gate, so the orchestration is proven before any spend.

## The quick check

If you only have time for one thing, run the shipping topology on a rented box:

```bash
export HF_TOKEN=...   # gated checkpoints do not download without it
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag <TAG> --image <IMAGE> --out runs/<TAG> --phase 4 --rent &
driver=$!
nohup fork/bench/watchdog.sh "fork-bench-<TAG>" "$driver" \
  >>/tmp/fork-bench-watchdog.log 2>&1 &
wait "$driver"
```

The watchdog is not optional. The reaper inside the driver is a thread: it dies
with the driver, and it stands down before teardown runs. On 2026-08-29 a
`--rent` driver was killed before it had written `runs/<TAG>/rental.json`,
leaving a 2xH100 billing at $4.04/hr with nothing watching it — no thread, no
file, no id. `fork/bench/watchdog.sh` keys on the instance **label**, which the
machine carries from the moment it is created, so it needs no file to exist and
covers the rental from its first second. It sweeps that label when the driver
has been gone for the grace period (default five minutes) or when its cap
(default 9900s) elapses, confirms the sweep against the provider, and writes
`GIVING UP` to its log in the one case that still needs a human.

`--rent` is phases 1 and 5 done for you: it searches its preference list once,
rents one instance, arms the reaper before anything else, pushes this tree onto
the box, runs the gate there, brings the results back, and destroys the instance
— confirming with the provider that it is gone. PCIe is preferred, but the
fallback requirements also admit H100 SXM. The topology gate may refuse that
box; the campaign collects the refusal and destroys it, but does not re-hunt.

Phase 4 itself is TP2 with the all-reduce workarounds on, carrying the full
receipt and behavioural probe set, plus the N3 arm with the workarounds off. It
answers "will this release serve the way production needs" without the
leave-one-out matrix. Exit code 0 means every gating probe passed.

Run the full `--phase 2 --phase 3 --phase 4` when you also want to know whether
the patch series can shrink.

## Phase 1 — provision

`--rent` does all of this. The steps are here because a failure mid-run leaves
you finishing by hand, and because the guarantees are worth knowing.

1. Search for an on-demand offer: 2 Hopper GPUs, at least 150 GB disk, and a
   directly mapped port. Never an interruptible bid — being outbid part-way
   truncates the run and voids its numbers.
2. Rent it. Arm the reaper immediately, before anything else — it owns teardown
   on a hard cap regardless of what the driver is doing. The instance id is
   written to `runs/<TAG>/rental.json` in the same breath, before the boot is
   waited on, so nothing outside the driver has to guess what is billing.
3. Run `nvidia-smi topo -m` and classify the GPU0-GPU1 link.
4. **Gate:** a `NV*` link disqualifies the box; disabling peer access on an
   NVLink pair does not reproduce the deployment topology. In an automated
   `--rent` run the gate refuses the box and the campaign collects and destroys
   it. Re-hunting is the operator's next move: start another `--rent` run (or
   rent a confirmed PCIe-only pair manually). One campaign never re-rents.
5. Stage both models' weights in parallel.
6. Before any engine launch, validate the committed files with the installed
   release's real parser:

   ```bash
   cd /workspace/bench
   python3 -m fork.bench.config_validation --tag <TAG>
   ```

   **Gate:** the command prints the installed release version. A version
   mismatch or parser rejection stops the run. The local and docker launchers
   perform this validation automatically before topology checks or launches.

The instance is booted *from* the image under test, so there is no daemon to
hand a container to. On the box the gate runs with `--launcher local`, which
starts the engine as a child process; `--rent` passes that for you.

Whatever goes wrong, one command cleans up after a run whose driver died:

```bash
uv run --no-project -- python -c \
  "from fork.bench.provision import sweep; from fork.bench.vast import VastCli; \
   print(sweep(VastCli(), 'fork-bench-<TAG>'))"
```

## Phase 2 — correctness (both GPUs in parallel)

Run every phase 2 profile. GPU 0 takes the Gemma profiles, GPU 1 the Qwen ones.
These are pass/fail and not timing-sensitive, so running both at once is free.

```bash
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag <TAG> --image <IMAGE> --out runs/<TAG> --phase 2
```

**Gate:** every profile produced either a receipt or a captured crash signature.
A profile that produced neither is a harness failure, not a finding — R5 fails
on an empty log for exactly this reason.

## Phase 3 — performance (strictly serialized)

Run one profile at a time with the other GPU idle. Two servers under load on one
box contend for CPU and PCIe and will understate throughput.

**Gate:** numbers recorded with the machine fingerprint attached.

## Phase 4 — TP2 arm

1. Stop everything from phases 2 and 3.
2. Run the phase 4 profiles.

## Phase 5 — verdict and teardown

1. Emit the report.
2. Collect the run directory back before anything is destroyed.
3. Destroy the instance.
4. **Gate:** confirm destruction against the provider API. An instance believed
   destroyed is not destroyed. `--rent` fails loudly rather than quietly when
   the provider will not confirm.

## Abort conditions

| condition | action |
| --- | --- |
| topology gate fails three times | abort, report, destroy |
| the box never accepts a login | give it back and re-hunt. The error names attempts, interval and elapsed: many refusals over the whole budget is a box that never got the key, a handful is a budget too short for the venue — raise `--ssh-deadline-minutes` only for the second |
| a boot exceeds its deadline | capture the full log, mark the profile failed, continue |
| a probe hangs past its deadline | capture partial results, continue |
| the instance dies mid-run | report what streamed, with an explicit truncation note |
| the reaper fires | the session is over; whatever streamed is the result |

## Changing a configuration

The procedure and both layer schemas are in
[`configs/README.md`](configs/README.md). Change engine behavior only in the
release's `engine/*.yaml`; change scheduling, environment, or probe assignment
only in `fleet.yaml`. This separation keeps the file named by a measurement
identical to the file vLLM consumed.

An engine file is immutable after `launches.jsonl` has recorded it. Put a later
change in a new release directory, update the parity witness and
[`configs/CATALOG.md`](configs/CATALOG.md), then run the CPU preflight. Validate
the new directory on a box running that exact vLLM release before spending on
the full gate. Local gate commands need `--with pyyaml`; an omitted dependency
is a tooling failure, not a configuration finding.

## Developing the harness itself

Renting per attempt re-downloads sixty gigabytes of weights every time. When
iterating on the gate rather than gating a release, keep **one** box warm and
push fixes onto it:

1. Rent once — under a label of your own, so the watchdog has something to match
   — and arm it so the cost is bounded whatever happens to your shell:

   ```bash
   vastai create instance <OFFER> --image <IMAGE> --disk 200 \
     --label fork-bench-dev --ssh --direct --cancel-unavail
   nohup fork/bench/watchdog.sh fork-bench-dev $$ 9000 \
     >>/tmp/fork-bench-watchdog.log 2>&1 &
   ```

   `$$` is this shell: close it and the box goes with it after the grace period,
   and the cap takes it regardless. Arm it before or after the rental — it reads
   no file and needs no instance id, so there is no window it cannot cover.

2. Push the tree and run one phase at a time:

   ```bash
   rsync -az --delete -e "ssh -p <PORT>" fork/ root@<HOST>:/workspace/bench/fork/
   ssh -p <PORT> root@<HOST> 'cd /workspace/bench && \
     python3 -m fork.bench --tag <TAG> --out run --launcher local --phase 2'
   ```

3. Read the boot logs of profiles that *passed*, not only the ones that failed.
   The worst defect found so far — patch reverts accumulating across profiles —
   was visible only as a log line in a run whose results all looked fine.

Destroy the box the moment you stop needing it, and confirm with
`vastai show instances`. An exit code is not evidence that anything was torn
down.

## After the run

Replace any `derived-from-source` fixture in
[`fixtures/README.md`](fixtures/README.md) with the real boot log this session
captured. Engine stdout carries no identifying data, so the captures can be
committed close to verbatim.
