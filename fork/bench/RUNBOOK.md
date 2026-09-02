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

## Before spending: name the question and campaign type

Write one sentence describing the property the rental must prove, then select
the least expensive valid profile from
[`../deploy/HARDWARE_PROFILES.md`](../deploy/HARDWARE_PROFILES.md). Do not rent
H200-class hardware for token counting, controller behavior, HTTP orchestration,
or evidence collection. Use H100-class Hopper when the failure is Hopper-specific
but does not need H200 memory. Rent the large-memory profile only when the real
checkpoint/context fit is the question.

Choose the campaign type before create:

| type | use it when | failure behavior |
| --- | --- | --- |
| **development/tuning** | configuration, runner, memory limits, placement, or probes may change | keep the watched rental and caches; stop model process groups, save logs, push the change, relaunch |
| **certification** | exact committed bytes and fixed probes already passed during development | collect and destroy on every outcome; no tuning inside the run |

`--rent` is a **certification** controller. It is intentionally one-shot. Using
it as a tuning loop destroys the cache after every typo or backend mismatch and
turns one model download into many. The warm-box procedure under
[Developing the harness itself](#developing-the-harness-itself) is mandatory for
development, not an optional optimization.

Every paid run records a cost envelope before create: hourly ceiling, hard cap,
expected model-transfer bytes and rate, and whether a persistent/warm cache is
being reused. No spend begins until CPU preflight is green.

## Phase 0 — static (free, local)

0. Start release preparation with `fork/scripts/new-release.sh <TAG>`. When a
   surviving patch's applicability must be checked by hand, use a pristine
   worktree rather than the overlay checkout:

   ```bash
   git worktree add /tmp/<TAG> <TAG>
   ```

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

If you only have time for one thing, run the reference topology on a rented box:

```bash
# No HF_TOKEN. The fleet's checkpoints are public; set one only for a gated
# checkpoint, and see "The token trap" below before you do.
set -m   # give the run its own process group; see below for why it matters
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag <TAG> --image <IMAGE> --out runs/<TAG> --phase 4 --rent &
driver=$!
nohup fork/bench/watchdog.sh "fork-bench-<TAG>" "$driver" \
  >>/tmp/fork-bench-watchdog.log 2>&1 &
wait "$driver"
```

`$!` is **`uv`'s** pid, not the gate's: `uv run` starts the gate as a child and
waits for it. Measured on 2026-08-29 — kill `uv` and the gate keeps running
while `kill -0 $!` already reports it gone, which is the watchdog's main
trigger firing on a live run. `set -m` puts the job in its own process group so
the watchdog can match the whole run rather than its launcher; it is already
the default in an interactive shell and matters when this is pasted into a
script. The watchdog checks the pid *and* the group, so it is correct either
way, and it refuses to arm at all against a pid that is not running.

The watchdog is not optional. The reaper inside the driver is a thread: it dies
with the driver, and it stands down before teardown runs. On 2026-08-29 a
`--rent` driver was killed before it had written `runs/<TAG>/rental.json`,
leaving a 2xH100 billing at $4.04/hr with nothing watching it — no thread, no
file, no id. `fork/bench/watchdog.sh` keys on the instance **label**, which the
machine carries from the moment it is created, so it needs no file to exist and
watches the rental from its first second. It sweeps that label when the driver
has been gone for the grace period (default five minutes) or when its cap
(default 9900s) elapses, and confirms the sweep against the provider.

**Read its last line, not its exit alone.** An empty sweep means nothing on its
own — a label nobody ever carried sweeps exactly like a box that was torn down
cleanly — so it polls for the label throughout and says which it saw:
`done: <label> is torn down` (exit 0) means it watched a machine and that
machine is gone; `GIVING UP` (exit 1) means it watched one and could not
confirm; `NOTHING WATCHED` (exit 2) means no instance ever carried that label,
so **nothing was guarded and nothing about the account was verified** — check
the label you armed it with against the one the run used.

### The token trap, and the open failure

Until 2026-08-29 this command began `export HF_TOKEN=...`, which is what made
the harness pass `--env` to the provider's create call. `--env` is not a list
of variables: the client's own help calls it "env variables **and port mapping
options**", with the example
`--env '-e TZ=PDT -e XNAME=XX4 -p 22:22 -p 8080:8080'`. It is one docker-run
argument string that **replaces** the default, so passing only `-e` entries
drops `-p 22:22` and leaves the box running with nothing on its SSH port.

The harness no longer passes `--env` at all, and a test holds that argv
env-free. A token now travels in the gate's own ssh invocation, exported
outside the group whose output becomes `gate.log`. Two things follow for you:
the fleet's checkpoints are **public**, so a token buys nothing on an ordinary
gate run — the 2026-08-11 v0.27.1 gate ran without one and its `gate.log`
records the hub "sending unauthenticated requests"; and if you do gate a
private checkpoint, the token is in a command line on the box for the life of
the run, so use one scoped to that repository.

**That hazard is documented, and it is not the failure we hit.** Do not read
this section as an explanation of a box that refuses you. All three preserved
driver logs from 2026-08-29 show `Permission denied (publickey)` — sshd
answering and rejecting the key — including one run that passed no `--env` at
all. Not one shows a refused connection or a timeout, which is what a dropped
port mapping produces. Seven probes went out across two hosts and both endpoint
modes; one reached a shell, and that success is as likely to be noise as
signal. **The key-provisioning failure is unexplained and unfixed.**

So read the error before deciding what to do:

| what ssh says | what it means | what to do |
| --- | --- | --- |
| connection refused, or a timeout, with the instance `running` | nothing is listening on the published port — the documented `--env` hazard | this harness cannot cause it any more; if it happens, capture the create argv before destroying |
| `Permission denied (publickey)` | sshd is up and refusing the account key — the open failure | give the box back and re-hunt. Do **not** raise `--ssh-deadline-minutes`: no budget outlasts a key that is not there |

`--rent` is phases 1 and 5 done for you: it searches its preference list once,
rents one instance, arms the reaper before anything else, pushes this tree onto
the box, runs the gate there, brings the results back, and destroys the instance
— confirming with the provider that it is gone. PCIe is preferred, but the
fallback requirements also admit H100 SXM. The topology gate may refuse that
box; the campaign collects the refusal and destroys it, but does not re-hunt.

Phase 4 itself is TP2 with the all-reduce workarounds on, carrying the full
receipt and behavioural probe set, plus the N3 arm with the workarounds off. It
answers "will this release serve on the reference profile" without the
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
| the box never accepts a login | give it back and re-hunt. Read the signature and follow the table in [The token trap, and the open failure](#the-token-trap-and-the-open-failure) — that table is the only place this decision is written. `Permission denied (publickey)` is the open failure and no budget helps it |
| a boot exceeds its deadline | capture the full log, mark the profile failed, continue |
| a probe hangs past its deadline | capture partial results, continue |
| the instance dies mid-run | report what streamed, with an explicit truncation note |
| the reaper fires | the session is over; whatever streamed is the result |

The table above governs certification. During development, a model boot failure,
OOM, unsupported backend, bad prompt, or probe assertion is **not** an instance
abort. Preserve the evidence, kill only the model process groups, and iterate on
the warm instance. Destroy a development rental only when:

- its hard cap or planned session end is reached;
- topology/hardware cannot answer the question;
- provider, SSH, security, disk, driver, or host state is unrecoverable;
- evidence has been collected and no further in-scope experiment remains.

A transient direct-port refusal is not proof the instance is gone. Read provider
state and retry both polling and collection while the provider reports it live.

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

Renting per attempt re-downloads the weights every time. Current checkpoint sets
can exceed 360 GB and cost more in transfer than the short GPU smoke itself.
When iterating on the gate, an engine configuration, placement, or a probe, keep
**one** box warm and push fixes onto it:

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
   no file and needs no instance id, so the window between creating a box and
   recording its id is covered too. It cannot cover a label that never matches:
   armed against the wrong name it guards nothing, waits out its cap rather
   than declaring a teardown it never did, and exits 2 saying so.

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

During this warm session:

1. stage every pinned model revision once, in parallel;
2. keep Hugging Face and compiler caches on the rental disk (or a bounded
   persistent volume when several sessions are planned);
3. give each launch a new log/result directory, but reuse checkpoint bytes;
4. on failure, stop the complete engine process group and verify GPU memory is
   released before relaunch;
5. collect small evidence after each iteration, not only at session end;
6. run the final committed bytes once more before calling the result certified.

The external watchdog is what makes a warm rental safe. Automatic destruction
after every configuration failure is not a substitute for a watchdog; it is an
expensive loss of useful state.

## Long-context and collection foot guns

- Transformers processors may return a `BatchEncoding`/`Mapping`; `len(value)`
  can count fields rather than token ids. Extract `input_ids`, handle batched
  shape, and assert local count equals the server-reported prompt count.
- A context proof records both actual prompt/completion tokens and the requested
  envelope. “Configured for 128K” is not evidence that a long request ran.
- Test prompt construction locally with tokenizer assets before a GPU rental.
- A detached job may finish and let an SSH-oriented rental container exit before
  the controller's next poll. Write a done marker, hold the container open for a
  bounded collection window, and retry collection while provider state is live.
- Classify the error text before naming OOM. An unsupported attention backend is
  not a memory failure, even if it appears during KV-cache initialization.

## After the run

Replace any `derived-from-source` fixture in
[`fixtures/README.md`](fixtures/README.md) with the real boot log this session
captured. Engine stdout carries no identifying data, so the captures can be
committed close to verbatim.
