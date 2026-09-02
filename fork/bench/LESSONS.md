# What the first hardware session cost to learn

Written after the gate's first real runs against `v0.26.0` on rented H100s.
Every entry below is a bug that a full local suite, a green dry run, and a
careful reading did **not** catch. They are recorded because the next person to
extend this harness will be tempted to make the same assumptions.

The pattern is one sentence long: **anything not exercised against the real
thing is a guess, and guesses about external systems are usually wrong.**

<!-- markdownlint-disable MD029 -->
<!-- The numbered rules below are one continuing sequence, cited by number
     from code, commits and the runbook. They are identifiers, not list
     ordinals, so the auto-fix must not renumber them. -->

## The expensive ones

### A live instance read as "gone", and teardown believed it

The provider's client prints a bare object for a live instance and a wrapper
holding `null` once it is destroyed. The adapter understood only the wrapper, so
every live instance looked absent. That told the run its machine had vanished
*and* told teardown a destroy had worked. The run reported success while a GPU
kept billing.

Two lessons. Teardown must confirm against the provider and treat silence as
failure, never as success — it now raises rather than returns. And every
response shape must be captured from a real call, not imagined: all three shapes
are now fixtures in `tests/test_vast.py`.

### The client stopped to ask a question nobody could answer

`destroy` prompts `[y/N]` by default and aborts when stdin is closed. Combined
with the bug above, the abort was invisible. Every external command now runs
with stdin closed, so anything that stops to ask sees end-of-input instead of
hanging with a rental open behind it.

### Reverts accumulated, and the matrix silently tested the wrong engine

Reverting a patch edits the installed package in place. A container launcher
gets a clean filesystem per profile; a rented box has **one** installation
shared by all of them. So the revert done for `gemma-minus-0001` was still in
effect for `gemma-minus-0002`, and would have been for every later phase. The
performance numbers and the shipping-topology verdict would have described
unpatched code while reporting themselves green.

This is the one that would have produced a confidently wrong release decision.
`patch-state.sh` now restates the entire series before every launch, reached
from whatever state the box is in, and fails closed.

The tell was in a boot log: the embedding-width guard firing in a profile that
never touched that patch. **Read the logs of a run you believe is healthy.**

### One probe raised and took two phases with it

An engine died mid-probe, the exception propagated, and the gate exited —
discarding phases 3 and 4, the results already computed for that profile, and
the engine's dying words. Three separate defects wearing one costume.

A probe that raises is now a failed result. Results are yielded as they are
produced. The saved log is written *after* the probes, from the list the drain
thread is still filling, because what an engine says as it dies is the only
evidence of why it died.

## The quieter ones

- **Offer fields are not query fields.** The search grammar says `verified` and
  `gpu_name=H100_PCIE`; the response says `verification` and `H100 PCIE`. Mapped
  by eye, the gate rejected every offer and would have rented nothing.
- **A redirect binds to the last command of an `&&` list, not the list.** Every
  line the staging step printed went to `/dev/null`, so a failed download would
  have left an exit code and no reason for it.
- **rsync creates the last path component, not the ones above it.** The push
  failed a minute after the meter started.
- **`SIGTERM` to the parent leaves the workers.** They hold GPU memory, and the
  next profile fails to allocate for a reason that says nothing about the
  release. Each server now gets its own process group and is killed if it will
  not stop.
- **A probe expectation belongs to the model, not the fork.** `R2` asserted
  `TRITON_ATTN` for everything and reported a healthy hybrid-model engine as a
  release blocker.
- **An engine default is not a configuration.** Qwen inherited `max_num_seqs`
  1024, which exceeds its Mamba cache blocks on one GPU and failed the boot.
  Pinning it also keeps throughput numbers comparable across releases that
  change the default.
- **A reasoning parser can buffer every delta until its block closes.** With a
  short token budget it never closes, so TTFT is not slow — it is unmeasurable.
  One model reported no TTFT for an entire run, and it read as a pass. Probes
  now say `NO MEASUREMENT` rather than reporting a null as a reading.

## Rules that came out of it

1. **Confirm, never assume, anything a remote system tells you.** Especially a
   success.
2. **Check the account, not the exit code.** The run that "passed" left a
   machine running.
3. **Capture real payloads as fixtures.** Every response shape, every boot log.
   No pattern in this directory is written from memory.
4. **State must be restated, not inherited.** Any step that mutates a shared
   installation has to put it back, from wherever it is.
5. **Failures are data.** A run that stops at the first problem tells you about
   one problem; a run that records and continues tells you about all of them.
6. **A missing measurement must look different from a good one.** Nulls that
   render like readings hide exactly the runs worth investigating.
7. **Iterate on one warm box.** Renting per attempt re-downloads sixty
   gigabytes; keeping one and pushing fixes turns a twenty-minute cycle into a
   three-minute one. Arm a detached reaper so the cost is bounded regardless.

## The one production paid for (2026-08-06)

### A boot-only probe retired the patch whose bug had moved past boot

The gate's first run declared patch `0001` retirable: reverted from a
verified-clean state, the engine served anyway. On 2026-08-06 the unpatched
`:latest` crash-looped a production Gemma-4-31B + MTP deployment with patch
`0001`'s exact signature (`[s, 6400] X [10752, 512]` at `pre_projection`).

Three things had to line up, and all three did:

1. **The failure model was stale.** "100% reproducible boot crash" was
   validated on `v0.25.1` and carried to `v0.26.0` by static reasoning
   ("the guard is byte-identical"). On `v0.26.0` the boot dummy-run feeds the
   drafter a preallocated `inputs_embeds` buffer and never exercises its own
   embedding lookup — the code the width guard breaks — so the crash moved to
   the **first scheduled speculative-decode step**. `gemma-minus-0001` carried
   only the boot receipt `R5`, so the reverted engine was never sent a single
   request. It booted; the probe passed; the verdict said retire.
2. **The contradicting datapoint was retracted, not pursued.** In the
   contaminated run, an engine missing `0001` died the moment a request had
   "spec-decode tokens scheduled". That was the production failure mode,
   observed on the box, hours before the verdict. It arrived tangled with the
   revert-accumulation bug, was retracted as unattributable, and the open
   question was never re-run in isolation.
3. **The dynamic result was allowed to outvote the static one.** Ancestry said
   `#47953` is not in `v0.26.0`; the probe said the engine is fine without it.
   That disagreement was read as the harness proving the ancestry check
   pessimistic — celebrated, even — when it was actually the probe confessing
   blindness. A probe that passes can be blind; ancestry cannot be wrong about
   absence.

Amplifier: the retirement push emptied `series`, which is itself a build
trigger, so the never-gated unpatched rebuild republished the certified
`v0.26.0` tag and took `:latest` with it.

## Rules added after the misretirement

8. **Retirement needs both witnesses.** A leave-one-out pass retires a patch
   only when `upstream.map` ancestry independently confirms the fix is in the
   tag. Probe-passes-but-fix-absent renders "probe blind"; ancestry-unknown
   renders "retirement unconfirmed". Both keep the patch
   (`verdict.patch_verdict`).
9. **A reverted engine must take traffic.** Every leave-one-out profile
   carries a B-probe; a boot receipt alone cannot see a failure that fires on
   the first real request (`test_static` enforces this).
10. **A revert must leave a receipt in the engine's own log.** `R6` checks the
    profile's `expect_boot_evidence` against what the booted engine actually
    logged, so a revert that never reached the running code fails the arm
    instead of informing a verdict.
11. **Candidates are not releases.** Push-triggered image builds publish under
    `<tag>-cand-<sha>` and never move `:latest` or republish a base tag;
    promotion is a deliberate dispatch after a gate run.

## The venue assumption that was never true (2026-08-06)

### Forcing PCIe on an NVLink box is not the same machine

The gate treated an NVLink pair as salvageable: detect the link, set
`NCCL_P2P_DISABLE=1`, call the all-reduce path equivalent, and keep the offer
rather than re-hunting. That reasoning is wrong, and the operator confirmed it
from production experience — **the bug class the TP2 workarounds exist for only
appears on hardware that genuinely lacks the link.** Disabling peer access on a
linked pair does not bring it back.

So the "not wasted" optimisation was a way to spend an hour and produce a green
run about a machine nobody deploys on. Nothing caught it, because the run would
have looked exactly like a good one: same probes, same receipts, same report.

An NVLink pair is now disqualifying. The gate refuses and says why; the
campaign destroys its single rental, and the operator re-hunts in a new run.
`NCCL_P2P_DISABLE` appears nowhere in `fork/bench`, and a test asserts it stays
that way — the convenience is easy to re-invent precisely because it sounds
thrifty.

The July run that this correction post-dates was, by luck, on a genuinely
PCIe-only pair (`interconnect: PXB`), so its all-reduce conclusions stand.

12. **Reproduce the topology, do not simulate it.** A configuration flag that
    approximates hardware is not the hardware. If the venue cannot be matched,
    refuse the run rather than substituting a proxy and reporting a verdict.

## The probes that had never met hardware (2026-08-11)

### A probe that has never touched a real engine is a hypothesis

Hardened B1 and B5 both landed on 2026-08-06 — *after* that day's re-gate —
so the v0.27.1 gate was their first contact with a live engine, and both
failed for reasons that had nothing to do with the release. B1 starved:
reasoning consumed its whole 256-token completion budget, so it inspected
0/100 constrained outputs (the payload it exists to inspect only begins after
`</think>`). B5 was aimed at a capability the fleet does not have: the gate's
Gemma checkpoint ships `audio_config: null` — the FP8 export strips the audio
tower — so every `input_audio` request 400s on every release, correctly.

The dry run caught neither, because the mock answers whatever shape the probe
hopes for. Preflight proves orchestration, not aim.

13. **First-run a new probe on hardware before trusting its verdict.** Until a
    probe has produced its expected pass AND its expected failure against a
    real engine at least once, treat what it says about a release as a claim
    about the probe. Instrument probes so a null measurement explains itself
    (B1 now tallies response shapes; B5 records rejection bodies).
14. **A configuration that is not the file the engine read is a claim, not a
    record.** Launch from the committed file and carry its digest into the
    result.

## The root cause that was named twice and held neither time (2026-08-29)

### What was actually observed

Seven ssh probes went out across two hosts and both endpoint modes. One reached
a shell. Every preserved driver log — including a run with `HF_TOKEN` unset and
therefore no `--env` anywhere — records the same signature:

```text
Permission denied (publickey).
```

Not one records a refused connection or a timeout. sshd was up on every box and
refused the account key. **That failure is unexplained and unfixed.** It is the
only thing the day established, and the single success is as likely to be noise
as signal.

### The first diagnosis: waiting longer

The first fix was a bigger `wait_for_ssh` budget, because `Permission denied`
after a fresh create reads like key propagation lag. It is a reasonable
hypothesis and it was wrong: no budget outlasts a key that is not there. The
retry is still correct for genuine lag and is kept, but it never addressed this.

### The second diagnosis: `--env`

`vast.py` passed `--env "-e HF_TOKEN=..."` to the create call whenever
`spec.env` was non-empty, which was exactly when the operator had `HF_TOKEN`
exported — which this runbook's own canonical command instructed. And `--env`
is genuinely a trap: the client's help calls it "env variables **and port
mapping options**", with the example
`--env '-e TZ=PDT -e XNAME=XX4 -p 22:22 -p 8080:8080'`. It is one docker-run
argument string that replaces the default, so passing only `-e` entries drops
`-p 22:22` and the box runs with no published SSH port.

An A/B appeared to confirm it — one probe without `--env` connected, one with
it did not — and it was written up as the root cause. It is not. The logs
disagree in the way that settles it: a dropped port mapping cannot produce
`Permission denied (publickey)`, because there would be no sshd to answer. Both
arms had failed for the same unexplained reason and the one success was noise;
the A/B was confounded and never had the power to indict anything.

What survives is worth keeping on its own terms. `--env` is a documented way to
lose a box, the harness no longer passes it, `InstanceSpec` no longer has an
`env` field for anyone to reach for, and the token travels over ssh instead —
out of instance creation entirely, and outside the group whose output becomes
`gate.log`. That is one hazard closed on documented grounds. It is not a
diagnosis.

### What the discriminator is

- **Connection refused, or a timeout, with the instance `running`** — nothing
  listening on the published port. The `--env` hazard looks like this.
- **`Permission denied (publickey)`** — sshd is up and rejecting the key. This
  is the open failure, and every observed one was this.

Reading the signature first is the whole difference between the two, and it is
what neither diagnosis did.

### The amplifier

The one gate that had ever worked, v0.27.1 on 2026-08-11, ran with `HF_TOKEN`
unset — its `gate.log` still records the hub "sending unauthenticated
requests". So the green run had been exercising the no-`--env` path while the
documented procedure chose the other one, and nobody had evidence about the
path the runbook actually told operators to take.

15. **A creation flag that replaces rather than adds is a trap, and the docs
    can be the thing that springs it.** Before passing an option to a provider,
    read what the option is documented to contain — this one carried the port
    mappings — and A/B the option itself, not the thing being configured with
    it.
16. **A run that passes because a variable happened to be unset has not been
    tested.** When a documented step changes which code path executes, the
    undocumented path is the one nobody has evidence about.
17. **Name the signature before naming the cause.** Two root causes were
    declared for this failure — propagation lag, then `--env` — and the error
    text disproved both: `Permission denied (publickey)` is sshd answering, so
    neither a longer wait nor a missing port mapping could produce it. Each was
    reached by reasoning from a mechanism that *could* explain a symptom nobody
    had read closely, and the second was confirmed by an A/B of two runs, which
    is a sample that cannot distinguish a cause from noise. A fix that stands
    on documented behaviour is worth shipping; it is still not a diagnosis, and
    an investigation stays open until the observed signature is accounted for.

## The $82 smoke test (2026-09-02)

### A certification controller was used as a development loop

The goal was one co-resident GLM+Gemma proof. Five paid rentals were created,
each staging roughly 363 GB, even though the final simultaneous inference took
28 seconds. Approximate total: $46 of instance time and $36 of repeated network
transfer. The successful run itself was about $13 of instance time plus one
transfer. Most of the difference was avoidable.

The root mistake was workflow selection. `rent(...)` is designed for release
certification: enter one rental scope, run immutable probes, collect, destroy in
`finally`. That safety property became destructive when configuration, runner,
and probe code were still changing. Each ordinary development failure deleted a
valuable checkpoint/compiler cache and forced the next attempt to start cold.

The repository already said “iterate on one warm box.” It was treated as advice
instead of a precondition. A detached hard-cap watchdog would have bounded cost
without destroying the useful state after each failed model process.

### The failure chain

1. Gemma selected a V2 FlashInfer draft path that explicitly refuses
   sliding-window attention on SM90. It was described initially as memory
   pressure even though the exception was `NotImplementedError`, not OOM.
2. A transient refusal on a direct SSH port was treated as instance death. The
   provider still reported the rental live; the controller destroyed it anyway.
3. The long-prompt sizer called `len()` on Transformers `BatchEncoding`. That
   counted fields (`input_ids`, `attention_mask`) rather than tokens, so the
   first proof stopped short of its context threshold.
4. Replacing `dict` with an exact-type check did not fix it: `BatchEncoding` is
   dictionary-like, not necessarily a `dict`. The doubling search kept building
   larger strings until host RAM made the bug obvious. The correct contract is
   `Mapping`, extract `input_ids`, then interpret batched shape.
5. A detached proof wrote its result and exited between ten-second polls. The
   SSH-oriented container stopped before rsync, so a successful result was lost.
   A done marker without a bounded post-result hold is still a race.
6. GPU placement was treated as exclusive territory: four devices “consumed” by
   one model plus two by another. Multi-model packing should first overlap TP
   groups within one memory budget; adding devices is a conclusion, not a
   starting assumption.

### What a clean run should look like

Use CPU/mock for prompt sizing, `BatchEncoding`, controller state, and evidence
format. Use the cheapest GPU that preserves a backend or lifecycle question.
Stage pinned checkpoints once on one watched development rental, push changes,
and restart only engine process groups. When the configuration and probes pass,
launch the exact committed bytes once through the certification controller.

For the four-GPU packing initiative, test `hopper-pcie-4-large` first and a
fabric-connected four-GPU fallback only when necessary. The profile describes a
capability; it must not be associated with private ownership or location.

18. **Choose the least expensive venue that preserves the property.** Stronger
    hardware can hide the memory/topology failure and always costs more.
19. **Declare development or certification before create.** If arguments or
    probes may change, a one-shot certification campaign is premature.
20. **A model failure is not a rental failure.** Save the log, kill the process
    group, verify memory release, and relaunch on the warm development rental.
21. **Checkpoint and compiler caches are paid assets.** Destroying them between
    iterations must be an explicit decision, not a `finally` side effect.
22. **GPU placement may overlap.** Prove packing and explicit memory policies
    before summing models' isolated GPU counts.
23. **Token counts need two witnesses.** Extract `input_ids` through the
    `Mapping` interface and require local count to equal server usage.
24. **Provider state outranks one failed connection.** Retry polling and
    collection while the provider reports the instance live.
25. **A result needs a collection window.** Write the done marker, hold the
    container open for a bounded interval, collect, then tear down.
26. **Hardware profiles are anonymous capabilities.** Never commit ownership,
    private location, host/offer/instance ids, IPs, endpoints, or GPU UUIDs.
27. **Account for transfer separately from GPU time.** Record bytes and rate;
    repeated checkpoint download can exceed the smoke's compute cost.
