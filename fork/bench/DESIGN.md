# fork/bench — release gate design

A harness that answers, for each new upstream vLLM release, whether this fork's
patch series is still needed and whether the engine still behaves the way the
image's target configurations require.

Status: in service. `bash fork/bench/preflight.sh` exercises the whole client
path on CPU, and all four phases have run green on rented H100s against
`v0.26.0`. What that first hardware session cost to learn is recorded in
[LESSONS.md](LESSONS.md); read it before changing anything here.

## Why it exists

[FORK.md](../../FORK.md) § Lockstep makes one obligation standing: at every
release, *first* drop what upstream absorbed, *then* rebase what it did not. The
drop check as written is a static ancestry test — `git merge-base --is-ancestor
<merge-commit> <tag>`. That proves the fix is **present**. It does not prove the
**bug is gone**, and it says nothing about the rest of the engine.

Three questions recur at every release, and today all three are answered by
reading diffs:

1. **Is each patch still required?** Ancestry answers this only when upstream
   merged the exact commit we backported. It cannot see a fix that arrived by a
   different route, nor a patch that has quietly become a no-op.
2. **Did a behaviour the image depends on change?** The `v0.26.0` image pins the
   V1 model runner and disables two all-reduce fusions. Each pin is a workaround
   for a specific upstream defect, and each one's necessity is currently
   re-argued from memory rather than measured.
3. **Did anything silently regress?** Speculative decoding falling back to
   disabled, or an attention backend being reselected, are both invisible until
   something downstream notices.

The gate turns each into a probe that produces a receipt.

## Non-goals

Stated so nobody later assumes coverage that is not here:

- **Accuracy and output quality.** Nothing here would catch a quantization bug
  that degrades generation quality while serving normally. B5 sends audio but
  never reads what came back: it asserts the engine survived, not that it
  heard anything.
- **Vision and video.** Audio is touched only by B5, and — since the fleet's
  checkpoints carry no audio tower — only as far as request validation and
  the #50957 crash mode: mixed clip durations arriving together.
- **Architectures the image does not target.**
- **Autonomous triggering.** The gate is run deliberately, by hand.
- **Capacity planning.** Throughput numbers are a trend record, not a sizing
  input.

## Shape

**Venue.** One rented two-GPU Hopper box, on-demand, PCIe-connected. The
deployment target is TP2 without NVLink, and the point of renting is to
reproduce that topology on comparable hardware (H100 class) rather than to test
on a proxy: prove it there, and it holds in production.

The absence of NVLink is a requirement, not a cost compromise: two of the
configurations under test are all-reduce workarounds that only fail on a machine
without it. Offers advertising no NVLink regularly turn out to have it, so
`classify_topology` classifies the pair rather than trusting the listing.

**A pair that turns out to have NVLink is disqualified, not salvaged.** The bug
class these workarounds exist for appears only on hardware that genuinely lacks
the link; renting a linked pair and disabling peer access with
`NCCL_P2P_DISABLE` does *not* reproduce it. The gate used to do exactly that and
treat the result as equivalent, which would have produced a green run that said
nothing about the deployed configuration. It now refuses and tells the operator
to destroy the instance and hunt another offer. No code in `fork/bench` sets
`NCCL_P2P_DISABLE`, and a test enforces that.

**Trigger.** Manual. The decision to spend stays a human one. Everything after
that decision — rent, verify, run, collect, destroy — is automated behind
`--rent`.

**Division of labour.** The operator chooses when to run. The harness owns
everything else: renting, probing, measuring, reporting, and giving the machine
back. This split is load-bearing: a gate whose probes are improvised each run
produces a fresh investigation rather than a comparison, which is the opposite
of what a release gate is for.

**Teardown is a property, not a step.** A rental that outlives its run bills
until someone notices, so four separate things have to fail before that can
happen. The reaper is armed as the first statement after creation and destroys
on a hard cap regardless of what the driver is doing. Teardown confirms with a
read rather than trusting that a destroy call returned cleanly, and retries
once. A sweep on the run's own label catches an instance whose create response
was lost — the one case an instance id cannot cover. And an unconfirmed
teardown raises even on an otherwise clean run.

Selection refuses an interruptible bid outright. Being outbid part-way through
truncates the run and leaves numbers that cannot be compared to anything, which
costs more than the bid saves.

**The box is the image.** A rented instance boots from the image under test, so
there is no docker daemon to hand a container to. On the box the gate runs with
`--launcher local` and starts each engine as a child process. This is why an
ordinary two-GPU rental is enough; requiring a full VM would narrow the offer
pool for no gain.

**Configurations.** The release-scoped source of truth is
[`configs/`](configs/README.md): engine YAML is the file `vllm serve --config`
reads, while `fleet.yaml` assigns that argument set to a phase, environment,
GPU set, replica count, and probes. The [catalog](configs/CATALOG.md) identifies
which files ship, isolate controls and negatives, or run off-gate.

**TP2 is the shipping topology**, so those profiles carry the full receipt,
behavioural, and performance set. Single-GPU profiles isolate patch relevance
or compare two replicas against the same two-GPU budget. Keeping the engine
arguments out of this document prevents a prose copy from becoming a second,
unmeasured configuration.

Budget: one box, roughly 75-90 minutes, per the phase table below.

## Patch relevance: leave-one-out

The obvious design — boot stock, boot patched, compare — **does not work**.

Patch `0002`'s failure mode requires MTP to be running. On stock `v0.26.0` MTP
does not boot at all, because that is exactly what patch `0001` fixes. Testing
`0002` against stock would report a failure for the wrong reason and mark the
patch "still required" no matter what upstream did.

So the negative control is **leave-one-out**: to test whether patch *i* is still
required, run the series with every patch *except i*.

| configuration | `0001` | `0002` |
| --- | --- | --- |
| full series | applied | applied |
| minus-0001 | reverted | applied |
| minus-0002 | applied | reverted |

**No image builds are needed.** Every patch in this series is a pure-Python edit
to files in the installed package — `vllm/v1/spec_decode/llm_base_proposer.py`
for `0001`; `vllm/v1/core/sched/scheduler.py` and
`vllm/v1/structured_output/__init__.py` for `0002`. Leave-one-out therefore
starts from the fork image with the full series applied and runs `patch -R -p1`
on the single patch under test before launching the server. Reverting is
instantaneous, the whole matrix rides on one image pull, and adding a future
patch to the matrix costs nothing.

This is a property of the series worth preserving: a patch that touched compiled
code would forfeit it.

### Verdict table

Each patch yields a 2×2 that feeds FORK.md's drop-then-rebase step directly:

| leave-one-out | full series | verdict | action |
| --- | --- | --- | --- |
| fails | passes | **still required** | carry forward |
| passes | passes | **retired** | delete the patch, its note, and its `series` line |
| fails | fails | **broken on this tag** | rebase before shipping |
| passes | fails | **now harmful** | drop urgently |

Each patch note under `../patches/notes/` carries a "Reproduce" section
describing exactly this check in prose, and the gate makes those sections
executable. (That directory is absent while the series is empty — it comes
back with the next patch. See [`../patches/README.md`](../patches/README.md).)

## Probes

Four kinds. The third is what makes this a *fork* gate rather than a generic
benchmark.

### Receipt probes — assert on the boot log

| id | assertion | applies to |
| --- | --- | --- |
| R1 | `Sharing target model embedding weights` present **and** `Keeping separate` absent | patch `0001` effective |
| R2 | attention backend is `TRITON_ATTN` | V1 configurations |
| R3 | `disable_custom_all_reduce=True`, fusion not explicitly on, and no `CUSTOM`/`FLASHINFER` path left in the dispatch list (captures show `['PYNCCL']` or `['SYMM_MEM','PYNCCL']`, so the exact list is not pinned), no `mnnvl` line | TP2 configurations |
| R4 | MTP detected, draft layers wired, `num_speculative_tokens` as configured | both |
| R5 | no crash markers; `/health` returns 200 within the deadline | all |

These need `VLLM_LOGGING_LEVEL=INFO`; at `WARNING` the engine hides the lines R2
and R3 assert on.

### Behavioural probes — fire requests, classify outcomes

| id | probe | pass condition |
| --- | --- | --- |
| B1 | 100 structured-output requests (50 native, 50 tool mode), each with `enable_thinking` set, a 2048-token completion budget so reasoning finishes before the budget does, and MTP on; count `{{` and `{"{` openers across **content and tool-call arguments**. Runs at TP2, the shipping topology, as well as single-GPU | 0 corrupt, and at least one constrained output actually inspected |
| B2 | 60 guided and tool-calling requests; count HTTP 500 `Failed to advance FSM` | 0 |
| B3 | `vllm:spec_decode_*` counters non-zero; acceptance rate recorded | speculative decoding actually live |
| B4 | a request carrying `thinking_token_budget` | accepted, not rejected |
| B5 | 12 concurrent chat requests carrying `input_audio` clips of **differing durations** (0.2–3.1 s), then `/health` | every request 200 or a clean 400, no 5xx or dropped connection, **and the engine still alive** |

B1's pass condition counts inspections because two of its three legs used to be
missing: it never asked for reasoning, and it read `message.content`, which is
null in tool mode. It scored 0/100 against an engine with patch `0002` reverted,
three times, and that clean result was read as evidence the patch could go. A
structured-output probe has to assemble all of MTP, guided decoding and
reasoning, or a pass means only that it did not look.

B5 asserts survival rather than transcription quality. Upstream #50957 kills
EngineCore on concurrent audio of mixed clip lengths, so the requests in flight
are not the damage — the dead server afterwards is. A clean 400 passes because
the fleet cannot do better: the FP8 Gemma checkpoint ships `audio_config: null`
(the export strips the audio tower), so every audio request is correctly
rejected on every release. Until an audio-tower model joins the fleet, B5
exercises request validation and survival, not audio inference.

### Negative probes — assert a known failure still happens

The leave-one-out boots above, plus the questions `v0.26.0` specifically raises:

| id | configuration | expectation | meaning if it passes instead |
| --- | --- | --- | --- |
| N1 | V2 runner + fp8 kv cache + MTP | crashes in the FlashInfer SM90 sliding-window guard | that defect is fixed; the V1 pin is no longer needed *for the crash* (still needed for `thinking_token_budget`) |
| N2 | V2 + fp8 target kv cache + `speculative_config.kv_cache_dtype` for the draft | unknown; the field is new in `v0.26.0` | a new escape hatch from the same crash |
| N3 | TP2 with both all-reduce workarounds disabled | crashes | the workarounds are retirable |

**Recorded prediction for N1: it still crashes.**
`FlashInferBackend.supports_sliding_window()` returns `True`
(`vllm/v1/attention/backends/flashinfer.py:385`) while the metadata builder still
raises `NotImplementedError` under `current_platform.is_device_capability(90)`
(same file, `:799`). `v0.26.0`'s new per-KV-cache-group backend selection will
therefore select FlashInfer on SM90 and only then hit the guard. Writing the
prediction down is what makes the probe falsifiable.

N3 is worth re-running on this release specifically: `v0.26.0` reworked the Qwen3.5
all-reduce path ("fuse more RMSNorm + all-reduce", "replace MoE all-reduce with
reduce-scatter"), which is the code the workaround disables.

### Performance probes — recorded, never gated

A rented box is different physical hardware every run, so a cross-run comparison
cannot support a pass/fail claim. These produce numbers only.

| id | measurement |
| --- | --- |
| P1 | TTFT p50/p99, concurrency 1, ~3k-token prompt |
| P2 | single-stream decode tokens/s |
| P3 | aggregate throughput at concurrency 1, 8, 32, read from Prometheus |
| P4 | speculative acceptance rate under natural generation |

Written to `baselines/<tag>.json` **with a machine fingerprint**: GPU model and
count, interconnect type from `nvidia-smi topo -m`, driver version, CPU model,
host identifier, image digest, and the engine's own version string. Without the
fingerprint the trend file cannot be honestly interpreted across runs.

Five invariants the harness enforces rather than documents, each guarding a way
this measurement is easy to get wrong:

1. `ignore_eos` with a fixed `max_tokens` — otherwise the probe measures how
   terse the model is, not how fast it generates.
2. TTFT counted on the first token of `content` **or** `reasoning_content`. A
   reasoning parser routes thinking tokens to a different field, so counting only
   `content` measures time-to-finish-thinking and inflates TTFT by orders of
   magnitude.
3. Server-side Prometheus counters, never client wall-clock, which understates
   throughput badly under client CPU contention.
4. Rotated distinct prompts, with prefix caching set to each configuration's
   real value — one repeated prompt against a warm prefix cache measures the
   cache.
5. P4 must **not** set `ignore_eos`: forced filler is unusually predictable and
   inflates the acceptance rate.
6. P1 switches thinking **off**. A reasoning parser may buffer every delta until
   its block closes, and a short token budget never closes one, so leaving it on
   makes TTFT unmeasurable rather than merely large. One model under test
   reported no TTFT at all for a whole run before this was pinned. The
   throughput probes leave it alone, because they read engine counters and
   buffering cannot hide their tokens.

**One TP2 server versus two TP1 replicas.** The same two GPUs, arranged two
ways, answering different questions: a single TP2 server usually wins
single-stream decode, while two TP1 replicas usually win aggregate throughput
at high concurrency. Which matters depends on the traffic, so the gate measures
both as same-box controls minutes apart rather than arguing about it.

A profile therefore runs one *or more* servers, and the measurement treats them
as one: load handed out round robin, counters summed across every replica.
Measuring one replica of two would report half a deployment as the whole. A
profile counts as up only when every replica it asked for is up — load shed
onto the survivors of a partial fleet reads as a throughput win.

## Phases and gates

| phase | what | gate to proceed | GPU time |
| --- | --- | --- | --- |
| 0 | **Static, local, free.** Per patch: is its recorded upstream merge commit an ancestor of the new tag? Does it still apply (`patch -p1 --dry-run`, the command the image build runs)? Scan the release notes for the areas this fork depends on: speculative decoding, sliding-window attention, all-reduce, kv-cache dtype, model runner, structured output. | brief reviewed; decision to spend | none |
| 0.5 | **CPU preflight.** Probe code exercised end to end against the bundled mock server. | preflight green | none |
| 1 | **Provision and host check.** Rent an on-demand offer, arm the reaper, classify the pair, stage both models' weights. `--rent` does all of it. | topology classified PCIe-only. An NVLink pair or an unreadable matrix refuses the run | ~15 min |
| 2 | **Correctness, both GPUs in parallel.** Every leave-one-out and negative boot. Not timing-sensitive, so the parallelism is free. | each boot produced a receipt or a captured crash signature | ~25 min |
| 3 | **Performance, strictly serialized.** Each shipping configuration at TP2, one at a time. | numbers recorded | ~20 min |
| 4 | **TP2 arm — the shipping topology.** Both configurations with the all-reduce workarounds on, carrying the full receipt *and* behavioural probe set (must pass), then N3 with the workarounds off. | every gating probe passed | ~25 min |
| 5 | **Verdict.** Emit the report, collect it back *before* anything is destroyed, then destroy the instance and **verify** destruction against the provider API. | instance list empty | ends |

**Phase 1's topology gate is not optional.** Offers advertising no NVLink
regularly turn out to have it. Detecting that costs about two minutes; skipping
the check invalidates every all-reduce conclusion in the run — and the answer
on detection is to hunt another offer, because forcing peer access off on a
linked pair does not reproduce the failure being tested for.

**Phase 5's ordering is not optional either.** Destroying before collecting
spends the money and throws away the answer, so collection happens inside the
rental's own scope on every path: a clean pass, a failing gate, a run that
never finishes, and a transfer that never lands.

## Layout

```text
fork/bench/
  DESIGN.md           # this file
  RUNBOOK.md          # session protocol: phases, preconditions, abort conditions
  configs/            # release-scoped engine YAML, fleet metadata, and results
  profiles.py         # load one tag's configuration store
  receipts.py         # boot-log parsing and receipt probes
  behaviour.py        # request-level probes
  perf.py             # measurement, and a profile's servers as one fleet
  runner.py           # build one server's command, run a profile's probes
  gate.py             # launchers and phase orchestration
  verdict.py          # emit markdown + JSON, set exit code
  campaign.py         # rent, push, run, collect, destroy
  provision.py        # offer policy, rental lifecycle, reaper, sweep
  vast.py             # one provider's dialect, behind the provider protocol
  remote.py           # ssh and rsync against the rented box
  mock.py             # deterministic OpenAI-compatible server for CPU preflight
  fixtures/           # captured boot logs for crash-signature tests
  baselines/          # per-tag performance trend records
```

`profiles.py` derives profile identity from the engine file and combines it
with the non-engine metadata in `fleet.yaml`. The store's
[`README.md`](configs/README.md) defines how to add or change a configuration.

**Results stream to disk as they are produced.** A run that collects everything
at the end loses everything when the box dies, and a rented box can vanish
mid-run.

`launches.jsonl` is written before each server process starts. One line records
the launch id, profile id, replica, launcher-resolved config path, engine and
fleet SHA-256, argv with token-like Docker environment entries omitted,
selected non-secret environment, image reference, and installed engine version
when known. It survives a boot crash and states which bytes the process was
asked to read.

Every probe line in `results.jsonl` carries the launch ids of the replicas it
observed. `baseline.json` stores `config.fleet` and `config.profiles`, each with
the committed path and SHA-256; `report.md` renders the same fleet identity and
the per-profile engine identities derived from launch receipts rather than by
re-reading the files. The path makes the configuration readable, and the
launch-time digest makes later byte drift visible without rewriting history.

**Exit non-zero** on any patch verdict of *broken* or *now harmful*, or any
probe failing on a gating profile — or on a profile the fleet does not declare
at all, which fails closed because a result nothing declares is a harness bug.
A *retired* verdict exits zero and is reported as an action item — it is good
news that requires a change to the series.

An outcome that contradicts a profile's declared `expect` — a negative arm that
served, or a profile that promised to serve and did not — is **recorded, not
gated**. `report.md` names it in an *Expectation mismatches* section beside the
patch verdicts, carrying the R5 detail behind the finding; the exit code is
unchanged. A first sighting on a new release is a finding to investigate, not a
reason to fail the release.

## Error handling

| failure | response |
| --- | --- |
| no offer meets the requirements | abort before spending, listing each candidate's reasons so the search is widened deliberately |
| topology unreadable | abort and report; neither all-reduce claim can be supported |
| a replica of a fleet never serves | the profile is not measured; a partial fleet reads as a throughput win |
| the gate never finishes | collect what it wrote, exit non-zero, destroy |
| boot exceeds deadline | capture the full log, mark the profile failed, continue |
| probe hangs | per-probe deadline, capture partial results, continue |
| instance dies mid-run | streamed results survive; report partial with an explicit truncation note |
| reaper fires | session ends; whatever streamed is the result |

Safety rails, given a machine that bills by the second:

- A reaper that terminates the instance past a hard deadline regardless of what
  the driver is doing, armed before anything else can fail.
- Teardown **verified** against the provider API, never assumed, and a label
  sweep for what the id cannot reach.
- CPU preflight before any GPU spend. Nearly every failure mode in the client
  path — request construction, streaming, classification, retries, report
  generation — reproduces for free against the mock. The provider itself is a
  protocol, so renting, waiting, collecting and destroying are proven against a
  fake with no spend at all.

## Testing

- `profiles.py` and `verdict.py` are data and pure functions: unit-testable with
  no I/O.
- `probes.py` runs against `mock.py`, a deterministic OpenAI-compatible server
  that can emit corrupted openers, FSM errors, hangs, and honest metrics on
  demand.
- Crash-signature extraction is tested against **captured real boot logs** in
  `fixtures/`, not synthetic ones, since the whole point is to recognise the real
  thing. Every fixture declares its provenance in `fixtures/README.md` as either
  `captured` or `derived-from-source`; the latter is a debt to be replaced with a
  real capture after the first run.
- Hand-writing a fixture is how these tests go tautological, recognising strings
  nobody's engine emits. Two rules prevent it: a **source-linkage test** asserts
  that every message a probe keys on still exists in the vLLM source at the base
  tag, so an upstream rewording fails the suite instead of silently blinding the
  parser; and **no test asserts on a log line number**, because line numbers move
  between releases while messages do not (the SM90 guard is at `flashinfer.py:757`
  in `v0.25.1` and `:799` in `v0.26.0`).
- **The whole gate runs end to end on CPU** via `python -m fork.bench --dry-run`,
  which replays fixtures through the real orchestration against the mock. This is
  what keeps the session from being improvised on a billing machine: provisioning
  and teardown are the driver's job, but everything between launching a profile
  and emitting the verdict is code that has already run green locally.
- The entire client path must pass on CPU before phase 1 begins.
