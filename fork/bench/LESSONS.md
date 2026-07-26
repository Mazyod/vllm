# What the first hardware session cost to learn

Written after the gate's first real runs against `v0.26.0` on rented H100s.
Every entry below is a bug that a full local suite, a green dry run, and a
careful reading did **not** catch. They are recorded because the next person to
extend this harness will be tempted to make the same assumptions.

The pattern is one sentence long: **anything not exercised against the real
thing is a guess, and guesses about external systems are usually wrong.**

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
