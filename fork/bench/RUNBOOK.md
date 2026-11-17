# Release runbook

The recurring artifact is `openimage/vllm:<tag>`, based on the matching upstream
release with only the declared fork patches. Audio is a separate, frozen
`openimage/vllm-audio:0.32.0-cu126-sm70` artifact. Never advance that image as
part of the recurring release. Its recipe and validation contract live under
[`../docker/volta/`](../docker/volta/).

## 1. Prepare the release without paid hardware

Use an isolated Git worktree. Read `FORK.md`, fetch upstream tags, and run:

```bash
git fetch upstream --tags
fork/scripts/new-release.sh <TAG>
```

Review upstream release notes, retire absorbed patches, inspect changed parser
and runtime defaults, and update the new release's engine YAML and fleet.
Do not copy old results into the new release. Preserve historical configuration
bytes; their results certify those bytes only. Keep current release pins in
lockstep and keep the patch export reproducible.

For v0.34.0, remove `VLLM_USE_V2_MODEL_RUNNER=0`: V1 was removed. The primary
release gate is DeepSeek V4.1 Flash TP4. The historical Gemma/Qwen suites remain
available by selecting their release tag. Do not treat old runner fixtures as
proof that the new runner boots.

Review every nondefault flag. Keep only required behavior or a measured gain.
There is no fork-imposed image-count limit. Leave default dtype, tokenizer,
Engram offload, prefix caching, and chunked prefill to upstream. Keep checkpoint
revisions pinned; model identity is part of the evidence.

Run all local Python tooling through `uv run --no-project`:

```bash
bash fork/bench/preflight.sh
fork/scripts/check-alignment.sh
```

**Gate: `PREFLIGHT GREEN`.** CPU mocks prove the harness, not GPU compatibility.
The real image's serve parser must accept every current engine file before
model launches. Review and sign off each atomic commit, open a PR, and require
`alignment` before merging. Never push directly to protected `main`.

## 2. Build a candidate

The `build-vllm.yml` workflow builds from `main` and publishes candidate tags
only. It does not install `vllm[audio]`. The base image is resolved once and
passed by digest. Runtime/parser checks and reverse-patch checks run on CPU.

Record the candidate digest and labels: overlay SHA, upstream release SHA,
patch-export hash, and upstream image digest. Use that exact digest for GPU
validation. Changing any image input requires a new candidate.

## 3. Develop on the cheapest valid hardware

Before searching offers, write the property being proved and choose an anonymous
profile from [`../deploy/HARDWARE_PROFILES.md`](../deploy/HARDWARE_PROFILES.md).
CPU is enough for schemas and orchestration. Use cheaper Hopper hardware for
architecture-only checks. Four high-memory Hopper GPUs are justified for the
actual DeepSeek checkpoint/context fit. Exact Tesla V100 16 GB/SM70 hardware is
required for the frozen image's final compatibility and capacity proof.

Choose **development** or **certification** before create. Development keeps one
watched rental, checkpoints, and compilation caches warm across model/config
failures: keep **one** box warm. A model boot failure is **not** an instance abort. Certification runs fixed, previously working bytes once and collects
and destroys on every outcome. `--rent` is a **certification** controller for historical TP2 profiles; do not use its fixed two-GPU offer policy for TP4 development.

Record hourly rate, hard cap, expected transfer bytes/rate, cache reuse, and
elapsed time in ignored `runs/`. Check the offer's expiry exceeds the watchdog
deadline plus collection margin. Arm the external label watchdog **before**
every paid create and verify its `armed` line:

```bash
# Run in a persistent driver shell; its lifetime owns this development session.
nohup fork/bench/watchdog.sh fork-bench-dev $$ 14400 \
  > runs/watchdog.log 2>&1 &
# Only after "armed": create under exactly fork-bench-dev.
```

Use the provider adapter's `AUTHORIZED_KEYS_REPAIR` onstart command. For
`Permission denied (publickey)`, inspect provider logs first; incorrect key-file
ownership can be repaired without weakening sshd's `StrictModes`.

Verify `nvidia-smi topo -m`, GPU model/memory, driver, host RAM, and disk before
staging. R3's historical PCIe collective regression requires a genuinely unlinked
pair; fabric hardware is not equivalent. DeepSeek TP4 records its actual fabric
and must not be mislabeled as PCIe proof.

Stage pinned checkpoint files once. Reuse Hugging Face and compilation caches.
Do not expose credentials through command lines or logs. Public checkpoints need
no token. Validate the real parser in the candidate environment:

```bash
uv run --no-project --with pyyaml -- python -m fork.bench.config_validation --tag <TAG>
```

A model failure stops its full process group and frees GPU allocations, then
saves logs and restarts on the same rental. Destroy development hardware only
for a hard cap, final session end, unsuitable hardware, provider/security
problem, or unrecoverable host state. Collect incrementally.

## 4. Verify the exact final configuration

Launch the committed YAML through the gate on the candidate image:

```bash
uv run --no-project --with httpx --with pyyaml -- python -m fork.bench \
  --tag <TAG> --image <IMAGE@sha256:DIGEST> --out runs/<TAG> \
  --launcher local --phase 4
```

The local launcher is for a rental booted from the candidate; use the Docker
launcher on a Docker host. Inspect successful boot logs too. Verify model
identity, ordinary text, structured output, tools, reasoning, images (including
more than eight), concurrent traffic, speculative acceptance, and actual token
counts for long-context retrieval. Booting and an advertised context are not
long-context correctness evidence. Record warm throughput and latency, memory
peaks, and cold startup separately. Compare changed performance flags on the
same warm hardware and retain only demonstrated gains.

Extensible KV sizing is not a memory coordinator between independent processes.
A shared-GPU deployment needs explicit budgets, including later allocations by
other residents. Fixed `kv-cache-memory-bytes` opts out of extensible sizing.
Validate co-residency separately; a dedicated-engine pass cannot establish it.

After tuning, run the final committed configuration again. A reviewed public
gate record may contain capabilities, versions, hashes, timings, rates and
limitations. Keep private deployment plans, ownership, addresses, provider ids,
raw rental records and credentials out of commits, CI logs and release notes.

## 5. Collect, destroy, then promote the tested digest

Collect complete logs and receipts before destruction. Confirm provider state
and the watchdog's `done: ... is torn down` line. `NOTHING WATCHED` is not teardown
proof. Record transfer accounting and cache reuse; no inference failures justify
abandoning a live development rental.

Commit the reviewed gate record outside `fork/patches/`; changing any file in
that directory invalidates the candidate fingerprint. After that PR lands,
dispatch `build-vllm.yml` with `promote_from`, `publish_tags`, `gate_record`, and
an explicit `promote_latest` choice. Promotion verifies labels against `main`,
freezes `fork/<tag>`, and retags the tested digest without rebuilding. Verify the
published digest and immutable source tag, then retire the release work branch.

A missing GPU proof, failed probe or unavailable hardware leaves a candidate;
it never becomes a certified release by relabeling old evidence.
