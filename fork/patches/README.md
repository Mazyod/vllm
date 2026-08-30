# Fork patches — filing and context

This directory is the fork's delta and the **documented context** behind it. Every
patch is filed with enough detail to answer, months later: what bug is this, why
does it hurt us, why does the fix work, how do I reproduce it to check whether the
patch is still relevant, and when do we drop it.

> **Public repository.** These notes are public. Never include private,
> confidential, or personally identifiable information — no customer/organization
> names, no internal hostnames, tokens, IPs, credentials, private URLs, or people.
> Keep model names, hardware classes (e.g. SM90/H100), commands, and upstream
> issue/PR links; those are what make a note useful and are safe to publish.

## Layout

```text
fork/patches/
├── README.md                 # this file — filing convention + index
├── series                    # apply order; one .patch filename per line
├── 000N-<slug>-prNNNNN.patch # the diffs (generated against the pinned tag)
└── notes/
    └── 000N-<slug>.md        # one context doc per patch (the "why")
```

The `.patch` files and `series` are the mechanical inputs consumed by
`../docker/apply-patches.sh`. The `notes/` docs are for humans and do not affect
the build. Operational mechanics (patch model, lockstep with upstream releases,
how the image is built) live in the top-level [`FORK.md`](../../FORK.md).

## Index

The series is **empty** as of `v0.27.1`: the release absorbed every patch the
fork carried, and both retirement witnesses agreed on 2026-08-11 — ancestry
verified against the tag, and the release gate passed every probe firing real
traffic through the patchless candidate (baseline:
`fork/bench/baselines/v0.27.1-20260811.json`). See `fork/bench/LESSONS.md`
rules 8–11 for why one witness is not enough. The empty series held at
`v0.28.0`: the 2026-08-30 gate passed every gating probe on the patchless
candidate (baseline: `fork/bench/baselines/v0.28.0-20260830.json`, record:
`fork/bench/configs/v0.28.0/results/20260830-attempt4.md`).

Retired at `v0.27.1` (patch files and notes live in git history at the
`fork/bump-v0.27.1` merge):

- **0001** — Gemma-4 MTP boot crash. Upstream
  [#47953](https://github.com/vllm-project/vllm/pull/47953), absorbed as
  `b2b8f679d0`. *Wrongly retired once at v0.26.0 by a boot-blind leave-one-out
  verdict; restored 2026-08-06 after the unpatched image crash-looped in
  production.*
- **0002** — Structured-output `{{` / `{"{` corruption across the reasoning
  boundary. Upstream [#44993](https://github.com/vllm-project/vllm/pull/44993),
  absorbed as `0416dab275` (named in the v0.27.0 release notes).
- **0003** — DeepSeek-V4 DSpark/FlashMLA sparse-prefill crash under
  concurrency. Upstream
  [#49302](https://github.com/vllm-project/vllm/pull/49302), absorbed as
  `de6ec294ef07`.

## Adding a patch

A patch is the fork's most expensive kind of divergence, so it has to clear the
charter's bar first (FORK.md § Charter):

- **One goal, traceable upstream.** Exactly one upstream PR, or one
  narrowly-stated fork need — never a bundle. If you cannot name the single
  thing it fixes, it is not ready.
- **Smallest possible diff.** Only the files that goal requires. No drive-by
  edits, no local "improvements" to vLLM, no test files that the runtime image
  does not carry.
- **A stated exit.** Record the upstream commit that will retire it, so the next
  release can drop it mechanically. A patch with no exit criterion does not go in.

Then:

1. Generate the `.patch` against the pinned base tag (see `FORK.md`; the diff
   paths must be repo-relative `vllm/...` so `patch -p1` applies in the image).
   If the upstream PR has already merged, generate from the **merged commit** so
   the fork ships what upstream shipped.
2. Add its filename to `series` in apply order.
3. Write `notes/000N-<slug>.md` using the template below.
4. Update the index table above.

## Note template

Copy this into `notes/000N-<slug>.md`:

```markdown
# Patch 000N — <short title>

| | |
|---|---|
| **Patch file** | [`../000N-<slug>-prNNNNN.patch`](../000N-<slug>-prNNNNN.patch) |
| **Upstream PR** | <url> |
| **Files touched** | `path/to/file.py` |
| **Applied on** | `vX.Y.Z` |
| **Upstream status** | Open, or **Merged** <date> as `<commit>` — say whether that commit is in a release yet. Related: #... |
| **Drop this patch when** | `git merge-base --is-ancestor <merge-commit> <tag>` succeeds for the tag we rebase onto — then delete the patch, this note and its `series` line. |

## Why it hurts us (impact)
<symptom, blast radius, error text/traceback, how often>

## Root cause (why the fix works)
<the mechanism; what upstream change introduced it, if any; why the fix is correct>

## Reproduce (portable)
<a self-contained command/steps; what stock vs patched output looks like;
a one-line "relevance check": if stock now behaves, the fix landed — drop it>

## Validation (point-in-time)
<what was tested, on what hardware, with dated results>

## Ruled out (do not re-explore)
<dead-end hypotheses, so nobody burns time on them again>
```
