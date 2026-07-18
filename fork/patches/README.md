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
how the image is built) live in the top-level [`FORK.md`](../../FORK.md); runtime
configuration facts live in [`../docs/deployment-notes.md`](../docs/deployment-notes.md).

## Index

- **0001** — Gemma-4 MTP boot crash (`6400`/`10752` at `pre_projection`).
  Upstream [#47953](https://github.com/vllm-project/vllm/pull/47953) ·
  context: [notes/0001-gemma4-mtp-boot-crash.md](notes/0001-gemma4-mtp-boot-crash.md)
- **0002** — Structured-output `{{` / `{"{` corruption across the reasoning
  boundary. Upstream [#44993](https://github.com/vllm-project/vllm/pull/44993) ·
  context: [notes/0002-structured-output-reasoning-corruption.md](notes/0002-structured-output-reasoning-corruption.md)

## Adding a patch

1. Generate the `.patch` against the pinned base tag (see `FORK.md`; the diff
   paths must be repo-relative `vllm/...` so `patch -p1` applies in the image).
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
| **Upstream status** | Open / merged in vX.Y.Z. Related: #... |
| **Drop this patch when** | <exit criteria — verify with the reproduce, then remove from series> |

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
