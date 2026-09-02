# Agent Instructions for the vLLM Fork

This repository is the fork overlay, not an upstream source checkout.
It intentionally contains only fork-owned release infrastructure.

Read `FORK.md` before making changes.

Never edit upstream code here; upstream code is not present on `main`.
Patches are commits on `release/<tag>`, exported into `fork/patches/`.
Each release must remain exactly its upstream tag plus those commits.

Run `fork/bench/preflight.sh` before submitting changes.
Run Python tooling through `uv run --no-project`.
Keep the overlay minimal, auditable, and generated files reproducible.
Follow the patch contract and release workflow documented in `FORK.md`.

Hardware work follows the cost and privacy policy in
`fork/deploy/HARDWARE_PROFILES.md` and `fork/bench/RUNBOOK.md`:

- Use the least expensive venue that preserves the property being tested. CPU,
  mock, or a cheaper GPU is the default; H200-class hardware is the final proof
  venue only when memory capacity or that exact architecture is load-bearing.
- Decide whether a rental is **development** or **certification** before create.
  Development keeps one watched rental and model cache warm across configuration
  failures; certification is the one-shot collect-and-destroy workflow.
- A model/config/probe failure stops model processes, not the development
  rental. Destroy only for a hard cap, provider/security problem, unsuitable
  hardware, unrecoverable host state, or the final end of the session.
- Describe hardware with anonymous capability profiles. Never infer or record
  ownership, private location, provider host ids, IPs, or an association between
  a profile and a person or organization.
- Arm an external label watchdog before every paid create. Record elapsed time,
  hourly rate, transfer bytes/rate, and whether caches were reused.

## Commands

- `bash fork/bench/preflight.sh` - unit suite + dry-run gate; must print PREFLIGHT GREEN.
- `uv run --no-project --with pytest --with httpx --with pyyaml -- pytest fork/bench/tests -q` - the suite alone.
- `fork/scripts/check-alignment.sh` - the CI gate on `main`; run before opening a PR.
- `uv run --no-project --with shellcheck-py -- shellcheck fork/scripts/*.sh fork/bench/*.sh` - shellcheck without a system install.
- `.venv/bin/pre-commit run --files <paths>` - pre-commit lives in `.venv`, not on PATH.
- `git fetch upstream --tags && git worktree add /tmp/<tag> <tag>` - upstream source is never on `main`; read it from a worktree of the tag.
- `gh <cmd> -R Mazyod/vllm` - plain `gh` can resolve to upstream `vllm-project/vllm` in a fresh clone.

## Conventions

- Commit with `git commit -s`; subjects `[fork] ...` on `main`, `[fork-patch] ...` on `release/<tag>` (body sections and trailers per `FORK.md`).
- `main` is ruleset-protected: changes land by PR and the `alignment` check must pass.
- Script tests build throwaway repos with `fork/bench/tests/gitfixtures.py`; never run `fork/scripts/migrate-to-overlay-main.sh` without `--dry-run`.
- vast.ai `Permission denied (publickey)` with the key present: read `vastai logs <id>` first — `bad ownership or modes` is the daemon-owned key file; the gate's `--onstart-cmd` repair (`fork/bench/vast.py`) handles it.
- For iterative hardware work, do not use a one-shot rental campaign until the
  exact configuration and probes have already passed on the warm development
  rental. Repeatedly downloading large checkpoints is a harness failure, not an
  acceptable tuning loop.
