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
