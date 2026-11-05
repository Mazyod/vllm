# v0.33.0 — release preparation

Candidate preparation; image build, real-parser validation, GPU validation,
and promotion are pending.

## Source and image identity

- Upstream: [v0.33.0](https://github.com/vllm-project/vllm/releases/tag/v0.33.0),
  `4ba787ec9959e566d3b587a796dff07673324da5`.
- Base: `vllm/vllm:v0.33.0`, manifest digest
  `sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`.
- The linux/amd64 image reports that same upstream commit and retains
  `ENTRYPOINT ["vllm", "serve"]` and `UV_OVERRIDE=/etc/uv-overrides.txt`.
  Upstream also publishes arm64; this fork still builds linux/amd64 only.
- The publication target is now `openimage/vllm-audio`. The source patch
  series remains empty, and the image still adds only `vllm[audio]` plus
  the existing patch-application scaffolding.

## Configuration and validation scope

All nine engine argument sets and fleet environments are carried from
v0.31.0. Gemma retains the composite backend expectation, V1 runner, and TP2
collective workarounds. Manual DeepSeek configurations remain unvalidated.

The CPU dry run reuses the v0.31.0 Gemma captures until v0.33.0 captures are
available. This tests the harness; it is not evidence of engine behavior on
v0.33.0. Historical release records and digest-pinned deployments retain
their original repository names and image identities.

Docker Hub and Buildx manifest inspection confirm that the new base exists.
Local image build and real-parser validation were not run: the host had about
9.6 GB free, less than needed for the roughly 9.7 GB compressed amd64 image
and its extracted layers. No GPU rental or image promotion was performed.

CPU preflight passed with `PREFLIGHT GREEN`: 599 tests passed, nine skipped,
and the complete mock gate passed. Alignment, changed-file pre-commit hooks,
and ShellCheck passed. The fixture replay regression failed before the
historical-capture fallback and passed afterward. Ruff settings include
upstream v0.33.0's additional `UP042` exemption.

The previous validated release remains
`openimage/vllm-openai-audio:v0.31.0`; the legacy repository's `latest`
remains on v0.28.0. Publication of v0.33.0 requires the normal candidate
build and release gate before promotion.
