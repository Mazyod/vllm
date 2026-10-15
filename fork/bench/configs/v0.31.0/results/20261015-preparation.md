# v0.31.0 — release preparation

Candidate preparation; hardware validation and promotion are pending.

## Source and configuration

- Upstream: [v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0),
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- The source patch series remains empty. The image adds only `vllm[audio]`,
  using upstream's uv installer and inherited dependency overrides.
- All nine engine argument sets retain the v0.30.0 values. Fleet environment
  retains the V1 runner and TP2 FlashInfer collective opt-out.
- Manual DeepSeek configurations remain unvalidated on this release.

## Validation scope and cost envelope

The release-note scan flags speculative decoding, sliding-window attention,
all-reduce, model runner, structured output, and attention backend changes.
These affect the Gemma/Qwen gate and warrant GPU validation. CPU checks cover
configuration parsing, audio decoding/resampling, and the patch export first.

The hardware question is whether the committed Gemma/Qwen configurations pass
correctness, speculative decoding, and TP2 PCIe collective checks on v0.31.0.
Use one watched development rental with the `hopper-pcie-2` capability profile;
H100 memory is sufficient. Preserve its model and compiler caches across
configuration or probe failures, then rerun the final committed bytes.

The planned ceiling is $6/hour with a two-hour hard cap and a $15 total
budget including storage and transfer. Expected checkpoint transfer is about
65 GB, with at least 1 Gbit/s advertised download bandwidth; record actual
bytes and rate. Start with an empty cache and reuse it across launches.
Arm the external label watchdog before create. Collect evidence and confirm
provider teardown at session end.

The B5 probe tests audio request validation and server survival. Its Gemma
checkpoint has no audio tower; CPU image checks separately exercise decoding
and default resampling. DeepSeek and other pending models are outside this
gate's validation scope.

## CPU validation

- CPU preflight: `PREFLIGHT GREEN`, 584 passed and 8 skipped, plus the
  complete mock gate.
- All nine committed engine files pass the real v0.31.0 parser in the upstream
  image on CPU. Their argument values match v0.30.0.
- Overlay Ruff settings follow v0.31.0. Upstream moved Markdown/ShellCheck
  configs under `tools/pre_commit/`; the overlay retains identical bytes at
  its existing root paths and checks the relocated upstream sources.

## Development finding: Gemma backend receipt

The first H100 PCIe launch selected `TRITON_FLASH_ATTN`, so R2 correctly
reported a mismatch with the inherited `TRITON_ATTN` expectation. Upstream
[#53175](https://github.com/vllm-project/vllm/pull/53175), commit
`4fb767ff0a0e0aae9dd2720c47245d0202136dfd`, enables Gemma FP8 KV with
FlashAttention 4 at head dimension 512. On Hopper, the composite backend
routes image masks through Triton and causal queries through FlashAttention.
The TP1 behavioral checks passed, including zero corrupt openers in 100
constrained outputs.

Update the two Gemma R2 expectations to this intentional upstream default.
Keep engine bytes and the V1/collective environment unchanged, retain real
TP1/TP2 boot excerpts for the dry run, then run the complete gate against the
final committed fleet. The initial development run is not a release pass.

Receipt-update validation: `PREFLIGHT GREEN`, 585 passed and 8 skipped, plus
all declared mock probes. Both the previous and current release replay checks
pass. The TP2 capture reports `TRITON_FLASH_ATTN` and `PYNCCL` collectives.
