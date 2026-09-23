# v0.30.0 — release preparation

Candidate preparation only; hardware validation and promotion are pending.
The last validated image is v0.29.0; `latest` remains on v0.28.0.

## Source and configuration

- Upstream: [v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0),
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- No source patches or dependency pins. The image adds `vllm[audio]` with upstream's uv installer and inherited dependency overrides.
- All nine engine configurations retain v0.29.0 argument values. Fleet
  environment retains the V1 runner and the TP2 FlashInfer collective opt-out.
- Manual DeepSeek configurations remain unvalidated on this release.

## Validation scope

The release-note scan flags speculative decoding, sliding-window attention,
all-reduce, model runner, structured output, and attention backend changes.
Upstream also changes the default audio resampler to torchaudio (#52598),
adds selectable audio decoding (#51826, #55642), and changes Gemma audio
profiling (#56721). CPU checks must exercise the audio dependencies as well
as the release's real configuration parser.

Upstream #50439 removes the SM90 FlashInfer sliding-window boot guard.
The historical crash fixture is retained, while its obsolete current-source
assertion is retired. The non-gating V2 diagnostic retains its historical
expectation pending hardware evidence; the shipping V1 configuration is unchanged.

GPU validation is warranted for the committed Gemma/Qwen speculative decoding
and TP2 PCIe collective configurations. Use one watched development rental
with the `hopper-pcie-2` capability profile and retain checkpoint caches
across retries. H200 memory is not required for this gate.

The existing B5 probe tests audio request validation and server survival;
the gate's Gemma FP8 checkpoint has no audio tower, so B5 does not certify
successful audio transcription.

## CPU checks and release status

- CPU preflight: `PREFLIGHT GREEN`, 575 passed and 7 skipped, plus the full
  mock gate. All nine engine configurations pass the upstream v0.30.0 parser.
- Upstream Ruff settings now include `INP`; the overlay follows that setting.
- The process launcher now waits after SIGKILL, reaping the child before
  clearing its process list. The existing shutdown regression check passes.
- The first local image build exposed pip downgrading upstream NCCL 2.30.7
  to torch's 2.29.7 dependency. Upstream explicitly requires NCCL >=2.30.4
  for DeepEPv2 and provides `/etc/uv-overrides.txt` through `UV_OVERRIDE`.
  Using upstream's `uv pip install` honors that override; its dry run adds
  only av, scipy, soundfile, and soxr. The corrected local build
  confirms those are the only added distributions, every pre-existing
  distribution keeps its version, and NCCL remains 2.30.7. No fork-owned
  NCCL pin is introduced.
- Image CI now decodes an 8 kHz WAV through auto, soundfile, and PyAV, then
  resamples to 16 kHz with the default torchaudio path on CPU. That exact
  workflow check passed against the locally built image.
- The provider search found no suitable two-to-four GPU H100 PCIe offer.
  Available fabric-connected Hopper offers cannot certify PCIe collectives.
  No rental was created and GPU spend is zero. Hardware gating and versioned
  promotion remain pending; this record is not a gate pass.
