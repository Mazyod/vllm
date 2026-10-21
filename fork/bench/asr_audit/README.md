# Whisper quality audit

This audit measures `mlmlaziz/bbyn-v2` at revision
`e910dbbc381deab0561e944c7149c3ace41b6bff`. Every request explicitly supplies
`language=en` or `language=ar`. It does not test automatic language detection.

The initial image is the newest published audio image found during discovery,
`openimage/vllm-openai-audio:0.31.0`, pinned to index digest
`sha256:94bc10e9a707bfb32d10933480c087507fef1e0f9e46869542bdd08c96d71843`.
Neither `0.32.0` nor `v0.32.0` resolved in that repository. Results for this
baseline must not be labeled as verification of a 0.32.0 deployment.

## Inputs and measurements

`prepare.py` consumes the English and Arabic validation parquet files from
`google/fleurs`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
(CC-BY-4.0). Their exact hashes are recorded in the generated manifest.
There are 128 individual utterances from 70 sentence IDs per language, four disjoint composites
longer than three minutes per language, shifted copies of those composites,
and one silence probe per language. Long inputs use sentence IDs absent from
the short set and from the other long inputs. They concatenate
utterances with 250 ms gaps; shifted copies prepend one second of silence.
No speech or transcript is truncated. Shifted copies are not independent data.

`run.py server` retains JSON and verbose JSON responses from the actual HTTP
API, with temperature zero and a 444-token completion allowance after the
four-token decoder prompt. The configured decoder context remains 448.
Use a new output directory for each serial, concurrent, or streaming run.

`run.py reference` runs the identical stored FP16 checkpoint using Transformers
with FP32 arithmetic, TF32 disabled, greedy decoding, explicit language, and
no previous-chunk text conditioning. Short audio is decoded both with and
without timestamps. Long audio has three controls: native timestamp-seeking
generation, plus timestamped and plain generation of the exact vLLM chunks.
The native long control uses independent segmentation; the chunk controls
isolate segmentation from decoder differences. All use the checkpoint's
generation configuration, which is saved with the reference results.
Engine-specific suppression and timestamp rules can differ; this is a paired
serving-quality comparison, not a claim of identical decoding algorithms.

`report.py` verifies model/input fingerprints before comparing transcripts
against human text. It records substitutions, deletions, insertions, WER, CER,
per-case transcripts, and paired sentence-cluster bootstrap confidence intervals.
Multiple recordings of the same sentence are resampled as one cluster.
Normalization preserves Arabic letter distinctions, removes diacritics and
punctuation, and does not expand English numbers. Retained text permits other
normalization policies without repeating inference. Silence is inspected
separately, not included in a corpus WER denominator.

## Reproduce

Prepare locally with Python tooling through `uv run --no-project`:

```bash
uv run --no-project --with numpy --with pyarrow --with soundfile \
  python fork/bench/asr_audit/prepare.py --source /path/to/fleurs --out /path/to/fixtures
```

On the watched rental, stage this directory at `/workspace/audit`, the pinned
checkpoint at `/workspace/model`, and fixtures at `/workspace/fixtures`.
Boot from the pinned image and start `vllm serve --config audit/engine.yaml`.
Then use the image's Python environment without changing its serving packages:

```bash
python3 audit/run.py server --fixtures fixtures --model model --out results/serial
# Stop the vLLM process group before using the GPU for the reference.
python3 audit/run.py reference --fixtures fixtures --model model --out results/reference
```

Score collected results locally:

```bash
uv run --no-project --with jiwer==4.0.0 python fork/bench/asr_audit/report.py \
  --fixtures /path/to/fixtures --server /path/to/serial \
  --reference /path/to/reference --out /path/to/quality.json
```

Public read speech does not certify conversational, dialectal, noisy, or
domain-specific accuracy. Training overlap is unknown. A native reference
also makes recognition errors; similar scores are not proof that both
transcripts are correct. Check absolute scores and individual failures.
