# bbyn-v2 accuracy audit — 2026-10-21

> Archived historical evidence. The effort is closed; local model/image archives
> and raw runs were intentionally removed. See [closeout notes](../README.md).

**Ordinary JSON closely matched the independent FP32 reference. Timestamped
responses failed and must not be treated as verified for this combination.**
Every inference request explicitly supplied `en` or `ar`; automatic language
detection was not tested.

## Exact object tested

- Model: `mlmlaziz/bbyn-v2`, revision
  `e910dbbc381deab0561e944c7149c3ace41b6bff`.
- Weight file: 3,087,130,976 bytes, SHA256
  `79d6f3b3717e35761254622c0d9f8347f5385cbaef5d0f1c34d0fa220d6ac330`.
  The downloaded hash matched the pinned Hub object.
- Image: `openimage/vllm-openai-audio:0.31.0`, index digest
  `sha256:94bc10e9a707bfb32d10933480c087507fef1e0f9e46869542bdd08c96d71843`;
  AMD64 manifest
  `sha256:2ffdd20232ffbbfbcf1e803820b70c68518ac13aa9c4788d5f693f4f3e80c6de`.
- The requested `0.32.0` and `v0.32.0` tags did not resolve in that repository,
  including a recheck after inference. This is not a 0.32.0 image certification.
- Anonymous `cuda-budget-1` profile: one NVIDIA L4, 23,034 MiB, driver 615.71.09.
  vLLM 0.31.0, Torch 2.13.0+cu130, Transformers 5.17.0, soundfile 0.14.0.
- [Engine configuration](engine.yaml): FP16, full 448-token decoder context,
  eight sequences, default attention/compilation behavior. No source patch,
  serving-package replacement, custom inference plugin, or altered audio split policy.
  Its SHA256 is
  `0153f7fe025b51deb5f514148da780115f13fe6cd856350dcff72fc278361110`.

The tokenizer JSON, vocabulary, merges and audio preprocessing configuration
matched stock Whisper large-v3. The model loaded in the older Transformers
version shipped by the image despite being saved with 5.20.2. The installed
Whisper model, audio splitter, speech serving and transcription protocol source
files matched the upstream v0.31.0 tag byte-for-byte. Vast's normal SSH bootstrap
was added around the base image; its serving Python packages were unchanged.

## Workload and reference

The [runner documentation](https://github.com/Mazyod/vllm/blob/5c945288ad7bb880fb65d60734a9ea9f1e794a77/fork/bench/asr_audit/README.md) specifies the pinned FLEURS inputs and
normalization. There were 128 distinct short recordings per language, covering
70 sentence IDs each; confidence intervals cluster repeated sentences. These
provided 20.40 minutes of English and 22.71 minutes of Arabic speech.

Eight further recordings longer than three minutes concatenate disjoint held-out
sentence IDs. Each also has a copy with one second of leading silence to move
chunk boundaries. Two silence probes bring the manifest to 274 cases. These
are public read-speech fixtures, not natural long conversations or private
production recordings. No speech, transcript, or model context was shortened.

The independent reference loaded the identical stored FP16 weights and executed
them in FP32 with TF32 disabled. Both paths used explicit language and greedy
decoding. Reference runs covered the same chunks and an additional native
Whisper timestamp-seeking path for long audio. Engine-specific timestamp and
suppression rules differ; their effects are part of this comparison.

Completed: 548 serial HTTP requests, 548 requests at concurrency four, 18
streaming requests and eight initial smoke requests; 564 reference results;
eight raw-decoder probes and six generation-policy probes. All reference cases
completed without runtime errors. Four HTTP requests failed: the two silence
inputs in timestamped mode, repeated in serial and concurrent runs. An HTTP
200 did not mean a valid transcript: many timestamped results were empty.

## Short-clip results

WER uses NFKC, lowercase, punctuation-to-space replacement and diacritic removal,
retaining Arabic letter distinctions. Lower is better.

| Language / response | vLLM WER | FP32 WER | vLLM CER | FP32 CER |
| --- | ---: | ---: | ---: | ---: |
| English JSON | 7.63% | 7.63% | 4.02% | 4.02% |
| Arabic JSON | 49.96% | 49.91% | 12.85% | 12.83% |
| English verbose JSON | 41.94% | 7.30% | 40.64% | 3.64% |
| Arabic verbose JSON | 100.00% | 49.82% | 100.00% | 12.74% |

All 128 normalized English JSON transcripts matched the reference exactly;
125 of 128 Arabic JSON transcripts did. Arabic JSON had one additional word
error over 2,264 reference words. Its paired sentence-cluster bootstrap interval
for the WER difference was 0.00–0.15 percentage points. These intervals describe
this observed corpus, not unseen production traffic.

Most Arabic JSON errors were substitutions: 1,067 substitutions, 28 deletions
and 36 insertions. A separately labeled, lossy orthographic normalization
reduced Arabic JSON WER to 24.16% versus 24.07% for FP32. It folds common letter
variants as documented in the runner; it does not replace the primary score.
The large absolute Arabic error rate was reproduced outside vLLM. It still
requires evaluation against the intended domain and transcription conventions.

## Long audio and repeatability

Weighted WER over the four original, unshifted long inputs per language:

| Language | vLLM JSON | FP32, same chunks | FP32, native long decoding |
| --- | ---: | ---: | ---: |
| English | 15.97% | 16.10% | 23.40% |
| Arabic | 51.26% | 51.26% | 53.15% |

Across all 16 long/shifted cases, the largest word-error-count difference
between JSON and the same-chunk reference was two words; vLLM had no excess
word errors in those cases. Four of eight English and seven of eight Arabic
normalized transcripts matched exactly. The native long reference was not
consistently more accurate on this checkpoint and fixture set.

Moving chunk boundaries still matters: one second of leading silence changed
an English case by 13 word errors, or 3.23 percentage points, in both the vLLM
and same-chunk reference paths. This is not evidence of universally robust
long-form segmentation.

Concurrent execution changed five normalized transcripts, with no net change
in word-error count. All 18 streams completed; one long English stream differed
by one word error from the serial result. There was no additional systematic
degradation at the tested concurrency.

## Confirmed failure mechanisms

Timestamped responses were empty for 50/128 English and 128/128 Arabic short
clips. The raw-decoder probe demonstrated the mechanism: an Arabic request
generated recognizable speech text and EOS, but no closing timestamp token.
The unchanged segment parser produced zero segments, and the HTTP path built
its text from those segments. The text was lost after decoding. The English
positive control generated a closing timestamp and retained its transcript.
The relevant [v0.31.0 serving implementation](https://docs.vllm.ai/en/v0.31.0/api/vllm/entrypoints/speech_to_text/base/serving/)
changes the decoder prompt for verbose responses and reconstructs text from
parsed segments.

Both silence inputs generated EOS alone. After removing EOS, the parser indexed
a tuple that was too short and raised `IndexError`, reproducing the HTTP 500.

The quiet English clip `en-short-092` was empty in ordinary JSON and the FP32
reference. Its waveform was nonzero but quiet (RMS 0.00159, peak 0.01315).
Restoring stock initial EOS suppression in the reference recovered partial
speech, but caused silent inputs to emit “you” and “شكرا”. This is a real
omission/hallucination tradeoff, not a safe blanket configuration fix. No
checkpoint configuration was changed. Changing that HF field alone also does
not add the corresponding rule to vLLM's transcription sampler.

For the tested combination, use explicit language and `response_format=json`.
Do not regard `verbose_json` as verified until timestamp generation/parsing is
fixed and revalidated. These results establish current behavior, not an
improvement over an unavailable previous revision of the fine-tune.

## Evidence, checks and cost

Local raw evidence is under `runs/whisper-quality-audit/collected/`; complete
paired reports are `report-serial.json`, `report-concurrent.json` and
`report-stream.json` under `runs/whisper-quality-audit/`. The serial report's
SHA256 is `b59c1b2252b32210735b40ad6f0f40fd6fb573790b46835ceb12c02e8f440946`.
The report rejects mismatched model/input fingerprints and records incomplete
collections separately. A live partial reference was correctly identified as
incomplete before the final reports were generated.

CPU preflight: **589 passed, eight skipped, PREFLIGHT GREEN**. Formatting,
lint and repository alignment checks passed. These are harness checks; they
do not turn the failed timestamped ASR cases into passes.

One watched Vast.ai development rental and one model staging transfer were
used. Checkpoint bytes and caches were reused across all process restarts.
Elapsed rental time was 65.57 minutes at $0.382222/hour, approximately $0.418
for compute/storage. Guest counters recorded 2,966,456,146 received bytes and
19,533,144 transmitted bytes before final collection. Advertised transfer rates
were $0.001302/GB in each direction. These counters exclude provider-side image
pulls and the last collection; this is not an invoice total. The pinned image
had 9,134,011,886 compressed layer bytes.

All model processes had exited and GPU allocation was zero before teardown.
The provider confirmed destruction, and the external label watchdog separately
confirmed that no matching rental remained. No image or model was published.
