"""Measure complete multilingual ASR against transcripts and an FP32 reference."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import soundfile as sf
import torch
from speech_metrics import errors, words
from transformers import AutoProcessor, WhisperForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=0, help="Development subset; 0 runs every case."
    )
    parser.add_argument("--reference-cache", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert errors(["a", "b"], ["a", "c"]) == 1 and errors(["a"], []) == 1
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    cases = manifest["speech"][: args.limit] if args.limit else manifest["speech"]
    cached_cases = (
        {r["id"]: r for r in json.loads(args.reference_cache.read_text())["cases"]}
        if args.reference_cache
        else {}
    )
    torch.set_num_threads(16)
    processor = AutoProcessor.from_pretrained("/models/whisper", local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        "/models/whisper",
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    report = {
        "passed": True,
        "cases": [],
        "reference": "Whisper FP32 Transformers, greedy timestamped decoding",
        "max_corpus_wer_regression": 0.02,
        "max_codec_word_regression": 1,
    }
    requests_path = args.out / "requests.jsonl"
    requests_path.write_text("")
    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )
    with (
        httpx.Client(
            base_url="http://127.0.0.1:8003", timeout=1200, headers=headers
        ) as client,
        torch.inference_mode(),
    ):
        for case in cases:
            path = args.fixtures / case["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
            data = {
                "model": "whisper",
                "language": case["language"],
                "temperature": "0",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            }
            if case.get("auto_language"):
                data.pop("language")
            started = time.monotonic()
            with path.open("rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    data=data,
                    files={"file": (path.name, f)},
                )
            response.raise_for_status()
            actual = response.json()
            served_s = time.monotonic() - started
            ref_started = time.monotonic()
            cached = (
                case["id"] in cached_cases
                and cached_cases[case["id"]].get("audio_sha256") == case["sha256"]
            )
            if cached:
                reference = cached_cases[case["id"]]["reference_text"]
            else:
                try:
                    audio, sr = sf.read(path, dtype="float32")
                except sf.LibsndfileError:
                    from torchcodec.decoders import AudioDecoder

                    decoded = AudioDecoder(
                        str(path), sample_rate=16000, num_channels=1
                    ).get_all_samples()
                    audio = decoded.data[0].numpy()
                    sr = decoded.sample_rate
                assert sr == 16000
                inputs = processor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="pt",
                    truncation=False,
                    return_attention_mask=True,
                )
                output = model.generate(
                    **inputs,
                    language=None if case.get("auto_language") else case["language"],
                    task="transcribe",
                    do_sample=False,
                    return_timestamps=True,
                    condition_on_prev_tokens=False,
                )
                sequences = output["sequences"] if isinstance(output, dict) else output
                reference = processor.batch_decode(sequences, skip_special_tokens=True)[
                    0
                ]
            truth = words(case["text"])
            actual_errors = errors(truth, words(actual["text"]))
            reference_errors = errors(truth, words(reference))
            segments = actual.get("segments") or []
            last_end = max((s["end"] for s in segments), default=0)
            complete = bool(
                abs(actual.get("duration", 0) - case["duration_s"]) < 1
                and (case["duration_s"] <= 30 or last_end >= case["duration_s"] - 8)
            )
            if case.get("auto_language"):
                accepted = {"en": {"en", "english"}, "ar": {"ar", "arabic"}}
                complete &= (
                    str(actual.get("language", "")).lower()
                    in accepted[case["language"]]
                )
            entry = {
                "id": case["id"],
                "audio_sha256": case["sha256"],
                "reference_reused": cached,
                "language": case["language"],
                "duration_s": case["duration_s"],
                "server_duration_s": actual.get("duration"),
                "last_segment_end_s": last_end,
                "complete": complete,
                "server_s": served_s,
                "real_time_factor": served_s / case["duration_s"],
                "reference_s": time.monotonic() - ref_started,
                "words": len(truth),
                "actual_errors": actual_errors,
                "reference_errors": reference_errors,
                "actual_wer": actual_errors / max(1, len(truth)),
                "reference_wer": reference_errors / max(1, len(truth)),
                "text": actual["text"],
                "reference_text": reference,
            }
            report["cases"].append(entry)
            report["passed"] &= complete
            if case["duration_s"] > 30:
                report["passed"] &= (
                    actual_errors / max(1, len(truth))
                    <= reference_errors / max(1, len(truth)) + 0.02
                )
            elif not case["id"].split("-")[-1].isdigit():
                report["passed"] &= actual_errors <= reference_errors + 1
            (args.out / "asr-quality.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
            with requests_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": case["id"],
                            "service": "whisper",
                            "endpoint": "/v1/audio/transcriptions",
                            "data": data,
                            "file": case["file"],
                            "duration_s": case["duration_s"],
                            "ground_truth": case["text"],
                            "reference_errors": reference_errors,
                            "language": case["language"],
                        }
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in entry.items()
                        if k not in ["text", "reference_text"]
                    }
                ),
                flush=True,
            )
    totals = {}
    for language in ["en", "ar"]:
        rows = [
            r
            for r in report["cases"]
            if r["language"] == language and r["id"].split("-")[-1].isdigit()
        ]
        if not rows:
            continue
        count = sum(r["words"] for r in rows)
        actual = sum(r["actual_errors"] for r in rows) / count
        reference = sum(r["reference_errors"] for r in rows) / count
        totals[language] = {
            "words": count,
            "actual_wer": actual,
            "reference_wer": reference,
            "passed": actual <= reference + 0.02,
        }
        report["passed"] &= totals[language]["passed"]
    report["corpus_wer"] = totals
    (args.out / "asr-quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"passed": report["passed"], "corpus_wer": totals}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
