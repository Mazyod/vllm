"""Run the same pinned checkpoint through HTTP vLLM and independent FP32 HF.

Responses and provenance are retained verbatim; this runner never decides that
an HTTP success implies transcription quality. Language is always explicit.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "server"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="bbyn-v2")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    cases = manifest["cases"]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
        assert len(cases) == len(set(args.ids)), "Unknown case id"
    if args.limit:
        cases = cases[: args.limit]
    assert cases
    for case in cases:
        assert case["language"] in ("en", "ar")
        assert sha(args.fixtures / case["file"]) == case["sha256"]
    versions = {}
    for package in ("vllm", "torch", "transformers", "soundfile", "httpx"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    provenance = dict(
        mode=args.mode,
        versions=versions,
        model_files={
            p.name: sha(p)
            for p in args.model.iterdir()
            if p.is_file() and not p.name.startswith(".")
        },
        manifest_sha256=sha(args.fixtures / "manifest.json"),
        concurrency=args.concurrency,
        case_ids=[c["id"] for c in cases],
        stream=args.stream,
        explicit_language=True,
        model_revision="e910dbbc381deab0561e944c7149c3ace41b6bff",
        reference_precision="FP32; TF32 disabled" if args.mode == "reference" else None,
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results_path = args.out / "responses.jsonl"
    if results_path.exists():
        raise FileExistsError("Use a new output directory; results are immutable")
    if args.mode == "reference":
        run_reference(args, cases, results_path)
    else:
        run_server(args, cases, results_path)


def emit(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    print(
        json.dumps(
            {
                k: v
                for k, v in row.items()
                if k in ("id", "variant", "elapsed_s", "error")
            }
        ),
        flush=True,
    )


def run_reference(args, cases, results_path):
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from vllm.multimodal.audio import split_audio

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    (args.out / "reference-generation-config.json").write_text(
        model.generation_config.to_json_string()
    )

    def decode(audio, language, timestamps):
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        ).to("cuda")
        output = model.generate(
            **inputs,
            language=language,
            task="transcribe",
            do_sample=False,
            num_beams=1,
            return_timestamps=timestamps,
            condition_on_prev_tokens=False,
            max_length=448,
        )
        sequences = output["sequences"] if isinstance(output, dict) else output
        return dict(
            text=processor.batch_decode(sequences, skip_special_tokens=True)[0],
            token_ids=sequences.cpu().tolist(),
        )

    with results_path.open("w") as stream, torch.inference_mode():
        for case in cases:
            audio, sr = sf.read(args.fixtures / case["file"], dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            variants = (
                ["json", "verbose_json"]
                if len(audio) <= 480000
                else ["native_long", "json", "verbose_json"]
            )
            for variant in variants:
                started = time.monotonic()
                try:
                    if variant == "native_long" or len(audio) <= 480000:
                        result = decode(audio, case["language"], variant != "json")
                    else:
                        chunks = split_audio(audio, 16000, 30, 1, 1600)
                        assert sum(map(len, chunks)) == len(audio)
                        parts = [
                            decode(c, case["language"], variant != "json")
                            for c in chunks
                        ]
                        result = dict(
                            text=" ".join(p["text"] for p in parts),
                            parts=parts,
                            chunk_samples=[len(c) for c in chunks],
                        )
                    row = dict(
                        id=case["id"],
                        variant=variant,
                        response=result,
                        audio_sha256=case["sha256"],
                        language=case["language"],
                    )
                except Exception as exc:
                    row = dict(id=case["id"], variant=variant, error=repr(exc))
                row["elapsed_s"] = time.monotonic() - started
                emit(stream, row)


def run_server(args, cases, results_path):
    import httpx

    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )

    def request(case, variant):
        started = time.monotonic()
        data = dict(
            model=args.served_model,
            language=case["language"],
            temperature="0",
            response_format=variant,
            max_completion_tokens="444",
        )
        if variant == "verbose_json":
            data["timestamp_granularities[]"] = "segment"
        if args.stream:
            data["stream"] = "true"
        row = dict(
            id=case["id"],
            variant=variant,
            request=data,
            audio_sha256=case["sha256"],
            language=case["language"],
        )
        try:
            with httpx.Client(
                base_url=args.base_url, headers=headers, timeout=900
            ) as client:
                with (args.fixtures / case["file"]).open("rb") as audio:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data=data,
                        files={"file": (case["file"], audio)},
                    )
                row["status"] = response.status_code
                response.raise_for_status()
                if args.stream:
                    events = [
                        json.loads(line[6:])
                        for line in response.text.splitlines()
                        if line.startswith("data: ") and line[6:] != "[DONE]"
                    ]
                    text = "".join(
                        c.get("delta", {}).get("content", "")
                        for e in events
                        for c in e.get("choices", [])
                    )
                    row["response"] = dict(text=text, events=events)
                else:
                    row["response"] = response.json()
        except Exception as exc:
            row["error"] = repr(exc)
        row["elapsed_s"] = time.monotonic() - started
        return row

    tasks = [
        (case, variant)
        for case in cases
        for variant in (["json"] if args.stream else ["json", "verbose_json"])
    ]
    with (
        results_path.open("w") as stream,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        for row in pool.map(lambda pair: request(*pair), tasks):
            emit(stream, row)


if __name__ == "__main__":
    main()
