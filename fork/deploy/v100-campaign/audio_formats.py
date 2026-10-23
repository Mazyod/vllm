"""Prepare uploaded WAV/MP3/AAC probes and bind their decoded-wave FP32 oracles."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "bind"])
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures/asr"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records_path = args.out / "encodings.json"
    if args.mode == "prepare":
        import av
        import numpy as np
        import soundfile as sf
        from torchcodec.decoders import AudioDecoder

        args.out.mkdir(parents=True, exist_ok=False)
        audio_root = args.out / "audio"
        reference_root = args.out / "reference-fixtures"
        audio_root.mkdir()
        reference_root.mkdir()
        manifest = json.loads((args.fixtures / "manifest.json").read_text())
        records, references = [], []
        for case in manifest["cases"]:
            if case["id"] not in (
                "en-short-042",
                "ar-short-110",
                "en-long-0",
                "ar-long-0",
            ):
                continue
            source = args.fixtures / case["file"]
            assert sha(source) == case["sha256"]
            audio, rate = sf.read(source, dtype="float32")
            assert rate == 16000 and audio.ndim == 1
            for extension in ("wav", "mp3", "m4a"):
                name = f"format-{case['id']}-{extension}"
                path = audio_root / (name + "." + extension)
                if extension == "wav":
                    sf.write(path, audio, rate, subtype="PCM_16")
                else:
                    codec = "libmp3lame" if extension == "mp3" else "aac"
                    with av.open(str(path), "w") as container:
                        stream = container.add_stream(codec, rate=rate)
                        stream.layout = "mono"
                        stream.bit_rate = 128000 if extension == "mp3" else 64000
                        frame = av.AudioFrame.from_ndarray(
                            audio[None, :], format="flt", layout="mono"
                        )
                        frame.sample_rate = rate
                        for packet in (*stream.encode(frame), *stream.encode(None)):
                            container.mux(packet)
                decoded = AudioDecoder(
                    str(path), sample_rate=16000, num_channels=1
                ).get_all_samples()
                waveform = decoded.data.numpy().reshape(-1)
                assert np.isfinite(waveform).all() and len(waveform) > 0
                assert abs(len(waveform) - len(audio)) <= 2048, "Codec changed duration"
                reference_path = reference_root / (name + ".wav")
                sf.write(reference_path, waveform, rate, subtype="FLOAT")
                reference = dict(
                    id=name,
                    language=case["language"],
                    file=reference_path.name,
                    duration_s=len(waveform) / rate,
                    text=case["text"],
                    dataset_row_ids=case["dataset_row_ids"],
                    kind="format",
                    sha256=sha(reference_path),
                )
                references.append(reference)
                records.append(
                    dict(
                        **reference,
                        encoded_file=str(path.relative_to(args.out)),
                        encoded_sha256=sha(path),
                        encoded_bytes=path.stat().st_size,
                        source_sha256=case["sha256"],
                    )
                )
        assert len(records) == 12
        (reference_root / "manifest.json").write_text(
            json.dumps(dict(cases=references), ensure_ascii=False) + "\n"
        )
        records_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        )
        print("Twelve encodings decoded successfully; FP32 reference waves prepared")
    else:
        provenance = json.loads((args.out / "reference/provenance.json").read_text())
        assert (
            provenance["model_revision"] == "e910dbbc381deab0561e944c7149c3ace41b6bff"
        )
        assert provenance["model_files"]["model.safetensors"] == (
            "79d6f3b3717e35761254622c0d9f8347f5385cbaef5d0f1c34d0fa220d6ac330"
        )
        references = {
            row["id"]: row
            for row in map(
                json.loads,
                (args.out / "reference/responses.jsonl").read_text().splitlines(),
            )
            if row["variant"] == "json"
        }
        cases = []
        for item in json.loads(records_path.read_text()):
            reference = references[item["id"]]
            assert (
                "error" not in reference and reference["audio_sha256"] == item["sha256"]
            )
            cases.append(
                dict(
                    id=item["id"],
                    service="whisper",
                    endpoint="/v1/audio/transcriptions",
                    file=item["encoded_file"],
                    audio_sha256=item["encoded_sha256"],
                    data=dict(
                        model="whisper",
                        language=item["language"],
                        temperature="0",
                        response_format="json",
                        max_completion_tokens="444",
                    ),
                    language=item["language"],
                    duration_s=item["duration_s"],
                    kind="format",
                    ground_truth=item["text"],
                    reference_text=reference["response"]["text"],
                    reference_wave_sha256=item["sha256"],
                    items=1,
                )
            )
        (args.out / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False) + "\n"
        )
        print("Twelve encoded-upload cases bound to their decoded-wave FP32 references")


if __name__ == "__main__":
    main()
