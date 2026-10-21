"""Build pinned, public FLEURS cases without truncating speech or transcripts."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    sources = {}

    def save(identifier, language, audio, text, row_ids, group, kind):
        path = args.out / (identifier + ".flac")
        sf.write(path, audio, 16000, subtype="PCM_16")
        cases.append(
            dict(
                id=identifier,
                language=language,
                file=path.name,
                duration_s=len(audio) / 16000,
                text=text,
                dataset_row_ids=row_ids,
                group=group,
                kind=kind,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )

    for language, config in [("en", "en_us"), ("ar", "ar_eg")]:
        path = (
            args.source / "parquet-data" / config / "validation-00000-of-00001.parquet"
        )
        sources[config] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = pq.read_table(path).to_pylist()
        decoded = []
        for row in rows:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            decoded.append(audio)
        for index in range(128):
            identifier = f"{language}-short-{index:03d}"
            save(
                identifier,
                language,
                decoded[index],
                rows[index]["transcription"],
                [rows[index]["id"]],
                identifier,
                "short",
            )
        cursor = 128
        for index in range(4):
            segments, texts, ids = [], [], []
            count = 0
            while count < 180 * 16000:
                segments.extend([decoded[cursor], np.zeros(4000, dtype=np.float32)])
                texts.append(rows[cursor]["transcription"])
                ids.append(rows[cursor]["id"])
                count += len(decoded[cursor]) + 4000
                cursor += 1
            audio = np.concatenate(segments)
            identifier = f"{language}-long-{index}"
            save(identifier, language, audio, " ".join(texts), ids, identifier, "long")
            # Same speech, different split alignment: never count as independent data.
            save(
                identifier + "-shift",
                language,
                np.concatenate([np.zeros(16000, dtype=np.float32), audio]),
                " ".join(texts),
                ids,
                identifier,
                "boundary_shift",
            )
        # Silence is a separate hallucination probe, never a zero-denominator WER.
        save(
            language + "-silence",
            language,
            np.zeros(160000, dtype=np.float32),
            "",
            [],
            language + "-silence",
            "silence",
        )
    manifest = dict(
        source=dict(
            repo="google/fleurs",
            revision="70bb2e84b976b7e960aa89f1c648e09c59f894dd",
            split="validation",
            license="CC-BY-4.0",
            parquet_sha256=sources,
        ),
        long_construction="Distinct held-out utterances with 250ms gaps",
        cases=cases,
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                cases=len(cases), audio_hours=sum(c["duration_s"] for c in cases) / 3600
            )
        )
    )


if __name__ == "__main__":
    main()
