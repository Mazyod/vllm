"""Create reproducible public-document and multilingual speech workloads.

Inputs: pinned NIST SP 800-53r5 PDF and FLEURS validation parquet files.
The media stays outside Git; the generated manifest records its provenance.
"""

import argparse
import hashlib
import io
import json
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pymupdf as fitz
import soundfile as sf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
pdf = args.source / "nist-controls.pdf"
assert (
    hashlib.sha256(pdf.read_bytes()).hexdigest()
    == "fc63bcd61715d0181dd8e85998b1e6201ae3515fc6626102101cab1841e11ec6"
)
doc = fitz.open(pdf)
pages = [{"page": i + 1, "text": p.get_text(sort=True)} for i, p in enumerate(doc)]
(args.out / "documents.json").write_text(json.dumps(pages, ensure_ascii=False) + "\n")
# Use substantive control pages, not a synthetic page or a repeated filler.
selected = [p for p in pages if p["page"] > 30 and len(p["text"]) > 2500]
selected = selected[:: max(1, len(selected) // 8)][:8]
images = []
for old in args.out.glob("nist-page-*.png"):
    old.unlink()
for item in selected:
    name = f"nist-page-{item['page']}.png"
    doc[item["page"] - 1].get_pixmap(dpi=250, alpha=False).save(args.out / name)
    images.append(
        {
            "file": name,
            "page": item["page"],
            "sha256": hashlib.sha256((args.out / name).read_bytes()).hexdigest(),
        }
    )

speech = []
for config, language in [("en_us", "en"), ("ar_eg", "ar")]:
    path = (
        args.source
        / "fleurs"
        / "parquet-data"
        / config
        / "validation-00000-of-00001.parquet"
    )
    batches = islice(pq.ParquetFile(path).iter_batches(batch_size=32), 2)
    for batch_index, batch in enumerate(batches):
        rows = batch.to_pylist()
        joined, transcripts, duration = [], [], 0.0
        for i, row in enumerate(rows):
            data = row["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            if batch_index == 0:
                name = f"{language}-{i:02d}.wav"
                (args.out / name).write_bytes(data)
                speech.append(
                    {
                        "id": f"{language}-{i:02d}",
                        "file": name,
                        "language": language,
                        "duration_s": len(audio) / sr,
                        "text": row["transcription"],
                        "dataset_row_id": row["id"],
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            if duration < 180:
                joined.extend([audio, np.zeros(sr // 4, dtype=np.float32)])
                transcripts.append(row["transcription"])
                duration += len(audio) / sr + 0.25
        assert duration >= 180, (config, duration)
        identifier = f"{language}-" + ("heldout-" if batch_index else "") + "long"
        name = identifier + ".flac"
        sf.write(args.out / name, np.concatenate(joined), 16000, subtype="PCM_16")
        speech.append(
            {
                "id": identifier,
                "heldout": batch_index == 1,
                "file": name,
                "language": language,
                "duration_s": duration,
                "text": " ".join(transcripts),
                "construction": (
                    "distinct validation utterances, 250 ms silence between utterances"
                ),
                "sha256": hashlib.sha256((args.out / name).read_bytes()).hexdigest(),
            }
        )
manifest = {
    "documents": {
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": len(pages),
    },
    "images": images,
    "speech": speech,
    "speech_source": {
        "repo": "google/fleurs",
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "split": "validation",
        "license": "CC-BY-4.0",
    },
}
(args.out / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        {
            "pages": len(pages),
            "images": len(images),
            "audio_cases": len(speech),
            "long_audio_s": {
                r["id"]: r["duration_s"] for r in speech if r["id"].endswith("long")
            },
        },
        indent=2,
    )
)
