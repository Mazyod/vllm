"""Add real MP3/AAC and automatic-language cases to the speech fixtures."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fixtures", type=Path)
args = parser.parse_args()
path = args.fixtures / "manifest.json"
manifest = json.loads(path.read_text())
variant_ids = {
    f"{language}-{kind}" for language in ["en", "ar"] for kind in ["mp3", "m4a", "auto"]
}
manifest["speech"] = [r for r in manifest["speech"] if r["id"] not in variant_ids]
for language in ["en", "ar"]:
    original = next(r for r in manifest["speech"] if r["id"] == language + "-00")
    for extension, options in [
        ("mp3", ["-c:a", "libmp3lame", "-q:a", "4"]),
        ("m4a", ["-c:a", "aac", "-b:a", "128k"]),
    ]:
        target = args.fixtures / (language + "-codec." + extension)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(args.fixtures / original["file"]),
                *options,
                str(target),
            ],
            check=True,
        )
        row = {
            **original,
            "id": language + "-" + extension,
            "file": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "variant": extension,
        }
        manifest["speech"].append(row)
    manifest["speech"].append(
        {**original, "id": language + "-auto", "auto_language": True}
    )
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print("Speech cases:", len(manifest["speech"]))
