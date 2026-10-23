"""Receive file-scoped download URLs on stdin; verify bytes before activation."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    import sys

    receipts = []
    for item in json.load(sys.stdin):
        path = Path("/models") / item["name"] / "model.safetensors"
        assert item["name"] in ("gemma", "whisper")
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        reused = path.exists() and path.stat().st_size == item["size"]
        if reused:
            with path.open("rb") as stream:
                reused = (
                    hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
                )
        if not reused:
            partial = path.with_suffix(".partial")
            digest = hashlib.sha256()
            count = 0
            with (
                urllib.request.urlopen(item["url"], timeout=180) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    count += len(chunk)
                    stream.write(chunk)
            assert count == item["size"] and digest.hexdigest() == item["sha256"]
            partial.replace(path)
        receipts.append(
            dict(
                name=item["name"],
                sha256=item["sha256"],
                bytes=item["size"],
                cache_reused=reused,
                elapsed_s=time.monotonic() - started,
            )
        )
        Path("/workspace/results/gated-transfer.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        print(json.dumps(receipts[-1]), flush=True)


if __name__ == "__main__":
    main()
