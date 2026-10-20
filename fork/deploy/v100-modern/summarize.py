"""Export anonymous, transcript-free validation measurements from a collected run."""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collected = args.run / "collected"

    def read(name):
        return json.loads((collected / name).read_text())

    report = {
        "hardware": {
            "profile": "volta-2",
            "gpu_count": 2,
            "model": "Tesla V100-SXM2-16GB",
            "memory_mib_per_gpu": 16384,
            "compute_capability": "7.0",
            "interconnect": "NODE, no NVLink",
            "driver": "580.173.02",
            "cpu_architecture": "amd64",
        },
        "models": json.loads(Path(__file__).with_name("models.json").read_text()),
        "quality": {},
        "loads": {},
    }
    for key, source in {
        "text": "text-quality-02/text-quality.json",
        "jina": "jina-quality-06/jina-quality.json",
        "asr": "asr-quality-04/asr-quality.json",
    }.items():
        quality = read(source)
        assert quality["passed"], source
        quality["cases"] = [
            {k: v for k, v in case.items() if k not in {"text", "reference_text"}}
            for case in quality["cases"]
        ]
        report["quality"][key] = quality
    for name in ["cold-bursts-allocator", "load-final-20"]:
        data = read(name + "/summary.json")
        assert data["passed"], name
        report["loads"][name] = data
    report["api_contract"] = read("asr-contract-file-media.json")
    assert report["api_contract"]["passed"]
    report["reference_pixel_parity"] = read("pixel-parity.json")
    report["reference_attention_equivalence"] = json.loads(
        (args.run / "reference-equivalence.json").read_text()
    )
    report["wheel_metadata_update"] = read("wheel-metadata-update.json")
    report["engine_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).with_name("engine").glob("*.yaml"))
    }
    windows = [(r["started_unix"], r["ended_unix"]) for r in report["loads"].values()]
    gpu = {index: {"memory_mib": 0, "utilization_percent": 0} for index in [0, 1]}
    with (collected / "gpu-telemetry.csv").open() as stream:
        rows = csv.reader(stream)
        next(rows)
        for row in rows:
            if len(row) != 6:
                continue
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            when = timestamp.replace(tzinfo=UTC).timestamp()
            if any(first <= when <= last for first, last in windows):
                item = gpu[int(row[1])]
                item["memory_mib"] = max(item["memory_mib"], int(row[2].split()[0]))
                item["utilization_percent"] = max(
                    item["utilization_percent"], int(row[3].split()[0])
                )
    host = [
        json.loads(line)
        for line in (collected / "host-metrics/samples.jsonl").read_text().splitlines()
    ]
    host = [
        r for r in host if any(first <= r["unix_s"] <= last for first, last in windows)
    ]
    incomplete = sum(r["unreadable"] != 0 for r in host)
    host = [r for r in host if r["unreadable"] == 0]
    assert host
    report["sampled_peaks"] = {
        "gpu_one_second_samples": gpu,
        "host_five_second_samples": {
            "complete_samples": len(host),
            "incomplete_samples": incomplete,
            "pss_bytes": max(r["pss_bytes"] for r in host),
            "summed_rss_bytes": max(r["rss_bytes"] for r in host),
            "cpu_percent": max(r["cpu_percent"] for r in host),
        },
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
