"""Export anonymous benchmark tables with resource samples from each cell window."""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = []
    with (args.collected / "gpu.csv").open() as stream:
        for row in csv.reader(stream):
            if len(row) != 9:
                continue
            when = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            gpu.append([when.replace(tzinfo=UTC).timestamp(), *map(float, row[1:])])
    host = read_rows(args.collected / "host.jsonl")
    case_sets = {
        hashlib.sha256(p.read_bytes()).hexdigest(): {
            c["id"]: c for c in json.loads(p.read_text())
        }
        for p in args.cases
    }
    cells, quality, provenance, asr_corpus = [], [], {}, []
    for protocol_path in sorted(args.collected.glob("*/protocol.json")):
        path = protocol_path.parent / "summary.json"
        run = path.parent.name
        protocol = json.loads(protocol_path.read_text())
        requests = read_rows(path.parent / "requests.jsonl")
        assert all(
            r["reported_prompt_tokens"] == r["tokens"]
            for r in requests
            if r["service"] != "whisper" and r.get("http_success")
        ), f"Server token accounting differs from the full input: {run}"
        provenance[run] = {
            "full_input_token_counts_verified": True,
            "mode": protocol["mode"],
            "parameters": {
                k: protocol[k]
                for k in (
                    "services",
                    "ids",
                    "concurrency",
                    "min_requests",
                    "mixed_cohort",
                    "minutes",
                    "rpm",
                )
            },
            "validated_requests": len(requests),
            "warmup_requests": sum(r["cell"].endswith("-warmup") for r in requests),
            "has_summary": path.exists(),
            "harness_sha256": protocol["harness_sha256"],
            "cases_sha256": protocol["cases_sha256"],
            "requests_sha256": hashlib.sha256(
                (path.parent / "requests.jsonl").read_bytes()
            ).hexdigest(),
            "config_sha256": {
                name: launch["config_sha256"]
                for name, launch in protocol["serving"].items()
            },
        }
        for summary in json.loads(path.read_text()) if path.exists() else []:
            selected = requests
            if "service" in summary:
                selected = [r for r in selected if r["service"] == summary["service"]]
            if "concurrency" in summary:
                cell = f"{summary['case']}-c{summary['concurrency']}"
                selected = [r for r in selected if r["cell"] == cell]
            assert selected, (run, summary)
            first = min(r["started_unix"] for r in selected)
            last = max(r["ended_unix"] for r in selected)
            entry = {
                "run": run,
                "mode": protocol["mode"],
                **{k: v for k, v in summary.items() if not isinstance(v, (dict, list))},
                "services": ",".join(sorted({r["service"] for r in selected})),
                "resource_window_s": last - first,
                "error_count": len(summary.get("errors", [])),
            }
            for index in (0, 1):
                samples = [r for r in gpu if r[1] == index and first <= r[0] <= last]
                entry[f"gpu{index}_samples"] = len(samples)
                if samples:
                    for label, column, reducer in (
                        ("peak_mib", 2, max),
                        ("mean_util_pct", 3, mean),
                        ("peak_util_pct", 3, max),
                        ("mean_power_w", 5, mean),
                        ("peak_temp_c", 6, max),
                        ("min_sm_mhz", 7, min),
                    ):
                        entry[f"gpu{index}_{label}"] = reducer(
                            r[column] for r in samples
                        )
            samples = [r for r in host if first <= r["unix_s"] <= last]
            entry["host_incomplete_samples"] = sum(
                r["unreadable"] != 0 for r in samples
            )
            samples = [r for r in samples if r["unreadable"] == 0]
            entry["host_samples"] = len(samples)
            if samples:
                entry["host_peak_pss_gib"] = (
                    max(r["pss_bytes"] for r in samples) / 2**30
                )
                entry["host_peak_cpu_pct"] = max(r["cpu_percent"] for r in samples)
            cells.append(entry)
        if protocol["mode"] == "quality" and case_sets:
            import jiwer

            from fork.bench.asr_audit.report import normalize

            cases = case_sets[protocol["cases_sha256"]]
            for language in ("en", "ar"):
                expected = [
                    c
                    for c in cases.values()
                    if c.get("kind") == "short" and c.get("language") == language
                ]
                rows = [
                    r
                    for r in requests
                    if "actual" in r
                    and r.get("language") == language
                    and cases[r["case"]].get("kind") == "short"
                ]
                if not rows:
                    continue
                truth = [normalize(cases[r["case"]]["ground_truth"]) for r in rows]
                actual = [normalize(r["text"]) for r in rows]
                reference = [
                    normalize(cases[r["case"]]["reference_text"]) for r in rows
                ]
                words = sum(r["actual"]["words"] for r in rows)
                asr_corpus.append(
                    dict(
                        run=run,
                        language=language,
                        clips=len(rows),
                        complete_cases=len(rows) == len(expected),
                        words=words,
                        actual_wer=sum(r["actual"]["errors"] for r in rows) / words,
                        reference_wer=sum(r["reference"]["errors"] for r in rows)
                        / words,
                        actual_cer=jiwer.cer(truth, actual),
                        reference_cer=jiwer.cer(truth, reference),
                        normalized_transcripts_equal=sum(
                            a == b for a, b in zip(actual, reference)
                        ),
                    )
                )
        if protocol["mode"] == "quality":
            for row in requests:
                entry = {
                    "run": run,
                    **{
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "service",
                            "case",
                            "passed",
                            "latency_s",
                            "cosine_mean",
                            "cosine_min",
                            "cosine_p01",
                            "max_abs_difference",
                            "top1_agrees",
                            "language",
                            "real_time_factor",
                            "audio_s",
                        )
                    },
                }
                if "actual" in row:
                    entry.update(
                        words=row["actual"]["words"],
                        actual_errors=row["actual"]["errors"],
                        reference_errors=row["reference"]["errors"],
                        actual_wer=row["actual"]["wer"],
                        reference_wer=row["reference"]["wer"],
                        actual_cer=row["actual"]["cer"],
                        reference_cer=row["reference"]["cer"],
                    )
                quality.append(entry)
    for name, rows in (("benchmarks", cells), ("quality", quality)):
        assert rows, name
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (args.out / (name + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if case_sets:
        (args.out / "asr-corpus.json").write_text(
            json.dumps(asr_corpus, indent=2) + "\n"
        )
    print(json.dumps(dict(cells=len(cells), quality_cases=len(quality))))


if __name__ == "__main__":
    main()
