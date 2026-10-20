"""Check correctness under open-loop mixed traffic and full-payload bursts."""

import argparse
import copy
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from speech_metrics import errors, words

PORTS = {"snowflake": 8001, "bge": 8002, "whisper": 8003, "gemma": 8004, "jina": 8005}


@lru_cache(maxsize=128)
def reference(path):
    with np.load(path) as f:
        return f["reference"].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--minutes", type=float, default=10, help="Steady duration; 0 runs bursts only."
    )
    parser.add_argument("--rpm", type=float, default=30)
    args = parser.parse_args()
    assert args.minutes >= 0 and args.rpm > 0
    args.out.mkdir(parents=True, exist_ok=False)
    cases = {name: [] for name in PORTS}
    for root in args.reports:
        reports = list(root.glob("*-quality.json"))
        assert len(reports) == 1, root
        quality = json.loads(reports[0].read_text())
        assert quality["passed"], f"Quality gate failed: {root}"
        if reports[0].name == "asr-quality.json":
            assert set(quality["corpus_wer"]) == {"en", "ar"}
        if reports[0].name == "jina-quality.json":
            assert len(quality["maxsim"]) == 3
            assert all(r["top1_agrees"] for r in quality["maxsim"])
        for line in (root / "requests.jsonl").read_text().splitlines():
            case = json.loads(line)
            case["root"] = str(root)
            cases[case["service"]].append(case)
    assert all(cases.values()), {k: len(v) for k, v in cases.items()}
    required = {
        "snowflake": {"snowflake-8192", "snowflake-batch"},
        "bge": {"bge-8192", "bge-batch"},
        "gemma": {"gemma-2048", "gemma-batch"},
        "whisper": {"en-long", "ar-long", "en-heldout-long", "ar-heldout-long"},
        "jina": {"text-32768", "images-8"},
    }
    for name, identifiers in required.items():
        assert identifiers <= {c["id"] for c in cases[name]}, name
    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )
    client = httpx.Client(
        timeout=1200, headers=headers, limits=httpx.Limits(max_connections=64)
    )
    log_lock = threading.Lock()
    started_unix = time.time()
    origin = time.monotonic()
    records = []

    def run(case, phase, scheduled):
        start = time.monotonic()
        item = {
            "service": case["service"],
            "case": case["id"],
            "phase": phase,
            "scheduled_s": scheduled - origin,
            "started_s": start - origin,
            "passed": False,
        }
        try:
            url = f"http://127.0.0.1:{PORTS[case['service']]}{case['endpoint']}"
            if "file" in case:
                path = args.fixtures / case["file"]
                with path.open("rb") as f:
                    response = client.post(
                        url, data=case["data"], files={"file": (path.name, f)}
                    )
            else:
                payload = copy.deepcopy(case["payload"])
                for message in payload.get("messages", []):
                    for part in message.get("content", []):
                        if part["type"] == "image_url":
                            part["uuid"] = uuid4().hex
                            item["fresh_image_ids"] = True
                response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            if case["service"] == "whisper":
                assert body.get("text", "").strip()
                assert abs(body.get("duration", 0) - case["duration_s"]) < 1
                if case["duration_s"] > 30:
                    assert (
                        max((s["end"] for s in body.get("segments", [])), default=0)
                        >= case["duration_s"] - 8
                    )
                item["real_time_factor"] = (time.monotonic() - start) / case[
                    "duration_s"
                ]
                truth = words(case["ground_truth"])
                item.update(
                    language=case["language"],
                    words=len(truth),
                    actual_errors=errors(truth, words(body["text"])),
                    reference_errors=case["reference_errors"],
                )
                if case["duration_s"] > 30:
                    assert item["actual_errors"] <= item[
                        "reference_errors"
                    ] + 0.02 * len(truth)
                elif not case["id"].split("-")[-1].isdigit():
                    assert item["actual_errors"] <= item["reference_errors"] + 1
            else:
                expected = reference(str(Path(case["root"]) / case["reference"]))
                if case["service"] == "bge":
                    rows = sorted(body["results"], key=lambda x: x["index"])
                    assert [r["index"] for r in rows] == list(range(len(rows)))
                    actual = np.asarray([r["relevance_score"] for r in rows])
                    assert actual.shape == expected.shape and np.isfinite(actual).all()
                    assert np.allclose(actual, expected, atol=0.01, rtol=0)
                    assert actual.argmax() == expected.argmax()
                else:
                    if case["service"] == "jina":
                        actual = np.asarray(body["data"][0]["data"], dtype=np.float32)
                    else:
                        actual = np.asarray(
                            [
                                r["embedding"]
                                for r in sorted(body["data"], key=lambda x: x["index"])
                            ]
                        )
                    assert actual.shape == expected.shape and np.isfinite(actual).all()
                    assert np.allclose(np.linalg.norm(actual, axis=-1), 1, atol=0.005)
                    cosine = (actual * expected).sum(-1) / (
                        np.linalg.norm(actual, axis=-1)
                        * np.linalg.norm(expected, axis=-1)
                    )
                    floor = case.get("cosine_floor", 0.999)
                    assert (
                        cosine.mean() if case["service"] == "jina" else cosine.min()
                    ) >= floor
                    item["cosine_mean"] = float(cosine.mean())
            item["passed"] = True
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        item["completed_s"] = time.monotonic() - origin
        item["latency_s"] = time.monotonic() - start
        with log_lock:
            records.append(item)
            with (args.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(item) + "\n")
        return item

    rng = random.Random(20261020)
    names = list(PORTS)
    futures = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        count = round(args.minutes * args.rpm)
        for i in range(count):
            scheduled = origin + i * 60 / args.rpm
            time.sleep(max(0, scheduled - time.monotonic()))
            name = names[i % len(names)]
            case = rng.choice(cases[name])
            futures.append(pool.submit(run, case, "steady", scheduled))
            if i % 15 == 0:
                print(
                    json.dumps(
                        {
                            "submitted": i + 1,
                            "completed": sum(f.done() for f in futures),
                        }
                    ),
                    flush=True,
                )
        for future in futures:
            future.result()
        for concurrency in [5, 10, 20]:
            burst = []
            for i in range(concurrency):
                name = names[i % len(names)]
                case = max(
                    cases[name], key=lambda c: c.get("tokens", c.get("duration_s", 0))
                )
                if name == "bge":
                    # Attention cost follows the longest pair, not the sum of a
                    # 32-candidate short batch. Exercise the real 8192-token limit.
                    case = next(c for c in cases[name] if c["id"] == "bge-8192")
                burst.append(
                    pool.submit(run, case, f"burst-{concurrency}", time.monotonic())
                )
            for future in burst:
                future.result()
        image_case = next(c for c in cases["jina"] if c["id"] == "images-8")
        text_case = next(c for c in cases["jina"] if c["id"] == "text-32768")
        mixed_images = [
            next(c for c in cases["jina"] if c["id"] == f"images-{n}")
            for n in [1, 2, 8]
        ]
        for phase, payloads in [
            ("jina-images-2", [image_case] * 2),
            ("jina-images-4", [image_case] * 4),
            ("jina-mixed-2", [image_case, text_case]),
            ("jina-images-mixed-3", mixed_images),
            (
                "bge-batches-4",
                [next(c for c in cases["bge"] if c["id"] == "bge-batch")] * 4,
            ),
        ]:
            burst = [pool.submit(run, c, phase, time.monotonic()) for c in payloads]
            for future in burst:
                future.result()
    client.close()
    report = {
        "passed": all(r["passed"] for r in records),
        "started_unix": started_unix,
        "ended_unix": time.time(),
        "target_rpm": args.rpm,
        "steady_minutes": args.minutes,
        "requests": len(records),
        "failures": [r for r in records if not r["passed"]],
        "wall_s": time.monotonic() - origin,
        "services": {},
    }
    for name in names:
        report["services"][name] = {}
        for phase in dict.fromkeys(r["phase"] for r in records):
            rows = [r for r in records if r["service"] == name and r["phase"] == phase]
            if rows:
                latencies = [r["latency_s"] for r in rows]
                report["services"][name][phase] = {
                    "requests": len(rows),
                    "p50_s": float(np.quantile(latencies, 0.5)),
                    "p95_s": float(np.quantile(latencies, 0.95)),
                    "max_s": max(latencies),
                    "max_client_queue_s": max(
                        r["started_s"] - r["scheduled_s"] for r in rows
                    ),
                }
    completed_in_window = sum(
        120 <= r["completed_s"] <= args.minutes * 60
        for r in records
        if r["phase"] == "steady" and r["passed"]
    )
    if args.minutes > 2:
        report["steady_completed_rpm_after_warmup"] = completed_in_window / (
            args.minutes - 2
        )
        report["capacity_passed"] = (
            report["steady_completed_rpm_after_warmup"] >= 0.95 * args.rpm
        )
        report["steady_rpm_by_service"] = {
            name: sum(
                120 <= r["completed_s"] <= args.minutes * 60
                for r in records
                if r["phase"] == "steady" and r["passed"] and r["service"] == name
            )
            / (args.minutes - 2)
            for name in names
        }
        report["capacity_passed"] &= all(
            rate >= 0.95 * args.rpm / len(names)
            for rate in report["steady_rpm_by_service"].values()
        )
        report["passed"] &= report["capacity_passed"]
    report["asr_under_load"] = {}
    for language in ["en", "ar"]:
        rows = [
            r for r in records if r.get("language") == language and "actual_errors" in r
        ]
        if rows:
            count = sum(r["words"] for r in rows)
            actual = sum(r["actual_errors"] for r in rows) / count
            expected = sum(r["reference_errors"] for r in rows) / count
            passed = actual <= expected + 0.02
            report["asr_under_load"][language] = {
                "actual_wer": actual,
                "reference_wer": expected,
                "passed": passed,
            }
            report["passed"] &= passed
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
