"""Measure real HTTP completions and validate every response against FP32 oracles.

Sweep cohorts collect every response before validation, keeping JSON/vector
validation outside the timed cohort. Mixed traffic uses scheduled arrivals and
records client launch delay separately. Every image receives a fresh cache ID.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v100-modern"))
from load_test import PORTS, check_vectors, reference  # noqa: E402

from fork.bench.asr_audit.report import paired_interval, score  # noqa: E402


def summarize(rows, started, ended):
    good = [r for r in rows if r.get("passed")]
    latency = [r["latency_s"] for r in rows if r.get("http_success")]
    span = max(ended - started, 1e-9)
    return dict(
        requests=len(rows),
        http_successes=sum(r.get("http_success", False) for r in rows),
        quality_successes=len(good),
        errors=[
            {k: r.get(k) for k in ("case", "error")}
            for r in rows
            if not r.get("passed")
        ],
        elapsed_s=span,
        successful_requests_s=len(good) / span,
        items_s=sum(r["items"] for r in good) / span,
        input_tokens_s=sum(r.get("tokens", 0) for r in good) / span,
        audio_seconds_s=sum(r.get("audio_s", 0) for r in good) / span,
        images_s=sum(r.get("images", 0) for r in good) / span,
        response_bytes_s=sum(r.get("response_bytes", 0) for r in rows) / span,
        p50_s=float(np.quantile(latency, 0.5)) if latency else None,
        p95_s=float(np.quantile(latency, 0.95)) if len(latency) >= 20 else None,
        p99_s=float(np.quantile(latency, 0.99)) if len(latency) >= 100 else None,
        max_s=max(latency, default=None),
        max_client_launch_delay_s=max(
            (r.get("client_launch_delay_s", 0) for r in rows), default=0
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["quality", "sweep", "mixed"], required=True)
    parser.add_argument("--services", nargs="+", default=list(PORTS))
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--min-requests", type=int, default=24)
    parser.add_argument("--mixed-cohort", action="store_true")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--rpm", type=float, default=30)
    args = parser.parse_args()
    assert all(1 <= c <= 32 for c in args.concurrency) and args.min_requests > 0
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    cases = [
        c
        for c in cases
        if c["service"] in args.services and (not args.ids or c["id"] in args.ids)
    ]
    assert cases
    for case in cases:
        if "root" in case:
            case["root"] = str((args.cases.parent / case["root"]).resolve())
            p = Path(case["root"]) / case["reference"]
            assert (
                hashlib.sha256(p.read_bytes()).hexdigest() == case["reference_sha256"]
            )
            assert (
                hashlib.sha256(
                    json.dumps(case["payload"], sort_keys=True).encode()
                ).hexdigest()
                == case["payload_sha256"]
            )
        else:
            p = args.fixtures / case["file"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == case["audio_sha256"]
            assert case["data"]["language"] == case["language"]
    serving = {}
    for name in sorted({c["service"] for c in cases}):
        state = json.loads((Path("/workspace/services") / (name + ".json")).read_text())
        launch = json.loads(Path(state["launch_file"]).read_text())
        assert launch["passed"], name
        config = Path(launch["argv"][launch["argv"].index("--config") + 1])
        assert (
            hashlib.sha256(config.read_bytes()).hexdigest() == launch["config_sha256"]
        )
        serving[name] = launch
    (args.out / "protocol.json").write_text(
        json.dumps(
            {
                **vars(args),
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "started_unix": time.time(),
                "serving": serving,
                "model_pins": json.loads(Path("/app/models.json").read_text()),
            },
            default=str,
            indent=2,
        )
        + "\n"
    )
    headers = (
        {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
        if os.getenv("VLLM_API_KEY")
        else {}
    )
    client = httpx.Client(
        timeout=1200,
        headers=headers,
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
    )
    lock = threading.Lock()
    records = []
    actual_bank = {}

    def request(case, cell, scheduled=None, barrier=None):
        payload = copy.deepcopy(case.get("payload"))
        if payload:
            for message in payload.get("messages", []):
                for part in message["content"]:
                    if part["type"] == "image_url":
                        part["uuid"] = uuid4().hex
        if barrier:
            barrier.wait(timeout=60)
        start = time.time()
        start_mono = time.monotonic()
        row = dict(
            service=case["service"],
            case=case["id"],
            cell=cell,
            started_unix=start,
            started_mono_s=start_mono,
            items=case["items"],
            tokens=case.get("tokens", 0),
            images=case.get("images", 0),
            audio_s=case.get("duration_s", 0),
            fresh_image_ids=bool(case.get("images")),
            client_launch_delay_s=max(0, start_mono - scheduled) if scheduled else 0,
            http_success=False,
            passed=False,
        )
        content = b""
        try:
            url = f"http://127.0.0.1:{PORTS[case['service']]}{case['endpoint']}"
            if "file" in case:
                with (args.fixtures / case["file"]).open("rb") as f:
                    response = client.post(
                        url,
                        data=case["data"],
                        files={"file": (Path(case["file"]).name, f)},
                    )
            else:
                response = client.post(url, json=payload)
            content = response.content
            row["status"] = response.status_code
            row["response_bytes"] = len(content)
            response.raise_for_status()
            row["http_success"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["ended_unix"] = time.time()
        row["ended_mono_s"] = time.monotonic()
        row["latency_s"] = row["ended_mono_s"] - start_mono
        return case, row, content

    def validate(result):
        case, row, content = result
        start = time.monotonic()
        try:
            if not row["http_success"]:
                raise ValueError(row["error"])
            body = json.loads(content)
            if case["service"] == "whisper":
                actual = score(case["ground_truth"], body["text"])
                expected = score(case["ground_truth"], case["reference_text"])
                row.update(
                    actual=actual,
                    reference=expected,
                    text=body["text"],
                    language=case["language"],
                    real_time_factor=row["latency_s"] / case["duration_s"],
                )
                allowance = 0.02 * actual["words"] if case["duration_s"] > 30 else 1
                assert actual["errors"] <= expected["errors"] + allowance, (
                    "ASR exceeds reference error allowance"
                )
                if case["reference_text"].strip():
                    assert body["text"].strip(), "Empty speech output"
                if not case["ground_truth"]:
                    assert actual["errors"] == 0, "Silence hallucination"
            else:
                row.update(check_vectors(case, body))
                usage = body.get("usage") or {}
                row["reported_prompt_tokens"] = usage.get("prompt_tokens")
                if case["service"] == "jina" and args.mode == "quality":
                    actual_bank[case["id"]] = np.asarray(
                        body["data"][0]["data"], dtype=np.float32
                    )
                    np.savez_compressed(
                        args.out / (case["id"] + ".npz"), actual=actual_bank[case["id"]]
                    )
            row["passed"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if content:
                with gzip.GzipFile(
                    filename=str(args.out / (case["id"] + "-failure.json.gz")),
                    mode="wb",
                    mtime=0,
                ) as stream:
                    stream.write(content)
        row["validation_s"] = time.monotonic() - start
        with lock:
            records.append(row)
            with (args.out / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    summaries = []
    ranking_passed = True
    asr_passed = True

    def save(summary):
        summaries.append(summary)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)

    if args.mode == "quality":
        for case in cases:
            row = validate(request(case, "quality"))
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in ("service", "case", "passed", "error", "latency_s")
                    }
                ),
                flush=True,
            )
        jina = {c["id"]: c for c in cases if c["service"] == "jina"}
        rankings = []
        documents = [n for n in jina if not n.startswith("query-")]
        for query in (n for n in jina if n.startswith("query-")):
            if len(documents) < 2 or query not in actual_bank:
                continue
            if not all(n in actual_bank for n in documents):
                continue

            def maxsim(a, b):
                return float((a @ b.T).max(axis=1).sum())

            q = reference(str(Path(jina[query]["root"]) / jina[query]["reference"]))
            expected = [
                maxsim(q, reference(str(Path(jina[n]["root"]) / jina[n]["reference"])))
                for n in documents
            ]
            actual = [maxsim(actual_bank[query], actual_bank[n]) for n in documents]
            rankings.append(
                dict(
                    query=query,
                    documents=documents,
                    actual=actual,
                    reference=expected,
                    top1_agrees=int(np.argmax(actual)) == int(np.argmax(expected)),
                )
            )
        (args.out / "maxsim.json").write_text(json.dumps(rankings, indent=2) + "\n")
        ranking_passed = all(r["top1_agrees"] for r in rankings)
        by_id = {c["id"]: c for c in cases}
        corpus = []
        for language in ("en", "ar"):
            expected_cases = [
                c
                for c in cases
                if c["service"] == "whisper"
                and c["kind"] == "short"
                and c["language"] == language
            ]
            rows = [
                r
                for r in records
                if r["service"] == "whisper"
                and "actual" in r
                and by_id[r["case"]]["kind"] == "short"
                and r["language"] == language
            ]
            if not expected_cases:
                continue
            words = sum(r["actual"]["words"] for r in rows)
            if not words:
                corpus.append(dict(language=language, passed=False, clips=0))
                asr_passed = False
                continue
            actual = sum(r["actual"]["errors"] for r in rows) / words
            expected = sum(r["reference"]["errors"] for r in rows) / words
            passed = len(rows) == len(expected_cases) and actual <= expected + 0.005
            asr_passed &= passed
            corpus.append(
                dict(
                    language=language,
                    clips=len(rows),
                    words=words,
                    actual_wer=actual,
                    reference_wer=expected,
                    passed=passed,
                    delta_95ci=paired_interval(
                        [
                            dict(
                                prompt_group=by_id[r["case"]]["prompt_group"],
                                actual=r["actual"],
                                reference=r["reference"],
                            )
                            for r in rows
                        ]
                    ),
                )
            )
        (args.out / "asr-corpus.json").write_text(json.dumps(corpus, indent=2) + "\n")
        save(
            dict(
                maxsim_passed=ranking_passed,
                asr_corpus_passed=asr_passed,
                **summarize(
                    records,
                    min(r["started_mono_s"] for r in records),
                    max(r["ended_mono_s"] for r in records),
                ),
            )
        )
    elif args.mode == "sweep":
        groups = (
            [("mixed", cases)] if args.mixed_cohort else [(c["id"], [c]) for c in cases]
        )
        for name, group in groups:
            for concurrency in args.concurrency:
                rows = []
                spans = []
                for repeat in range(
                    math.ceil(max(args.min_requests, 2 * concurrency) / concurrency)
                ):
                    barrier = threading.Barrier(concurrency)
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = [
                            pool.submit(
                                request,
                                group[(repeat * concurrency + i) % len(group)],
                                f"{name}-c{concurrency}",
                                None,
                                barrier,
                            )
                            for i in range(concurrency)
                        ]
                        results = [f.result() for f in futures]
                    spans.append(
                        max(r[1]["ended_mono_s"] for r in results)
                        - min(r[1]["started_mono_s"] for r in results)
                    )
                    rows.extend(validate(result) for result in results)
                    if not all(r["http_success"] for r in rows):
                        break
                save(
                    dict(
                        case=name,
                        concurrency=concurrency,
                        cohorts=len(spans),
                        **summarize(rows, 0, sum(spans)),
                    )
                )
                if not all(r["http_success"] for r in rows):
                    return 2
    else:
        assert args.minutes > 0 and args.rpm > 0
        groups = {
            name: [c for c in cases if c["service"] == name] for name in args.services
        }
        assert all(groups.values())
        start = time.monotonic()
        futures = []
        with ThreadPoolExecutor(max_workers=128) as pool:
            for i in range(round(args.minutes * args.rpm)):
                scheduled = start + i * 60 / args.rpm
                time.sleep(max(0, scheduled - time.monotonic()))
                name = args.services[i % len(args.services)]
                case = groups[name][(i // len(args.services)) % len(groups[name])]
                futures.append(
                    pool.submit(
                        lambda c, t: validate(request(c, "mixed", t)), case, scheduled
                    )
                )
                if i % 30 == 0:
                    print(
                        json.dumps(
                            dict(
                                submitted=i + 1,
                                completed=sum(f.done() for f in futures),
                            )
                        ),
                        flush=True,
                    )
            for f in futures:
                f.result()
        end = time.monotonic()
        save(
            dict(
                offered_rpm=args.rpm,
                offered_minutes=args.minutes,
                drain_s=max(0, end - start - args.minutes * 60),
                **summarize(records, start, end),
            )
        )
        for name in args.services:
            rows = [r for r in records if r["service"] == name]
            window = [
                r
                for r in rows
                if start + 120 <= r["ended_mono_s"] <= start + args.minutes * 60
            ]
            save(
                dict(
                    service=name,
                    steady_completed_rpm=sum(r["passed"] for r in window)
                    / max(args.minutes - 2, 1e-9),
                    **summarize(rows, start, end),
                )
            )
    client.close()
    return (
        0 if all(r["passed"] for r in records) and ranking_passed and asr_passed else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
