"""Small real-HTTP acceptance workload for V100 auxiliary services.

This checks semantic ordering, vector shape/normalization, and a known speech
fixture. It is a smoke workload, not a representative quality benchmark.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Mars is known as the Red Planet because of its reddish appearance.",
    "Pasta is made from wheat flour and water.",
    "The Pacific Ocean is the largest ocean on Earth.",
]
TRANSCRIPT = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)
PORTS = {"snowflake": 8001, "bge": 8002, "whisper": 8003, "gemma": 8004}


def embedding_inputs(name):
    if name == "gemma":
        return ["task: search result | query: " + QUERY] + [
            "title: none | text: " + d for d in DOCUMENTS
        ]
    return ["query: " + QUERY] + DOCUMENTS


def request(name, audio):
    started = time.perf_counter()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{PORTS[name]}", timeout=180
    ) as client:
        if name in ("snowflake", "gemma"):
            r = client.post(
                "/v1/embeddings", json={"model": name, "input": embedding_inputs(name)}
            )
            r.raise_for_status()
            vectors = np.array(
                [
                    d["embedding"]
                    for d in sorted(r.json()["data"], key=lambda d: d["index"])
                ]
            )
            assert vectors.shape == (4, 768 if name == "gemma" else 1024), vectors.shape
            assert np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.005)
            scores = vectors[1:] @ vectors[0]
            assert scores.argmax() == 0, scores.tolist()
            result = {"similarities": scores.tolist(), "vectors": vectors.tolist()}
        elif name == "bge":
            r = client.post(
                "/rerank", json={"model": name, "query": QUERY, "documents": DOCUMENTS}
            )
            r.raise_for_status()
            results = r.json()["results"]
            assert sorted(d["index"] for d in results) == [0, 1, 2]
            assert results[0]["index"] == 0, results
            assert all(np.isfinite(d["relevance_score"]) for d in results)
            result = {
                "scores": [
                    d["relevance_score"]
                    for d in sorted(results, key=lambda d: d["index"])
                ]
            }
        else:
            with audio.open("rb") as f:
                r = client.post(
                    "/v1/audio/transcriptions",
                    data={"model": name, "language": "en", "temperature": "0"},
                    files={"file": (audio.name, f, "audio/flac")},
                )
            r.raise_for_status()
            text = r.json()["text"]
            words = re.findall(r"[a-z]+", text.lower())
            # Exact known transcript; no fuzzy success based only on nonempty text.
            assert words == TRANSCRIPT.split(), text
            result = {"text": text}
    result["elapsed_s"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", nargs="+", choices=list(PORTS), required=True)
    parser.add_argument("--audio", type=Path, default=Path("/workspace/jfk.flac"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.repeats > 0
    report = {"services": {}, "passed": True, "repeats": args.repeats}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(args.services)) as executor:
        for repeat in range(args.repeats):
            jobs = {
                name: executor.submit(request, name, args.audio)
                for name in args.services
            }
            for name, job in jobs.items():
                entry = report["services"].setdefault(name, {"latencies_s": []})
                try:
                    result = job.result()
                    entry["latencies_s"].append(result["elapsed_s"])
                    if repeat == 0:
                        entry["first_response"] = result
                except Exception as exc:
                    entry.setdefault("errors", []).append(str(exc))
                    report["passed"] = False
    report["wall_s"] = time.perf_counter() - started
    for entry in report["services"].values():
        warm = entry["latencies_s"][1:]
        if warm:
            entry["warm_p50_s"], entry["warm_p95_s"] = np.percentile(
                warm, [50, 95]
            ).tolist()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "wall_s": report["wall_s"],
                "services": {
                    k: {a: b for a, b in v.items() if a != "first_response"}
                    for k, v in report["services"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
