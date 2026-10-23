"""Replay saved numerical outputs and verify failure accounting before a rental."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from benchmark import check_vectors, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    checked = 0
    for case in json.loads(args.cases.read_text()):
        if "root" not in case:
            continue
        case["root"] = str(args.cases.parent / case["root"])
        with np.load(Path(case["root"]) / case["reference"]) as saved:
            actual = saved["actual"]
        if case["service"] == "bge":
            body = {
                "results": [
                    {"index": i, "relevance_score": float(value)}
                    for i, value in enumerate(actual)
                ]
            }
            bad = copy.deepcopy(body)
            bad["results"][0]["relevance_score"] = 1000
        elif case["service"] == "jina":
            body = {"data": [{"index": 0, "data": actual.tolist()}]}
            bad = {"data": [{"index": 0, "data": np.zeros_like(actual).tolist()}]}
        else:
            body = {
                "data": [
                    {"index": i, "embedding": value.tolist()}
                    for i, value in enumerate(actual)
                ]
            }
            bad = copy.deepcopy(body)
            bad["data"][0]["embedding"] = [0] * len(actual[0])
        check_vectors(case, body)
        try:
            check_vectors(case, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Invalid output accepted: {case['id']}")
        checked += 1
    assert checked == 21
    rows = [
        dict(passed=True, http_success=True, latency_s=2, items=8, tokens=64),
        dict(passed=False, http_success=True, latency_s=4, items=8, tokens=64),
    ]
    result = summarize(rows, 10, 14)
    assert result["successful_requests_s"] == 0.25
    assert result["items_s"] == 2 and result["input_tokens_s"] == 16
    assert result["http_successes"] == 2 and result["quality_successes"] == 1
    assert result["p50_s"] == 3 and result["p95_s"] is None
    print(
        f"{checked} reference replays and corrupt-output rejections passed; "
        "timing accounting passed"
    )


if __name__ == "__main__":
    main()
