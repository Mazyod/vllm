"""Retain token outliers and cross-precision MaxSim sensitivity from saved outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    actual, reference, details = {}, {}, []
    for case in json.loads(args.cases.read_text()):
        if case["service"] != "jina":
            continue
        path = args.cases.parent / case["root"] / case["reference"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["reference_sha256"]
        with np.load(path) as data:
            ref = data["reference"]
        with np.load(args.quality / (case["id"] + ".npz")) as data:
            served = data["actual"]
        assert served.shape == ref.shape
        cosine = (served * ref).sum(-1) / (
            np.linalg.norm(served, axis=-1) * np.linalg.norm(ref, axis=-1)
        )
        actual[case["id"]], reference[case["id"]] = served, ref
        details.append(
            dict(
                case=case["id"],
                rows=len(cosine),
                cosine_mean=float(cosine.mean()),
                cosine_min=float(cosine.min()),
                cosine_p01=float(np.quantile(cosine, 0.01)),
                rows_below_0_9=int((cosine < 0.9).sum()),
                negative_cosines=int((cosine < 0).sum()),
            )
        )
    docs = [name for name in actual if not name.startswith("query-")]
    rankings = []
    for query in (name for name in actual if name.startswith("query-")):
        expected = np.array(
            [(reference[query] @ reference[d].T).max(axis=1).sum() for d in docs]
        )
        expected_order = np.argsort(-expected).tolist()
        for name, queries, documents in (
            ("served_query_served_documents", actual, actual),
            ("served_query_fp32_documents", actual, reference),
            ("fp32_query_served_documents", reference, actual),
        ):
            scores = np.array(
                [(queries[query] @ documents[d].T).max(axis=1).sum() for d in docs]
            )
            order = np.argsort(-scores).tolist()
            rankings.append(
                dict(
                    query=query,
                    mode=name,
                    reference_order=[docs[i] for i in expected_order],
                    actual_order=[docs[i] for i in order],
                    top1_agrees=order[0] == expected_order[0],
                    full_order_agrees=order == expected_order,
                    max_score_difference=float(np.abs(scores - expected).max()),
                )
            )
    args.out.write_text(
        json.dumps(
            dict(
                token_fidelity=details,
                rankings=rankings,
                scope=(
                    "Numerical sensitivity on three queries and six documents; "
                    "not a labeled relevance benchmark."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                comparisons=len(rankings),
                matching_top1=sum(r["top1_agrees"] for r in rankings),
                matching_full_order=sum(r["full_order_agrees"] for r in rankings),
            )
        )
    )


if __name__ == "__main__":
    main()
