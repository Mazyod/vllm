"""Score retained transcripts against human text and paired FP32 references.

Short-clip confidence intervals resample source sentence clusters, not words. Derived
long/shifted inputs are reported separately and never inflate sample counts.
"""

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(
        "".join(
            " " if unicodedata.category(c).startswith("P") else c
            for c in text
            if not unicodedata.category(c).startswith("M")
        ).split()
    )


def score(truth, text):
    reference, hypothesis = normalize(truth), normalize(text)
    words = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    return dict(
        words=words.hits + words.substitutions + words.deletions,
        substitutions=words.substitutions,
        deletions=words.deletions,
        insertions=words.insertions,
        wer=words.wer,
        cer=chars.cer,
        errors=words.substitutions + words.deletions + words.insertions,
    )


def paired_interval(rows):
    rng = random.Random(20261021)
    clusters = {}
    for row in rows:
        clusters.setdefault(row["prompt_group"], []).append(row)
    groups = list(clusters.values())
    deltas = []
    for _ in range(2000):
        sample = [r for group in rng.choices(groups, k=len(groups)) for r in group]
        deltas.append(
            sum(r["actual"]["errors"] - r["reference"]["errors"] for r in sample)
            / sum(r["actual"]["words"] for r in sample)
        )
    deltas.sort()
    return [deltas[49], deltas[1949]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # One independent, known-answer check on metric accounting and Unicode handling.
    assert score("one two three", "one four")["errors"] == 2
    assert normalize("السَّلَام، عليكم!") == "السلام عليكم"
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    server_provenance = json.loads((args.server / "provenance.json").read_text())
    reference_provenance = json.loads((args.reference / "provenance.json").read_text())
    assert server_provenance["model_files"] == reference_provenance["model_files"]
    assert (
        server_provenance["manifest_sha256"] == reference_provenance["manifest_sha256"]
    )
    assert (
        hashlib.sha256((args.fixtures / "manifest.json").read_bytes()).hexdigest()
        == server_provenance["manifest_sha256"]
    )
    references = {
        (r["id"], r["variant"]): r
        for r in map(
            json.loads, (args.reference / "responses.jsonl").read_text().splitlines()
        )
    }
    rows = []
    failures = []
    observed = []
    for served in map(
        json.loads, (args.server / "responses.jsonl").read_text().splitlines()
    ):
        observed.append((served["id"], served["variant"]))
        case = cases[served["id"]]
        ref = references.get((case["id"], served["variant"]))
        if "error" in served or not ref or "error" in ref:
            failures.append(
                dict(
                    id=case["id"],
                    variant=served["variant"],
                    server_error=served.get("error"),
                    reference_missing_or_error=not ref or ref.get("error"),
                )
            )
            continue
        assert served["audio_sha256"] == ref["audio_sha256"] == case["sha256"]
        assert served["request"]["language"] == case["language"]
        text, reference_text = served["response"]["text"], ref["response"]["text"]
        row = dict(
            id=case["id"],
            prompt_group=case["dataset_row_ids"][0]
            if case["dataset_row_ids"]
            else case["id"],
            kind=case["kind"],
            language=case["language"],
            variant=served["variant"],
            duration_s=case["duration_s"],
            actual=score(case["text"], text),
            reference=score(case["text"], reference_text),
            text=text,
            reference_text=reference_text,
            ground_truth=case["text"],
            elapsed_s=served["elapsed_s"],
        )
        native = references.get((case["id"], "native_long"))
        if native and "error" not in native:
            row["native_long_reference"] = score(
                case["text"], native["response"]["text"]
            )
            row["native_long_reference_text"] = native["response"]["text"]
        rows.append(row)
    corpus = []
    for language in ("en", "ar"):
        for variant in ("json", "verbose_json"):
            selected = [
                r
                for r in rows
                if r["kind"] == "short"
                and r["language"] == language
                and r["variant"] == variant
            ]
            if not selected:
                continue
            count = sum(r["actual"]["words"] for r in selected)
            summary = dict(
                language=language,
                variant=variant,
                clips=len(selected),
                distinct_prompts=len({r["prompt_group"] for r in selected}),
                words=count,
                actual_wer=sum(r["actual"]["errors"] for r in selected) / count,
                reference_wer=sum(r["reference"]["errors"] for r in selected) / count,
                actual_cer=jiwer.cer(
                    [normalize(r["ground_truth"]) for r in selected],
                    [normalize(r["text"]) for r in selected],
                ),
                reference_cer=jiwer.cer(
                    [normalize(r["ground_truth"]) for r in selected],
                    [normalize(r["reference_text"]) for r in selected],
                ),
                actual_word_errors={
                    k: sum(r["actual"][k] for r in selected)
                    for k in ("substitutions", "deletions", "insertions")
                },
                paired_delta_95ci=paired_interval(selected),
            )
            if language == "ar":
                # Sensitivity analysis only: these folds can erase real distinctions.
                fold = str.maketrans(
                    {
                        "أ": "ا",
                        "إ": "ا",
                        "آ": "ا",
                        "ٱ": "ا",
                        "ى": "ي",
                        "ة": "ه",
                        "ـ": None,
                    }
                )
                truth = [normalize(r["ground_truth"]).translate(fold) for r in selected]
                summary["secondary_lossy_orthographic_fold_wer"] = {
                    "actual": jiwer.wer(
                        truth, [normalize(r["text"]).translate(fold) for r in selected]
                    ),
                    "reference": jiwer.wer(
                        truth,
                        [
                            normalize(r["reference_text"]).translate(fold)
                            for r in selected
                        ],
                    ),
                }
            corpus.append(summary)
    formats = ["json"] if server_provenance["stream"] else ["json", "verbose_json"]
    expected = {
        (identifier, fmt)
        for identifier in server_provenance["case_ids"]
        for fmt in formats
    }
    expected_reference = expected | {
        (identifier, "native_long")
        for identifier in server_provenance["case_ids"]
        if cases[identifier]["duration_s"] > 30
    }
    report = dict(
        execution_complete=set(observed) == expected and len(observed) == len(expected),
        reference_execution_complete=expected_reference.issubset(references),
        expected_responses=len(expected),
        recorded_responses=len(observed),
        missing_responses=sorted(expected - set(observed)),
        reference_runtime_errors=[
            {"id": r["id"], "variant": r["variant"], "error": r["error"]}
            for r in references.values()
            if "error" in r
        ],
        corpus=corpus,
        errors=failures,
        cases=rows,
        normalization=(
            "NFKC, lowercase, punctuation to spaces, remove combining marks; "
            "no Arabic letter folding or English number expansion"
        ),
        model_revision=server_provenance["model_revision"],
        limitation=(
            "Public read speech; concatenated long cases are not natural "
            "conversations. Training overlap and production-domain accuracy "
            "are unknown."
        ),
    )
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(corpus=corpus, errors=len(failures)), indent=2))


if __name__ == "__main__":
    main()
