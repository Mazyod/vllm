"""Bind the campaign to previously computed independent FP32 reference outputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--jina", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--asr-fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], {}
    for label, root in (("text", args.text), ("jina", args.jina)):
        report = next(root.glob("*-quality.json"))
        data = json.loads(report.read_text())
        assert data["passed"], report
        target = args.out / label
        target.mkdir(exist_ok=True)
        provenance[label] = {"report_sha256": digest(report), "files": {}}
        for line in (root / "requests.jsonl").read_text().splitlines():
            case = json.loads(line)
            reference = root / case["reference"]
            shutil.copyfile(reference, target / reference.name)
            case["root"] = label
            case["reference_sha256"] = digest(reference)
            provenance[label]["files"][reference.name] = digest(reference)
            case["payload_sha256"] = hashlib.sha256(
                json.dumps(case["payload"], sort_keys=True).encode()
            ).hexdigest()
            if label == "text" and not case["id"].endswith("-batch"):
                recorded = next(
                    c
                    for c in data["cases"]
                    if c["service"] == case["service"]
                    and c.get("requested_budget") == case["tokens"]
                )
                assert case["payload_sha256"] == recorded["input_sha256"]
            if label == "jina":
                recorded = next(c for c in data["cases"] if c["id"] == case["id"])
                assert case["payload_sha256"] == recorded["request_sha256"]
                case["images"] = sum(
                    p["type"] == "image_url"
                    for m in case["payload"].get("messages", [])
                    for p in m["content"]
                )
            case["items"] = (
                len(case["payload"]["documents"])
                if case["service"] == "bge"
                else len(np.load(reference)["reference"])
                if label == "text"
                else 1
            )
            cases.append(case)
    manifest = json.loads((args.asr_fixtures / "manifest.json").read_text())
    reference_provenance = json.loads((args.asr / "provenance.json").read_text())
    assert reference_provenance["manifest_sha256"] == digest(
        args.asr_fixtures / "manifest.json"
    )
    assert (
        reference_provenance["model_revision"]
        == "e910dbbc381deab0561e944c7149c3ace41b6bff"
    )
    refs = {
        (r["id"], r["variant"]): r
        for r in map(
            json.loads, (args.asr / "responses.jsonl").read_text().splitlines()
        )
    }
    for item in manifest["cases"]:
        ref = refs[item["id"], "json"]
        assert "error" not in ref and ref["audio_sha256"] == item["sha256"]
        cases.append(
            dict(
                id=item["id"],
                service="whisper",
                endpoint="/v1/audio/transcriptions",
                file="asr/" + item["file"],
                audio_sha256=item["sha256"],
                data=dict(
                    model="whisper",
                    language=item["language"],
                    temperature="0",
                    response_format="json",
                    max_completion_tokens="444",
                ),
                language=item["language"],
                duration_s=item["duration_s"],
                kind=item["kind"],
                prompt_group=item["dataset_row_ids"][0]
                if item["dataset_row_ids"]
                else item["id"],
                ground_truth=item["text"],
                reference_text=ref["response"]["text"],
                items=1,
            )
        )
    provenance["asr"] = reference_provenance
    provenance["asr"]["responses_sha256"] = digest(args.asr / "responses.jsonl")
    (args.out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {"cases": len(cases), "services": sorted({c["service"] for c in cases})}
        )
    )


if __name__ == "__main__":
    main()
