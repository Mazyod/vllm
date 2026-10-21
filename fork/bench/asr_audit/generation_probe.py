"""Check the observed quiet-clip failure against initial EOS suppression.

This is a configuration diagnostic, not a change to benchmark inputs or the
published serving image. Silence controls expose the hallucination tradeoff.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    cases = json.loads((args.fixtures / "manifest.json").read_text())["cases"]
    rows = []
    with torch.inference_mode():
        for case in cases:
            if case["id"] not in ("en-short-092", "en-silence", "ar-silence"):
                continue
            audio, sr = sf.read(args.fixtures / case["file"], dtype="float32")
            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                return_attention_mask=True,
            ).to("cuda")
            for suppressed in ([220], [220, 50257]):
                output = model.generate(
                    **inputs,
                    language=case["language"],
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    return_timestamps=False,
                    max_length=448,
                    begin_suppress_tokens=suppressed,
                )
                row = dict(
                    id=case["id"],
                    audio_sha256=case["sha256"],
                    language=case["language"],
                    begin_suppress_tokens=suppressed,
                    text=processor.batch_decode(output, skip_special_tokens=True)[0],
                    token_ids=output.cpu().tolist(),
                )
                rows.append(row)
                args.out.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                )
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
