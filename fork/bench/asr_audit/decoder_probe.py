"""Retain raw Whisper decoder tokens to diagnose empty timestamped responses.

Uses the published image's unchanged engine and segment parser. Run after the
HTTP server/reference have released the GPU. This is diagnostic, not a new
serving implementation.
"""

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
from fastapi import UploadFile
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
)
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.fixtures / "manifest.json").read_text())["cases"]
    cases = [
        c
        for c in cases
        if c["id"] in ("en-short-000", "ar-short-000", "en-silence", "ar-silence")
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=str(args.model),
        dtype="half",
        max_model_len=448,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
    )
    rows = []
    for case in cases:
        audio, sr = sf.read(args.fixtures / case["file"], dtype="float32")
        for timestamps in (False, True):
            prefix = f"<|startoftranscript|><|{case['language']}|><|transcribe|>" + (
                "<|0.00|>" if timestamps else "<|notimestamps|>"
            )
            prompt = ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="", multi_modal_data={"audio": (audio, sr)}
                ),
                decoder_prompt=TextPrompt(prompt=prefix),
            )
            output = llm.generate(
                [prompt], SamplingParams(temperature=0, max_tokens=444, logprobs=1)
            )[0].outputs[0]
            row = dict(
                id=case["id"],
                timestamps=timestamps,
                prompt=prefix,
                token_ids=output.token_ids,
                text=output.text,
                decoded_with_special_tokens=tokenizer.decode(
                    output.token_ids,
                    skip_special_tokens=False,
                    decode_with_timestamps=True,
                ),
                generated_timestamp_tokens=sum(
                    token >= tokenizer.convert_tokens_to_ids("<|0.00|>")
                    for token in output.token_ids
                ),
                finish_reason=output.finish_reason,
            )
            if timestamps:
                request = TranscriptionRequest(
                    file=UploadFile(
                        file=io.BytesIO((args.fixtures / case["file"]).read_bytes()),
                        filename=case["file"],
                    ),
                    model="bbyn-v2",
                    language=case["language"],
                    temperature=0,
                    response_format="verbose_json",
                )
                try:
                    segments = SpeechToTextBaseServing._get_verbose_segments(
                        SimpleNamespace(tokenizer=tokenizer),
                        tokens=tuple(output.token_ids),
                        log_probs=output.logprobs,
                        request=request,
                        segment_class=TranscriptionSegment,
                    )
                    row["parsed_segments"] = [s.model_dump() for s in segments]
                except Exception as exc:
                    row["parser_error"] = repr(exc)
            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
