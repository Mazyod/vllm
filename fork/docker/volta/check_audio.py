# ruff: noqa: INP001
"""Exercise the frozen image's decoding, resampling and TorchAudio linkage."""

from io import BytesIO

import av
import numpy as np
import soundfile
import torch
import torchaudio
from vllm.multimodal.audio import AudioResampler
from vllm.multimodal.media.audio import load_audio

assert torch.__version__ == "2.13.0+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
wav = BytesIO()
soundfile.write(wav, np.zeros(8000, dtype=np.float32), 8000, format="WAV")
for backend in ("auto", "soundfile", "pyav"):
    audio, rate = load_audio(BytesIO(wav.getvalue()), sr=None, backend=backend)
    assert rate == 8000 and audio.shape == (8000,), backend
    resampled = AudioResampler(target_sr=16000).resample(audio, orig_sr=rate)
    assert resampled.shape == (16000,) and np.isfinite(resampled).all(), backend
waveform, rate = torchaudio.load(BytesIO(wav.getvalue()))
assert rate == 8000 and waveform.shape == (1, 8000)
print(
    "audio decode/resample OK:",
    av.__version__,
    soundfile.__version__,
    torchaudio.__version__,
)
