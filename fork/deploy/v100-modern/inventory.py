"""Inspect a built runtime without a GPU; emit package, codec and source evidence."""

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import av
from torchcodec.decoders import AudioDecoder


def main():
    assert os.getuid() == 10001
    assert importlib.util.find_spec("PyNvVideoCodec") is None
    assert importlib.util.find_spec("cv2") is None
    audio = AudioDecoder("/app/warmup.flac").get_all_samples()
    records = subprocess.check_output(
        ["dpkg-query", "-W", "-f", "${binary:Package}\t${Version}\n"], text=True
    )
    packages = dict(line.split("\t", 1) for line in records.splitlines())
    assert not any(
        name.startswith(
            (
                "ffmpeg",
                "libavcodec",
                "libavformat",
                "libavutil",
                "libavdevice",
                "libavfilter",
            )
        )
        for name in packages
    )
    headers = subprocess.check_output(
        ["dpkg-query", "-L", "linux-libc-dev"], text=True
    ).splitlines()
    header_files = [Path(p) for p in headers if Path(p).is_file()]
    elf = []
    for path in header_files:
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                elf.append(str(path))
    assert not elf and not any(p.startswith("linux-image") for p in packages)
    root = Path("/opt/vllm/lib/python3.12/site-packages")
    files = [
        *Path("/app").glob("*.py"),
        Path("/app/entrypoint.sh"),
        *Path("/app/engine").glob("*.yaml"),
        root / "vllm_volta.py",
        *root.joinpath("vllm").glob("*.so"),
        root / "nvidia/cuda_nvcc/bin/ptxas",
    ]
    with tempfile.TemporaryDirectory(dir="/data/cache") as directory:
        temp = Path(directory)
        source = temp / "probe.c"
        source.write_text(
            "#include <Python.h>\nint readiness_probe(void) { return 42; }\n"
        )
        library = temp / "probe.so"
        subprocess.run(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-I" + sysconfig.get_paths()["include"],
                str(source),
                "-o",
                str(library),
            ],
            check=True,
        )
        assert ctypes.CDLL(str(library)).readiness_probe() == 42
        ptx = temp / "probe.ptx"
        ptx.write_text(
            ".version 7.0\n.target sm_70\n.address_size 64\n"
            ".visible .entry readiness_probe() { ret; }\n"
        )
        subprocess.run(
            [
                str(root / "nvidia/cuda_nvcc/bin/ptxas"),
                "-arch=sm_70",
                str(ptx),
                "-o",
                str(temp / "probe.cubin"),
            ],
            check=True,
        )
        assert (temp / "probe.cubin").stat().st_size > 0
    report = {
        "uid": os.getuid(),
        "nonroot_compiler_and_sm70_assembler_passed": True,
        "ffmpeg": av.ffmpeg_version_info,
        "ffmpeg_library_versions": av.library_versions,
        "audio_samples": list(audio.data.shape),
        "sample_rate": audio.sample_rate,
        "video_decoder_removed": True,
        "allocator": os.getenv("PYTORCH_ALLOC_CONF"),
        "python_packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "os_packages": packages,
        "linux_headers": {
            "package_version": packages["linux-libc-dev:amd64"]
            if "linux-libc-dev:amd64" in packages
            else packages["linux-libc-dev"],
            "files": len(header_files),
            "elf_files": elf,
            "kernel_image_packages": [],
        },
        "source_and_native_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
