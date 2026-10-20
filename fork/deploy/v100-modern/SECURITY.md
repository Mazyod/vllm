# V100 image security review

The image is not a Prisma approval. Scan the final deployed digest against the
organization's Prisma policy. No console endpoint or scan credentials were
provided for this work. The image inventories and unfiltered scanner results
are retained with the validation evidence.

## Changes made because of findings

- Use a patched Ubuntu 24.04 runtime and hash-pinned Python dependencies.
- Permit setuptools 83.0.0, fixing CVE-2026-59890 while retaining the upstream
  build toolchain in the builder.
- Omit the unused PyNvVideoCodec 2.0.4 and OpenCV video-I/O requirements from the vLLM wheel metadata
  and remove both packages before constructing the final runtime layer. Their
  bundled FFmpeg 8.1.1 produced 22 high-severity matches. These services do not
  use GPU video decoding; Jina serves text/images and Whisper decodes uploaded
  audio with TorchCodec/faster-whisper.
- Exclude the distro FFmpeg installation from the final stage. TorchCodec and
  PyAV share PyAV 19.0.1's bundled FFmpeg 9.0.2 libraries through explicit
  SONAME aliases. Runtime decoding and the loaded library paths were checked.
- Run as UID 10001, require bearer authentication, mount models read-only, drop
  capabilities, use `no-new-privileges`, and keep writable compilation caches
  outside the read-only root filesystem.

The native wheel's model code and native libraries are unchanged by the video
metadata update: only `METADATA` and `RECORD` differ. The exact dependency diff
and wheel checksum are shipped in the image. This is an auditable packaging
change, not a hidden scanner suppression.

## Kernel-header findings

`linux-libc-dev` supplies userspace API headers needed by the runtime compiler.
It does not supply the running Linux kernel. The inventory checks its file
list for ELF binaries and checks that no `linux-image` package is installed.
The host kernel and NVIDIA driver need their own security lifecycle.

Trivy associates many source-kernel advisories with this header package.
Grype has a built-in rule excluding this indirect match; its
[implementation](https://github.com/anchore/grype/blob/v0.121.0/cmd/grype/cli/commands/root.go)
and the [scanner discussion](https://github.com/aquasecurity/trivy/discussions/6562)
explain the distinction. Retain the raw Trivy findings for Prisma applicability
review rather than representing them as a clean raw scan. No custom ignore
entries are supplied to either scanner.

## Remaining FFmpeg RTSP advisory

CVE-2026-107696 concerns an RTSP redirect loop. FFmpeg 9.0.2 is the
[current stable release](https://ffmpeg.org/download.html); the scanner lists
no fixed release for this advisory. Its
[upstream change](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/24902) is tracked
separately from the image's model compatibility work.

The ASR endpoint passes uploaded bytes to the decoder, never a user-supplied
URL or filename. Jina's tested configuration accepts images and has video
inputs disabled. `asr_contract.py` checks that RTSP bytes and image/video URL
requests are rejected, that an RTSP-looking upload filename is treated only as
a filename, and that these tests make no connection to a local RTSP listener.

This is evidence about the exposed API surface, not a claim that the bundled
library has been patched. Keep the finding visible and have the Prisma policy
owner review its applicability. Reassess it before enabling URL-based audio,
video or RTSP decoding.

## Final scan evidence

The final image digest and exact counts are in
[results/security.json](results/security.json). Grype reports one high finding
(CVE-2026-107696), zero critical findings, and no available fixed versions in
its matches. Trivy reports 11 critical and 298 high source-kernel matches,
all on the header package; outside that package it reports 87 medium and 21
low findings, with no available Ubuntu fixes. Public Python/Rust advisory scans
reported no vulnerabilities.

The complete, unfiltered reports are compressed deterministically alongside
this record: [Trivy](results/trivy-shipping.json.gz),
[Grype](results/grype-shipping.json.gz), and the
[SBOM](results/sbom.cdx.json.gz). The SBOM supplements the package inventory
with measured FFmpeg 9.0.2 native-library metadata. The
[runtime inventory](results/runtime-inventory.json) records 990 header-package
files, no ELF binaries in that package, and no installed kernel-image package.

## Reproduce

Use explicit empty scanner configuration files, retain default scanner rules,
and record tool/database versions and the image digest:

```bash
printf '{}\n' > scanner-defaults.yaml
trivy image --config scanner-defaults.yaml --scanners vuln \
  --format json --output trivy.json "$V100_IMAGE"
grype --config scanner-defaults.yaml "$V100_IMAGE" \
  -o json --file grype.json
trivy image --config scanner-defaults.yaml --format cyclonedx \
  --output sbom.cdx.json "$V100_IMAGE"
```

Give scanner temporary storage enough space for the image; a small `/tmp`
tmpfs is insufficient. `inventory.py` can run with no network, no GPU, a
read-only root and UID 10001. It records package versions, source/native hashes,
loaded codec functionality and the header-package inventory. Public advisory
coverage and database freshness are separate from Prisma's own policy checks.
