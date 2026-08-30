# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 0: everything that can be answered without renting a GPU."""

import subprocess
from pathlib import Path

_LANDMINE_AREAS = {
    "speculative decoding": (
        "speculative",
        "spec decode",
        "spec-decode",
        "mtp",
        "eagle",
        "draft",
    ),
    "sliding-window attention": ("sliding-window", "sliding window"),
    "all-reduce": ("all-reduce", "allreduce", "reduce-scatter", "nvlink"),
    "kv-cache dtype": ("kv_cache_dtype", "kv-cache dtype", "kv cache dtype"),
    "model runner": ("model runner", "v2 runner", "model_runner"),
    "structured output": ("structured output", "guided", "grammar", "xgrammar"),
    "attention backend": ("attention backend", "flashinfer", "triton_attn"),
}


def read_upstream_map(path: Path) -> dict[str, str]:
    """Read the patch-to-upstream-commit map.

    Args:
        path: Location of upstream.map.

    Returns:
        Mapping of patch filename to upstream merge commit.

    Raises:
        ValueError: If a non-comment line is not exactly two fields.
    """
    mapping: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected '<patch> <commit>'")
        mapping[fields[0]] = fields[1]
    return mapping


def _object_exists(rev: str, repo: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", rev],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def is_absorbed(merge_commit: str, tag: str, repo: Path) -> bool:
    """Report whether a tag already contains an upstream commit.

    Args:
        merge_commit: Upstream merge commit for a patch.
        tag: Release tag being considered.
        repo: Repository checkout.

    Returns:
        True when the commit is an ancestor of the tag.

    Raises:
        LookupError: If either revision is missing from the checkout. Answering
            "not absorbed" for an unfetched commit would keep a patch forever
            for the wrong reason, which is the failure this check exists to
            prevent.
    """
    if merge_commit == "none":
        return False

    for rev in (merge_commit, tag):
        if not _object_exists(rev, repo):
            raise LookupError(f"{rev} is not in this checkout; fetch upstream first")

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", merge_commit, tag],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def applies_cleanly(patch: Path, tree: Path) -> bool:
    """Report whether a patch still applies to a tree.

    Uses the same invocation as the image build, so a pass here means the build
    will not fail on this patch.

    Args:
        patch: Patch file.
        tree: Directory to apply against.

    Returns:
        True when a dry run succeeds.
    """
    with patch.open("rb") as handle:
        result = subprocess.run(
            ["patch", "-p1", "--dry-run", "--force", f"--directory={tree}"],
            stdin=handle,
            capture_output=True,
            check=False,
        )
    return result.returncode == 0


def scan_release_notes(text: str) -> tuple[str, ...]:
    """Find areas of a release that this fork depends on.

    Args:
        text: Release notes body.

    Returns:
        Area names with at least one keyword hit, in declaration order.
    """
    lowered = text.lower()
    return tuple(
        area
        for area, keywords in _LANDMINE_AREAS.items()
        if any(keyword in lowered for keyword in keywords)
    )


def brief(tag: str, repo: Path) -> str:
    """Render the phase 0 brief for a candidate tag.

    Args:
        tag: Release tag being considered.
        repo: Repository checkout.

    Returns:
        A Markdown brief.
    """
    patch_dir = repo / "fork" / "patches"
    mapping = read_upstream_map(patch_dir / "upstream.map")
    lines = [
        f"# Phase 0 brief: {tag}",
        "",
        "| patch | upstream commit | absorbed by tag |",
        "|---|---|---|",
    ]
    for name, commit in sorted(mapping.items()):
        absorbed = "yes" if is_absorbed(commit, tag, repo) else "no"
        lines.append(f"| {name} | `{commit}` | {absorbed} |")
    lines.append("")
    return "\n".join(lines)
