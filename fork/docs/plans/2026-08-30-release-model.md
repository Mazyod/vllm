# Release model (frozen tags on pristine upstream tags; overlay-only `main`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a release equal "pristine upstream tag + documented patch commits", frozen as an immutable `fork/<tag>` tag, with `main` reduced to the fork overlay and every generated artefact CI-verified.

**Architecture:** Bash scripts under `fork/scripts/` do all the git mechanics (export, history rules, alignment, new-release, freeze, migration); each is a pure function of its arguments so it is unit-tested from pytest with throwaway git repos. The image workflow calls those scripts instead of inlining logic. Root tooling files that upstream owns today are staged under `fork/overlay-root/` and moved to the root by the one-shot migration.

**Tech Stack:** bash (shellcheck-clean, `set -euo pipefail`), git, GNU patch, pytest via `uv run --no-project --with pytest --with httpx --with pyyaml`, GitHub Actions, `gh`.

**Spec:** `fork/docs/specs/2026-08-30-release-branches-design.md`

## Global Constraints

- All Python: `uv run --no-project --with pytest --with httpx --with pyyaml -- pytest …`. Never bare `python`/`pip`.
- Pre-commit hooks run on commit: ruff, ruff-format, typos, markdownlint (tables need `| --- |` separators with spaces; fenced blocks need a language), shellcheck, `check-forbidden-imports` (**no `import re` in Python** — use `str` methods or `urllib`), SPDX header on every `.py` (`# SPDX-License-Identifier: Apache-2.0` + `# SPDX-FileCopyrightText: Copyright contributors to the vLLM project`), `signoff-commit` (commit with `git commit -s`).
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` above the Signed-off-by line.
- Do not modify any upstream-owned file (anything outside `FORK.md`, `fork/**`, `.github/workflows/build-vllm-audio.yml`, `.github/workflows/fork-alignment.yml`, `runs/.gitignore`). Root tooling copies go under `fork/overlay-root/`.
- Scripts: `#!/usr/bin/env bash`, `set -euo pipefail`, every script prints a one-line `usage:` on bad args and exits 2; all git calls take an explicit `-C "$REPO"`.
- `git format-patch` flags, verbatim: `-k --full-index --no-renames --zero-commit --no-signature --no-stat`.
- Patch commit contract: subject starts `[fork-patch]`; trailers `Upstream-PR:`, `Upstream-Merge:`, `Exit-Criterion:` all present (`git interpret-trailers --parse`); `Upstream-Merge` is a 40-hex sha or the literal `none`.
- Tests create git repos under `tmp_path` with `git -c user.name=t -c user.email=t@t -c commit.gpgsign=false` and an initial commit tagged `v9.9.9` as the fake upstream tag; use helper `fork/bench/tests/gitfixtures.py` (Task 1) everywhere.
- Test names state the behaviour; one behaviour per test; no sleeps.

---

## File structure

| path | responsibility |
| --- | --- |
| `fork/bench/tests/gitfixtures.py` | helpers: `init_repo`, `commit_file`, `patch_commit` (writes a `[fork-patch]` commit with trailers), `run_script` |
| `fork/scripts/export-patches.sh` | `<sha>` → regenerate `fork/patches/{NNNN-*.patch,series,upstream.map,RELEASE}` |
| `fork/scripts/check-release-history.sh` | `<tag> <sha>` → enforce the per-commit rules; exit 0/1 with one line per violation |
| `fork/scripts/check-alignment.sh` | rewritten: main-mode rules 1–5; `--pre-migration` keeps today's diff-vs-tag check |
| `fork/scripts/new-release.sh` | `<tag>` → work branch from tag, replay surviving patches, export, copy configs, bump four pins |
| `fork/scripts/freeze-release.sh` | `<tag> <release-sha> <candidate-digest> <base-digest> <main-sha> <export-hash> <gate-record>` → create-or-verify `fork/<tag>` |
| `fork/scripts/export-hash.sh` | sha256 over `fork/patches/` in a stable order (shared by build label and freeze) |
| `fork/scripts/migrate-to-overlay-main.sh` | one-shot: archive tag, freeze v0.27.1/v0.28.0, orphan `main`, branch deletions, rulesets; `--dry-run` |
| `fork/overlay-root/` | fork-owned `pyproject.toml`, `.pre-commit-config.yaml`, `.markdownlint.yaml`, `.shellcheckrc`, `AGENTS.md`, `CLAUDE.md` staged for the migration |
| `fork/patches/RELEASE` | pointer: `tag:` and `release-sha:` |
| `fork/alignment.ledger` | `add` lines only |
| `.github/workflows/build-vllm-audio.yml` | ref guard, pinned checkout, labels, promote → `freeze-release.sh` |
| `.github/workflows/fork-alignment.yml` | unchanged trigger; passes `--pre-migration` until migration |
| docs | `FORK.md`, `fork/README.md`, `fork/patches/README.md`, `fork/bench/RUNBOOK.md`, `fork/bench/LESSONS.md` |

Deleted: `fork/scripts/refresh-patches.sh`.

---

### Task 1: Git fixtures for script tests

**Files:**

- Create: `fork/bench/tests/gitfixtures.py`
- Test: `fork/bench/tests/test_gitfixtures.py`

**Interfaces:**

- Produces:
    - `init_repo(path: Path, *, tag: str = "v9.9.9") -> Path` — `git init`, config user, commit `vllm/__init__.py` (`"# upstream\n"`) and `vllm/v1/core.py` (`"x = 1\n"`), tag `<tag>` (annotated). Returns `path`.
    - `patch_commit(repo: Path, rel_path: str, content: str, subject: str = "[fork-patch] change", *, pr: str = "https://github.com/vllm-project/vllm/pull/1", merge: str = "none", exit_criterion: str = "upstream merges #1", trailers: bool = True) -> str` — writes file, commits with subject + body `Impact: t.\n` + trailers (omitted when `trailers=False`); returns full sha.
    - `run_script(script: Path, *args: str, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess[str]` — runs `bash script args` with `capture_output=True, text=True, check=False`, `REPO=cwd` in env.
    - `SCRIPTS = REPO_ROOT / "fork" / "scripts"` where `REPO_ROOT = Path(__file__).resolve().parents[3]`.

- [ ] **Step 1: Write the failing test**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixture helpers themselves: a bad fixture fails every script test."""

import subprocess

from fork.bench.tests.gitfixtures import init_repo, patch_commit


def test_init_repo_tags_the_upstream_commit(tmp_path):
    repo = init_repo(tmp_path / "r")
    out = subprocess.run(["git", "-C", str(repo), "tag"], capture_output=True, text=True)
    assert out.stdout.split() == ["v9.9.9"]


def test_patch_commit_carries_the_three_trailers(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    body = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B", sha],
        capture_output=True, text=True,
    ).stdout
    assert body.startswith("[fork-patch] change")
    for key in ("Upstream-PR:", "Upstream-Merge:", "Exit-Criterion:"):
        assert key in body
```

- [ ] **Step 2: Run to verify it fails** — `uv run --no-project --with pytest --with httpx --with pyyaml -- pytest fork/bench/tests/test_gitfixtures.py -q` → ImportError.

- [ ] **Step 3: Implement `gitfixtures.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throwaway git repositories for exercising fork/scripts."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "fork" / "scripts"
_GIT_CFG = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *_GIT_CFG, "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def init_repo(path: Path, *, tag: str = "v9.9.9") -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    (path / "vllm" / "v1").mkdir(parents=True)
    (path / "vllm" / "__init__.py").write_text("# upstream\n", encoding="utf-8")
    (path / "vllm" / "v1" / "core.py").write_text("x = 1\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "upstream")
    git(path, "tag", "-a", tag, "-m", tag)
    return path


def patch_commit(
    repo: Path, rel_path: str, content: str, subject: str = "[fork-patch] change", *,
    pr: str = "https://github.com/vllm-project/vllm/pull/1", merge: str = "none",
    exit_criterion: str = "upstream merges #1", trailers: bool = True,
) -> str:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    message = f"{subject}\n\nImpact: t.\n"
    if trailers:
        message += (
            f"\nUpstream-PR: {pr}\nUpstream-Merge: {merge}\n"
            f"Exit-Criterion: {exit_criterion}\n"
        )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def run_script(script: Path, *args: str, cwd: Path, env: dict | None = None):
    merged = {**os.environ, "REPO": str(cwd), **(env or {})}
    return subprocess.run(
        ["bash", str(script), *args], cwd=cwd, env=merged,
        capture_output=True, text=True, check=False,
    )
```

- [ ] **Step 4: Run to verify it passes** — same command → 2 passed.
- [ ] **Step 5: Commit** — `git add fork/bench/tests/gitfixtures.py fork/bench/tests/test_gitfixtures.py && git commit -s -m "[fork] Add throwaway-git fixtures for script tests"` (+ Co-Authored-By trailer).

---

### Task 2: `export-patches.sh`

**Files:**

- Create: `fork/scripts/export-patches.sh`
- Test: `fork/bench/tests/test_export.py`

**Interfaces:**

- Consumes: `gitfixtures`.
- Produces: `export-patches.sh <sha>` (env `REPO` = repo root, default `git rev-parse --show-toplevel`; env `BASE_TAG` overrides the tag read from `fork/docker/Dockerfile.audio`'s `ARG BASE_TAG=`; env `PATCH_DIR` overrides `$REPO/fork/patches`). Writes into `PATCH_DIR`: `NNNN-<slug>.patch` (slug = `git format-patch`'s own filename, `-k` keeps the subject; strip the `[fork-patch]` prefix from the slug: format-patch already drops bracketed prefixes), `series` (one filename per line, header comment `# generated by fork/scripts/export-patches.sh from <sha>; do not edit`), `upstream.map` (header comment + `<patch> <Upstream-Merge>` per patch whose `Upstream-Merge` is not `none`), `RELEASE` (`tag: <BASE_TAG>\nrelease-sha: <full sha>\n`). Deletes any `*.patch` in `PATCH_DIR` not regenerated. Exit 1 with message if `<sha>` is not a descendant of the tag.

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fork/patches is generated from the release commits and reproducible."""

import subprocess
from pathlib import Path

from fork.bench.tests.gitfixtures import SCRIPTS, git, init_repo, patch_commit, run_script

EXPORT = SCRIPTS / "export-patches.sh"


def _export(repo: Path, sha: str, out: Path):
    out.mkdir(exist_ok=True)
    result = run_script(EXPORT, sha, cwd=repo, env={"BASE_TAG": "v9.9.9", "PATCH_DIR": str(out)})
    assert result.returncode == 0, result.stderr
    return sorted(p.name for p in out.iterdir())


def test_export_writes_one_patch_per_commit_plus_series_map_and_release(tmp_path):
    repo = init_repo(tmp_path / "r")
    patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    sha = patch_commit(repo, "vllm/v1/other.py", "y = 1\n", "[fork-patch] add y", merge="a" * 40)
    names = _export(repo, sha, tmp_path / "out")
    assert names == ["0001-bump-x.patch", "0002-add-y.patch", "RELEASE", "series", "upstream.map"]
    out = tmp_path / "out"
    assert (out / "series").read_text().splitlines()[-2:] == ["0001-bump-x.patch", "0002-add-y.patch"]
    assert (out / "upstream.map").read_text().splitlines()[-1] == "0002-add-y.patch " + "a" * 40
    assert (out / "RELEASE").read_text() == f"tag: v9.9.9\nrelease-sha: {sha}\n"


def test_export_is_byte_identical_after_a_content_identical_rebase(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    _export(repo, sha, tmp_path / "a")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo),
         "commit", "-q", "--amend", "--no-edit"],
        env={"GIT_COMMITTER_DATE": "2030-01-01T00:00:00+0000", "PATH": "/usr/bin:/bin"},
        check=True,
    )
    sha2 = git(repo, "rev-parse", "HEAD").strip()
    assert sha2 != sha
    _export(repo, sha2, tmp_path / "b")
    a = (tmp_path / "a" / "0001-bump-x.patch").read_bytes()
    b = (tmp_path / "b" / "0001-bump-x.patch").read_bytes()
    assert a == b


def test_export_of_the_tag_itself_yields_no_patches_but_a_release_pointer(tmp_path):
    repo = init_repo(tmp_path / "r")
    tag_sha = git(repo, "rev-parse", "v9.9.9^{commit}").strip()
    names = _export(repo, tag_sha, tmp_path / "out")
    assert names == ["RELEASE", "series", "upstream.map"]


def test_export_removes_stale_patch_files(tmp_path):
    repo = init_repo(tmp_path / "r")
    out = tmp_path / "out"; out.mkdir()
    (out / "0009-stale.patch").write_text("junk")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n", "[fork-patch] bump x")
    names = _export(repo, sha, out)
    assert "0009-stale.patch" not in names


def test_export_refuses_a_sha_not_descended_from_the_tag(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "checkout", "-q", "--orphan", "other")
    (repo / "z").write_text("z"); git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "z")
    sha = git(repo, "rev-parse", "HEAD").strip()
    result = run_script(EXPORT, sha, cwd=repo, env={"BASE_TAG": "v9.9.9", "PATCH_DIR": str(tmp_path / "o")})
    assert result.returncode == 1 and "not descended from v9.9.9" in result.stderr
```

- [ ] **Step 2: Run to verify they fail** — script missing → all 5 fail.

- [ ] **Step 3: Implement**

```bash
#!/usr/bin/env bash
# Regenerate fork/patches from the release commits on top of the base tag.
#
#   export-patches.sh <sha>        (env: REPO, BASE_TAG, PATCH_DIR)
#
# Output is byte-reproducible: --zero-commit drops the commit id, --full-index
# drops the abbreviation dependence, the Date: line is the author date. The
# RELEASE file pins the exact commit the files came from.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: export-patches.sh <sha>" >&2; exit 2; }
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
PATCH_DIR="${PATCH_DIR:-$REPO/fork/patches}"
BASE_TAG="${BASE_TAG:-$(sed -n 's/^ARG BASE_TAG=//p' "$REPO/fork/docker/Dockerfile.audio" | head -1)}"
[ -n "$BASE_TAG" ] || { echo "ERROR: no base tag (ARG BASE_TAG or \$BASE_TAG)" >&2; exit 1; }
SHA="$(git -C "$REPO" rev-parse --verify "$1^{commit}")"
BASE="$(git -C "$REPO" rev-parse --verify "$BASE_TAG^{commit}")"
git -C "$REPO" merge-base --is-ancestor "$BASE" "$SHA" ||
  { echo "ERROR: $SHA is not descended from $BASE_TAG" >&2; exit 1; }
mkdir -p "$PATCH_DIR"
find "$PATCH_DIR" -maxdepth 1 -name '*.patch' -delete
git -C "$REPO" format-patch -q -k --full-index --no-renames --zero-commit \
  --no-signature --no-stat -o "$PATCH_DIR" "$BASE..$SHA"
{
  echo "# generated by fork/scripts/export-patches.sh from $SHA; do not edit"
  for f in "$PATCH_DIR"/*.patch; do [ -e "$f" ] && basename "$f"; done
} > "$PATCH_DIR/series"
{
  echo "# generated by fork/scripts/export-patches.sh from $SHA; do not edit"
  echo "# <patch-filename> <Upstream-Merge sha>; patches with Upstream-Merge: none are omitted"
  i=0
  for c in $(git -C "$REPO" rev-list --reverse "$BASE..$SHA"); do
    i=$((i + 1))
    merge="$(git -C "$REPO" log -1 --format=%B "$c" | git interpret-trailers --parse |
      sed -n 's/^Upstream-Merge: *//p' | head -1)"
    name="$(printf '%04d' "$i")"
    file="$(find "$PATCH_DIR" -maxdepth 1 -name "${name}-*.patch" -printf '%f\n')"
    [ "$merge" != "none" ] && [ -n "$merge" ] && echo "$file $merge"
  done
} > "$PATCH_DIR/upstream.map"
printf 'tag: %s\nrelease-sha: %s\n' "$BASE_TAG" "$SHA" > "$PATCH_DIR/RELEASE"
echo "exported $(git -C "$REPO" rev-list --count "$BASE..$SHA") patch(es) from $SHA into ${PATCH_DIR#"$REPO"/}"
```

Note: `format-patch -k` names files `0001-bump-x.patch` from the subject with the bracketed prefix removed. If the produced slug differs, adjust the test expectation to the observed name — the rule is "format-patch's own name".

- [ ] **Step 4: Run to verify they pass.** Also `shellcheck fork/scripts/export-patches.sh`.
- [ ] **Step 5: Commit** — `[fork] Generate fork/patches from the release commits`.

---

### Task 3: `check-release-history.sh`

**Files:**

- Create: `fork/scripts/check-release-history.sh`
- Test: `fork/bench/tests/test_release_history.py`

**Interfaces:**

- Produces: `check-release-history.sh <tag> <sha>` (env `REPO`). Prints `ok: N patch commit(s) on <tag>` and exits 0, or one `violation: <sha7> <reason>` line per problem and exits 1. Rules: (a) `<tag>` is an ancestor of `<sha>`; (b) no merge commits in `<tag>..<sha>`; (c) no empty commits; (d) subject starts with `[fork-patch]`; (e) trailers `Upstream-PR`, `Upstream-Merge` (40-hex or `none`), `Exit-Criterion` present; (f) every changed path in each commit starts with `vllm/`; (g) `git diff-tree --no-renames -r --raw` status is only `M` or `A` or `D`, modes only `100644`, and `git diff-tree -p` contains no `Binary files` / `GIT binary patch`; (h) no symlinks (`120000`).

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only well-formed patch commits may sit between the upstream tag and a release."""

import os
from pathlib import Path

import pytest

from fork.bench.tests.gitfixtures import SCRIPTS, git, init_repo, patch_commit, run_script

CHECK = SCRIPTS / "check-release-history.sh"


def _check(repo: Path, sha: str):
    return run_script(CHECK, "v9.9.9", sha, cwd=repo)


def test_a_clean_patch_series_passes(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = _check(repo, sha)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok: 1 patch commit" in result.stdout


def test_the_tag_itself_passes_with_zero_patches(tmp_path):
    repo = init_repo(tmp_path / "r")
    assert _check(repo, git(repo, "rev-parse", "v9.9.9^{commit}").strip()).returncode == 0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda r: patch_commit(r, "vllm/v1/core.py", "x = 2\n", "fix x"), "subject"),
        (lambda r: patch_commit(r, "vllm/v1/core.py", "x = 2\n", trailers=False), "trailer"),
        (lambda r: patch_commit(r, "vllm/v1/core.py", "x = 2\n", merge="notasha"), "Upstream-Merge"),
        (lambda r: patch_commit(r, "docs/x.md", "hi\n"), "outside vllm/"),
    ],
)
def test_each_rule_names_its_violation(tmp_path, mutate, reason):
    repo = init_repo(tmp_path / "r")
    sha = mutate(repo)
    result = _check(repo, sha)
    assert result.returncode == 1
    assert reason in result.stdout


def test_a_merge_commit_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    git(repo, "checkout", "-q", "-b", "side")
    patch_commit(repo, "vllm/v1/side.py", "s = 1\n")
    git(repo, "checkout", "-q", "main")
    patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    result = _check(repo, git(repo, "rev-parse", "HEAD").strip())
    assert result.returncode == 1 and "merge" in result.stdout


def test_a_symlink_or_binary_or_mode_change_is_rejected(tmp_path):
    repo = init_repo(tmp_path / "r")
    os.symlink("core.py", repo / "vllm" / "v1" / "link.py")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "[fork-patch] link\n\nUpstream-PR: u\nUpstream-Merge: none\nExit-Criterion: e\n")
    assert "symlink" in _check(repo, git(repo, "rev-parse", "HEAD").strip()).stdout
    (repo / "vllm" / "v1" / "blob.bin").write_bytes(b"\x00\x01\x02")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "[fork-patch] bin\n\nUpstream-PR: u\nUpstream-Merge: none\nExit-Criterion: e\n")
    assert "binary" in _check(repo, git(repo, "rev-parse", "HEAD").strip()).stdout
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

```bash
#!/usr/bin/env bash
# Enforce the patch-commit contract on <tag>..<sha>. One line per violation.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: check-release-history.sh <tag> <sha>" >&2; exit 2; }
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
TAG="$1"; SHA="$(git -C "$REPO" rev-parse --verify "$2^{commit}")"
BASE="$(git -C "$REPO" rev-parse --verify "$TAG^{commit}")"
violations=0
bad() { echo "violation: ${1:0:7} $2"; violations=$((violations + 1)); }
git -C "$REPO" merge-base --is-ancestor "$BASE" "$SHA" || { bad "$SHA" "$TAG is not an ancestor"; echo "FAIL"; exit 1; }
for c in $(git -C "$REPO" rev-list --reverse "$BASE..$SHA"); do
  [ "$(git -C "$REPO" rev-list --parents -n1 "$c" | wc -w)" -le 2 ] || bad "$c" "merge commit"
  subject="$(git -C "$REPO" log -1 --format=%s "$c")"
  case "$subject" in "[fork-patch] "*) ;; *) bad "$c" "subject must start with '[fork-patch] '";; esac
  trailers="$(git -C "$REPO" log -1 --format=%B "$c" | git interpret-trailers --parse)"
  for key in Upstream-PR Upstream-Merge Exit-Criterion; do
    grep -q "^$key: ." <<<"$trailers" || bad "$c" "missing trailer $key"
  done
  merge="$(sed -n 's/^Upstream-Merge: *//p' <<<"$trailers" | head -1)"
  case "$merge" in none | "") ;; *) [[ "$merge" =~ ^[0-9a-f]{40}$ ]] || bad "$c" "Upstream-Merge must be a 40-hex sha or none";; esac
  raw="$(git -C "$REPO" diff-tree --no-commit-id --no-renames -r --raw "$c")"
  [ -n "$raw" ] || bad "$c" "empty commit"
  while read -r srcmode dstmode _ _ status path; do
    [ -n "$path" ] || continue
    case "$path" in vllm/*) ;; *) bad "$c" "touches $path outside vllm/";; esac
    case "$status" in M | A | D) ;; *) bad "$c" "unsupported change $status on $path";; esac
    for m in "$srcmode" "$dstmode"; do
      case "$m" in :100644 | 100644 | :000000 | 000000) ;; :120000 | 120000) bad "$c" "symlink $path";; *) bad "$c" "mode $m on $path";; esac
    done
  done <<<"$raw"
  git -C "$REPO" diff-tree --no-commit-id --no-renames -r -p "$c" | grep -qE '^(Binary files|GIT binary patch)' && bad "$c" "binary content"
done
if [ "$violations" -gt 0 ]; then echo "FAIL: $violations violation(s) on $TAG..${SHA:0:7}"; exit 1; fi
echo "ok: $(git -C "$REPO" rev-list --count "$BASE..$SHA") patch commit(s) on $TAG"
```

- [ ] **Step 4: Run to verify they pass**; `shellcheck`.
- [ ] **Step 5: Commit** — `[fork] Enforce the patch-commit contract between a tag and a release`.

---

### Task 4: `export-hash.sh` and `freeze-release.sh`

**Files:**

- Create: `fork/scripts/export-hash.sh`, `fork/scripts/freeze-release.sh`
- Test: `fork/bench/tests/test_freeze.py`

**Interfaces:**

- `export-hash.sh` (env `PATCH_DIR`, default `$REPO/fork/patches`): prints `sha256:<hex>` over the concatenation of `<name>\0<bytes>` for every regular file in `PATCH_DIR` sorted by name (LC_ALL=C). Used by the build label and the freeze.
- `freeze-release.sh <tag> <release-sha> <candidate-digest> <base-digest> <main-sha> <export-hash> <gate-record>` (env `REPO`, `REMOTE` default `origin`, `PUSH` default `1`): if tag `fork/<tag>` is absent → `git tag -a fork/<tag> <release-sha> -m "<message>"` where message is exactly:

```text
fork release <tag>

release-sha: <release-sha>
candidate-digest: <candidate-digest>
base-digest: <base-digest>
main-sha: <main-sha>
patch-export: <export-hash>
gate-record: <gate-record>
```

  then (if `PUSH=1`) `git push <REMOTE> refs/tags/fork/<tag>` and verify `git ls-remote <REMOTE> refs/tags/fork/<tag>^{}` equals `<release-sha>`; print `frozen fork/<tag> at <release-sha>`. If the tag exists → parse its message (`git tag -l --format=%(contents) fork/<tag>`); require `release-sha` and `candidate-digest` to equal the arguments; print `already frozen: fork/<tag> matches`; any mismatch → print `refusing: fork/<tag> records <field>=<old>, got <new>` and exit 1. All fields must be non-empty (exit 2 otherwise).

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frozen tag binds source and shipped bytes; it is written once."""

from fork.bench.tests.gitfixtures import SCRIPTS, git, init_repo, patch_commit, run_script

FREEZE = SCRIPTS / "freeze-release.sh"
HASH = SCRIPTS / "export-hash.sh"
ARGS = ("sha256:cand", "sha256:base", "m" * 40, "sha256:exp", "fork/bench/configs/v9.9.9/results/x.md")


def test_export_hash_is_order_independent_and_content_sensitive(tmp_path):
    d = tmp_path / "p"; d.mkdir()
    (d / "b.patch").write_text("B"); (d / "a.patch").write_text("A")
    h1 = run_script(HASH, cwd=tmp_path, env={"PATCH_DIR": str(d)}).stdout.strip()
    (d / "a.patch").write_text("A2")
    h2 = run_script(HASH, cwd=tmp_path, env={"PATCH_DIR": str(d)}).stdout.strip()
    assert h1.startswith("sha256:") and h1 != h2


def test_first_freeze_creates_the_annotated_tag_on_the_release_sha(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    assert result.returncode == 0, result.stderr
    assert git(repo, "rev-parse", "fork/v9.9.9^{commit}").strip() == sha
    assert "candidate-digest: sha256:cand" in git(repo, "tag", "-l", "--format=%(contents)", "fork/v9.9.9")


def test_second_freeze_with_the_same_digest_is_a_no_op(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    again = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    assert again.returncode == 0 and "already frozen" in again.stdout


def test_freeze_refuses_a_different_digest_or_sha_for_a_frozen_tag(tmp_path):
    repo = init_repo(tmp_path / "r")
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo, env={"PUSH": "0"})
    other = ("sha256:other",) + ARGS[1:]
    result = run_script(FREEZE, "v9.9.9", sha, *other, cwd=repo, env={"PUSH": "0"})
    assert result.returncode == 1 and "refusing" in result.stdout
    sha2 = patch_commit(repo, "vllm/v1/core.py", "x = 3\n")
    result = run_script(FREEZE, "v9.9.9", sha2, *ARGS, cwd=repo, env={"PUSH": "0"})
    assert result.returncode == 1 and "release-sha" in result.stdout


def test_freeze_pushes_and_verifies_the_remote_tag(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    repo = init_repo(tmp_path / "r")
    git(repo, "remote", "add", "origin", str(remote))
    sha = patch_commit(repo, "vllm/v1/core.py", "x = 2\n")
    result = run_script(FREEZE, "v9.9.9", sha, *ARGS, cwd=repo)
    assert result.returncode == 0, result.stderr
    assert sha in git(repo, "ls-remote", "origin", "refs/tags/fork/v9.9.9^{}")
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement both scripts** per the interface (export-hash: `find "$PATCH_DIR" -maxdepth 1 -type f | LC_ALL=C sort | while read -r f; do printf '%s\0' "$(basename "$f")"; cat "$f"; done | sha256sum` → `sha256:<hex>`).
- [ ] **Step 4: Run to verify they pass**; `shellcheck` both.
- [ ] **Step 5: Commit** — `[fork] Freeze a release as an annotated tag that binds source and image digest`.

---

### Task 5: `check-alignment.sh` rewrite

**Files:**

- Modify: `fork/scripts/check-alignment.sh` (rewrite; keep today's body as `legacy_check()`)
- Modify: `fork/alignment.ledger` (remove all `del` lines; add `add` lines for `pyproject.toml`, `.pre-commit-config.yaml`, `.markdownlint.yaml`, `.shellcheckrc`, `AGENTS.md`, `CLAUDE.md`, `fork/patches/RELEASE` is under `fork/**` already)
- Modify: `.github/workflows/fork-alignment.yml` — run `fork/scripts/check-alignment.sh --fetch --pre-migration`
- Test: `fork/bench/tests/test_alignment.py`

**Interfaces:**

- `check-alignment.sh [--fetch] [--pre-migration]` (env `REPO`, `LEDGER`, `UPSTREAM_REMOTE`, `ORIGIN_REMOTE` default `origin`, `SKIP_FETCH_RELEASE=1` to use only local objects — tests set it).
    - `--pre-migration`: run `legacy_check` exactly as today (diff HEAD vs tag against the ledger, incl. `del` handling read from the ledger if present) and exit with its status. Nothing else.
    - default (post-migration) rules, each printing `ok  <rule>` or `FAIL <rule>: <detail>`:
    1. `tracked-paths`: every path in `git ls-files` matches a ledger `add` pattern.
    2. `pins`: `ARG BASE_TAG` (Dockerfile.audio) == `DEFAULT_BASE_TAG:` (workflow) == `DEFAULT_TAG = "…"` in `fork/bench/profiles.py` == the `--tag` in `fork/bench/preflight.sh` == `tag:` in `fork/patches/RELEASE`, and `fork/bench/configs/<tag>/fleet.yaml` exists.
    3. `release-history`: fetch `release-sha` (unless `SKIP_FETCH_RELEASE=1`: `git fetch $ORIGIN_REMOTE <sha>`; if that fails, `git fetch $ORIGIN_REMOTE --tags`), then `check-release-history.sh <tag> <sha>`.
    4. `export`: export `<sha>` into a temp dir with `BASE_TAG=<tag>`, `diff -r` against `fork/patches/` (ignore nothing) — must be empty.
    5. `frozen`: if `fork/<tag>` exists locally or on origin, its peeled commit == `release-sha`.
    - Exit 1 on any FAIL.

- [ ] **Step 1: Write the failing tests** (overlay repos are built by a helper `overlay_repo(tmp_path, upstream_repo, release_sha)` in this test file: a second git repo containing `fork/docker/Dockerfile.audio` with `ARG BASE_TAG=v9.9.9`, `.github/workflows/build-vllm-audio.yml` containing `DEFAULT_BASE_TAG: v9.9.9`, `fork/bench/profiles.py` containing `DEFAULT_TAG = "v9.9.9"`, `fork/bench/preflight.sh` containing `--tag v9.9.9`, `fork/bench/configs/v9.9.9/fleet.yaml`, a ledger with `add fork/** permanent x` and `add .github/workflows/build-vllm-audio.yml permanent x`, and `fork/patches/` produced by running `export-patches.sh` with `REPO=<upstream_repo>`; the overlay repo gets `origin` = the upstream repo so `git fetch origin <sha>` works — configure `uploadpack.allowReachableSHA1InWant=true` on the upstream repo).

```python
def test_a_consistent_overlay_passes_every_rule(tmp_path): ...            # returncode 0, five "ok" lines
def test_an_upstream_file_on_main_fails_tracked_paths(tmp_path): ...       # add vllm/x.py → "FAIL tracked-paths"
def test_a_pin_mismatch_fails_pins(tmp_path): ...                          # profiles DEFAULT_TAG = "v0.0.1" → "FAIL pins"
def test_a_stale_export_fails_export(tmp_path): ...                        # edit one .patch byte → "FAIL export"
def test_a_bad_release_history_fails_release_history(tmp_path): ...       # release-sha with a missing trailer → "FAIL release-history"
def test_a_moved_pointer_after_freeze_fails_frozen(tmp_path): ...          # tag fork/v9.9.9 at old sha, RELEASE at new → "FAIL frozen"
def test_pre_migration_mode_runs_only_the_legacy_diff_check(tmp_path): ... # repo built on the tag with an undeclared modified upstream file → "FORBIDDEN", and no "tracked-paths" line
```

Write each test fully (no `...` in the committed file); the helper builds the fixture, the test mutates one thing, asserts the one line.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Structure: parse args; `legacy_check()` = today's script body verbatim (it still reads `del` lines if present); `main_check()` = rules 1–5 calling `check-release-history.sh` and `export-patches.sh` from `$(dirname "$0")`. Keep the "behind upstream/main" note out of `main_check` (it is meaningless for an orphan `main`).
- [ ] **Step 4: Run tests; `shellcheck`; then run the real check both ways on this checkout:** `fork/scripts/check-alignment.sh --pre-migration` must pass (today's tree); `fork/scripts/check-alignment.sh` is expected to FAIL `tracked-paths` on today's tree — confirm that is the only failing rule after Task 6 writes `RELEASE`.
- [ ] **Step 5: Commit** — `[fork] Alignment: verify the overlay, the pointer, and the export; keep the tag diff behind --pre-migration`.

---

### Task 6: Pointer, ledger, and `new-release.sh`

**Files:**

- Create: `fork/patches/RELEASE` (`tag: v0.28.0`, `release-sha: <git rev-parse v0.28.0^{commit}>`)
- Modify: `fork/patches/series`, `fork/patches/upstream.map` (regenerate by running `fork/scripts/export-patches.sh v0.28.0` — they become the generated header-only files)
- Delete: `fork/scripts/refresh-patches.sh`
- Create: `fork/scripts/new-release.sh`
- Modify: `fork/bench/static.py` — none required; `fork/bench/tests/test_static.py` — add `test_release_pointer_names_the_pinned_tag` (reads `RELEASE`, asserts `tag:` == `profiles.DEFAULT_TAG` and `release-sha` is 40 hex)
- Test: `fork/bench/tests/test_new_release.py`

**Interfaces:**

- `new-release.sh <tag>` (env `REPO`, `UPSTREAM_REMOTE` default `upstream`, `ORIGIN_REMOTE` default `origin`, `PREV_TAG` default from `RELEASE.tag`, `NO_PUSH=1` for tests, `NO_FETCH=1` for tests): (1) `git fetch upstream tag <tag>` unless NO_FETCH; (2) create branch `release/<tag>` at `<tag>^{commit}` (fail if it exists); (3) for each commit in `<prev>^{commit}..fork/<prev>` (fall back to `RELEASE.release-sha` if the frozen tag is absent): read `Upstream-Merge`; if it is a sha and `git merge-base --is-ancestor <merge> <tag>` → print `dropped: <subject> (absorbed by <tag>)`; else `git cherry-pick -x`? **No `-x`** (keeps the export reproducible) → `git cherry-pick <c>`; on conflict print `CONFLICT: <sha7> <subject> — resolve on release/<tag> and re-run export-patches.sh`, abort the cherry-pick, exit 1; (4) push `release/<tag>` unless NO_PUSH; (5) switch to a new branch `fork/bump-<tag>` off `main`; run `export-patches.sh release/<tag>` with `BASE_TAG=<tag>`; (6) `cp -r fork/bench/configs/<prev> fork/bench/configs/<tag>` excluding `results/`; (7) `sed` the four pins; (8) print the `git status --short` and stop — no commit.

- [ ] **Step 1: Write the failing tests** in a fixture with an upstream repo that has `v9.9.9` and `v9.9.10` (the latter containing the "absorbed" change) and an overlay repo with `RELEASE` at `v9.9.9` + `fork/v9.9.9` tag holding two patch commits (one with `Upstream-Merge` = the sha of the absorbing upstream commit, one `none`):

```python
def test_new_release_drops_absorbed_patches_and_replays_the_rest(tmp_path): ...  # release/v9.9.10 has 1 commit; stdout has "dropped:"
def test_new_release_bumps_all_four_pins_and_copies_configs(tmp_path): ...       # grep each pin == v9.9.10; configs/v9.9.10/fleet.yaml exists; no results/
def test_new_release_stops_on_a_conflict_and_names_the_commit(tmp_path): ...     # upstream v9.9.10 edits the same lines → returncode 1, "CONFLICT:"
def test_new_release_refuses_an_existing_work_branch(tmp_path): ...
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement**; regenerate `fork/patches/` for v0.28.0 with the real export script; delete `refresh-patches.sh`; add the `test_static` pointer test.
- [ ] **Step 4: Run the whole suite** — `uv run --no-project --with pytest --with httpx --with pyyaml -- pytest fork/bench/tests -q` — and `bash fork/bench/preflight.sh` → PREFLIGHT GREEN.
- [ ] **Step 5: Commit** — `[fork] Point the overlay at its release commit; replace refresh-patches with new-release`.

---

### Task 7: Image workflow — pinned checkout, labels, freeze on promote

**Files:**

- Modify: `.github/workflows/build-vllm-audio.yml`
- Test: `actionlint .github/workflows/build-vllm-audio.yml` (pre-commit hook) + `fork/bench/tests/test_workflow_contract.py` (parses the YAML with `yaml.safe_load` and asserts the contract below)

**Changes (exact):**

1. New input `gate_record` (string, default `""`, description "Path of the gate results page, e.g. fork/bench/configs/v0.28.0/results/20260830-attempt4.md. Required the first time a base tag is published.").
2. `resolve` job: first step `Refuse dispatches off main`: `if [[ "${{ github.event_name }}" == "workflow_dispatch" && "${{ github.ref }}" != "refs/heads/main" ]]; then echo "::error::dispatch from main only"; exit 1; fi`. Add output `main_sha: ${{ github.sha }}`. Read `fork/patches/RELEASE` (checkout needed in this job — add `actions/checkout@v5` with `ref: ${{ github.sha }}`) and export outputs `release_sha`, `release_tag`; fail if `release_tag != base_tag`. Compute `export_hash` via `fork/scripts/export-hash.sh`. Resolve `base_digest` with `docker buildx imagetools inspect vllm/vllm-openai:<base_tag> --format '{{.Manifest.Digest}}'` (needs buildx setup; no login required for a public image).
3. Every `actions/checkout@v5` in `alignment`, `build-and-push`, `test`, `promote`: add `with: ref: ${{ github.sha }}` (plus the existing `fetch-depth`/`filter` where present).
4. `build-and-push`: add
   ```yaml
   labels: |
     org.opencontainers.image.revision=${{ needs.resolve.outputs.main_sha }}
     io.openimage.release-sha=${{ needs.resolve.outputs.release_sha }}
     io.openimage.patch-export=${{ needs.resolve.outputs.export_hash }}
     io.openimage.base-digest=${{ needs.resolve.outputs.base_digest }}
   ```
5. `promote`: `permissions: contents: write`; after login, step `Read candidate labels`: `docker buildx imagetools inspect "$SOURCE" --raw` is a manifest, not config — use `docker pull "$SOURCE"` then `docker inspect --format '{{ index .Config.Labels "io.openimage.release-sha" }}'` etc. into `CAND_RELEASE_SHA`, `CAND_EXPORT`, `CAND_BASE`; step `Require the candidate to match main`: fail unless `CAND_RELEASE_SHA == release_sha` and `CAND_EXPORT == export_hash`. Step `Freeze`: only when `publish_tags` contains the base tag (i.e. `image_refs` contains `${IMAGE_NAME}:${base_tag}`): `fork/scripts/freeze-release.sh "$BASE_TAG" "$CAND_RELEASE_SHA" "$DIGEST" "$CAND_BASE" "$MAIN_SHA" "$CAND_EXPORT" "${{ github.event.inputs.gate_record }}"` with `git config user.name "github-actions[bot]"` / email `41898282+github-actions[bot]@users.noreply.github.com`; `freeze-release.sh` exits 2 on an empty `gate_record` when the tag is absent — surface as `::error::gate_record is required for the first promotion of <tag>`. Retagging happens **after** a successful freeze. When only `:latest` moves (`image_refs` lacks the base tag), still run the freeze script in verify mode by passing the same args — it must print `already frozen` or fail.
6. Summary step: add the frozen tag and its message.

- [ ] **Step 1: Write `test_workflow_contract.py`** asserting: the `gate_record` input exists; `resolve` has a step whose `run` contains `refs/heads/main`; every `uses: actions/checkout@v5` has `with.ref == "${{ github.sha }}"`; `build-and-push` `labels` contains the four keys; `promote` `permissions.contents == "write"` and some step `run` contains `freeze-release.sh`.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Edit the workflow.**
- [ ] **Step 4: `pre-commit run actionlint --files .github/workflows/build-vllm-audio.yml`; tests pass.**
- [ ] **Step 5: Commit** — `[fork] Image workflow: build only from main, label provenance, freeze on first publication`.

---

### Task 8: Overlay root files and the round-trip test scope

**Files:**

- Create: `fork/overlay-root/pyproject.toml` — `[tool.ruff]` `line-length = 88`, `[tool.ruff.lint]` and `[tool.ruff.format]` copied verbatim from the upstream root `pyproject.toml` (only those two tables), `[tool.pytest.ini_options] testpaths = ["fork/bench/tests"]`, `[tool.mypy]` copied verbatim.
- Create: `fork/overlay-root/.pre-commit-config.yaml` — hooks `ruff-check`, `ruff-format`, `typos`, `markdownlint-cli2`, `shellcheck`, `actionlint`, `signoff-commit`, each copied verbatim (repo/rev/id/args) from the upstream root `.pre-commit-config.yaml`; nothing else.
- Create: `fork/overlay-root/.markdownlint.yaml`, `fork/overlay-root/.shellcheckrc` — byte copies of the root files.
- Create: `fork/overlay-root/AGENTS.md` — 15 lines: this is the fork overlay; read `FORK.md`; never edit upstream code here (it is not here); patches are commits on `release/<tag>`; run `fork/bench/preflight.sh`; `uv run --no-project`. `fork/overlay-root/CLAUDE.md` — one line `@AGENTS.md`.
- Modify: `fork/bench/tests/test_revert_roundtrip.py` — snapshot `base_tree / "vllm"` instead of `"vllm" / "v1"` (both `copytree` and `diff -r` lines).
- Test: `fork/bench/tests/test_overlay_root.py` — `test_overlay_ruff_config_matches_upstream` (parse both tomls with `tomllib`; `tool.ruff.lint` and `tool.ruff.format` equal), `test_overlay_precommit_hooks_are_a_subset_of_upstream_with_identical_revs` (yaml), `test_overlay_lint_configs_are_byte_copies`.

- [ ] Steps 1–5 as usual; commit `[fork] Stage the root tooling the overlay-only main will carry`.

---

### Task 9: Docs

**Files:**

- Modify: `FORK.md` — replace § "The model", § "Lockstep with upstream releases", § "Testing the patches locally", and the `del` mention in § CI hygiene with the spec's model (refs table, pointer file, patch-commit contract, `new-release.sh` flow, freeze-on-promote, rulesets). Keep § Charter R1–R3 (R1's wording changes from "never modifies an upstream-owned file" to "`main` contains no upstream file; release commits touch only `vllm/**`").
- Modify: `fork/README.md` — layout tree (add `overlay-root/`, `docs/`, `scripts/{export-patches,new-release,check-release-history,freeze-release,export-hash,migrate-to-overlay-main}.sh`; remove `refresh-patches.sh`, `notes/`); § Patches: commit template.
- Modify: `fork/patches/README.md` — replace the `notes/` template with the commit-message template and trailers; state that files here are generated.
- Modify: `fork/bench/RUNBOOK.md` phase 0 — add step 0: `git worktree add /tmp/<TAG> <TAG>` when a patch's applicability must be checked by hand; reference `new-release.sh`.
- Modify: `fork/bench/LESSONS.md` — the one line mentioning `fork/<tag>` branches → `fork/<tag>` tags.

- [ ] Write; run `pre-commit run markdownlint-cli2 --files <each>`; commit `[fork] Document the release model`.

---

### Task 10: Migration script (dry-run tested; execution is a separate, confirmed step)

**Files:**

- Create: `fork/scripts/migrate-to-overlay-main.sh`
- Test: `fork/bench/tests/test_migration.py` (only the tree-building function, on a fixture)

**Interfaces:**

- `migrate-to-overlay-main.sh [--dry-run]` (env `REPO`, `ORIGIN_REMOTE`). Steps, each printed before it runs, all skipped under `--dry-run` except the local branch build:
  1. `git tag -a archive/main-$(date +%F) origin/main -m "main before the overlay-only migration"`; push.
  2. `fork/scripts/freeze-release.sh v0.28.0 <v0.28.0 commit> sha256:673580b7bafed843c2251c5d2bcf0eb2b64a097f40fd0d4ff8dec4f988bd0349 <base digest of vllm/vllm-openai:v0.28.0> <origin/main sha> <export-hash of fork/patches at origin/main> fork/bench/configs/v0.28.0/results/20260830-attempt4.md`; same for `v0.27.1` with `sha256:78817c882a0bd8a1bd8031b48f91ff92381bacee12c5e5e6111eb4b5f143ca2c` and `fork/bench/configs/v0.27.1/results/20260811-attempt4.md`.
  3. Build branch `overlay-main` (function `build_overlay_tree <src-ref> <dst-branch>`): `git checkout --orphan`; `git rm -rq --cached .`; restore from `<src-ref>` only the ledger `add` paths (`git checkout <src-ref> -- FORK.md fork .github/workflows/build-vllm-audio.yml .github/workflows/fork-alignment.yml runs/.gitignore`); `git mv fork/overlay-root/* .` and dotfiles; remove `--pre-migration` from `fork-alignment.yml`; clean the worktree of untracked upstream files (`git clean -fdxq -e .venv -e runs`); commit `[fork] Overlay-only main: the fork's files and nothing else` with `-s` and the Co-Authored-By trailer.
  4. Run `fork/scripts/check-alignment.sh` on `overlay-main` — must pass (uses `fork/v0.28.0` from step 2).
  5. `git push --force-with-lease=main:<origin/main sha> origin overlay-main:main`.
  6. Delete remote branches: `fork/alignment-charter fork/bump-v0.26.0 fork/bump-v0.28.0 fork/gate-ssh-hardening fork/lint-fixes fork/v0.25.1 fork/v0.26.0` and the work branch this plan lands from.
  7. Rulesets via `gh api -X POST repos/Mazyod/vllm/rulesets` (two JSON bodies inline in the script): `fork tags immutable` (`target: tag`, include `refs/tags/fork/*`, rules `update`, `deletion`, `non_fast_forward`), `main protected` (`target: branch`, include `refs/heads/main`, rules `non_fast_forward`, `deletion`, `required_status_checks` with context `alignment`).
  8. Print the local-clone instructions.

- [ ] **Step 1: Test** `build_overlay_tree` on a fixture repo (an "old main" with `vllm/`, `docs/`, `fork/x`, `fork/overlay-root/pyproject.toml`, `FORK.md`, a ledger): the result branch tracks exactly `FORK.md fork/x pyproject.toml` (+ workflows if present), no `vllm/`, and `fork/overlay-root` is gone. Invoke the function via `bash -c 'source script; build_overlay_tree old-main new'` with `MIGRATE_SOURCED=1` guard at the top of the script so sourcing does not run main.
- [ ] **Step 2–4:** fail → implement → pass; `shellcheck`.
- [ ] **Step 5: Commit** — `[fork] One-shot migration to the overlay-only main`. **Do not run the script without `--dry-run`.** Execution is the operator-confirmed step after merge.

---

### Task 11: Whole-suite verification and PR

- [ ] `bash fork/bench/preflight.sh` → PREFLIGHT GREEN.
- [ ] `fork/scripts/check-alignment.sh --pre-migration` → passes on the branch.
- [ ] `pre-commit run --all-files` limited to changed files → clean.
- [ ] `git log --oneline main..HEAD` reads as the task list above; push `fork/release-model`; open the PR with: what/why, the spec path, the test commands and results, "AI assistance was used", and the explicit note that the migration script is landed but **not executed**.

---

## Self-review

- Spec coverage: model/pointer (T6), commit contract (T3), export flags (T2), new-release + four pins (T6), alignment rules 1–5 + pre-migration (T5), workflow guard/labels/freeze/gate_record/rulesets (T7, T10), tooling config (T8), round-trip scope (T8), migration incl. branch deletions and clone advice (T10), docs (T9), tests (T1–T10). Out of scope items untouched.
- Placeholders: T5 and T6 list test names with `...` **as a checklist for the implementer to expand**; the committed files must contain full bodies. Everything else is concrete.
- Names: `export-patches.sh`, `check-release-history.sh`, `check-alignment.sh`, `new-release.sh`, `freeze-release.sh`, `export-hash.sh`, `migrate-to-overlay-main.sh`, `fork/patches/RELEASE` (`tag:`/`release-sha:`), label keys `io.openimage.release-sha` / `io.openimage.patch-export` / `io.openimage.base-digest`, tag `fork/<tag>` — used consistently across tasks.
