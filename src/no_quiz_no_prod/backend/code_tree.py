"""Code-side primitives of the sandbox: tree listing, prod -> staging diff, tree copy and the isolated pytest run.

Everything here is pure filesystem/subprocess work with no state of its own; ``SandboxWorkspace`` adds the locking,
the ledger and the persistence around it.
"""

import difflib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Final, NamedTuple

from no_quiz_no_prod.contracts import (
    DiffReport,
    FileDiff,
    _is_hash_ignored,  # the exact ignore rule of compute_tree_sha256: listing and hashing can never disagree
    compute_tree_sha256,
)

DIFF_CONTEXT_LINES: Final = 3
PYTEST_TIMEOUT_S: Final = 60
PYTEST_TIMEOUT_EXIT_CODE: Final = 124  # the `timeout(1)` convention: the run was killed, it did not pass
PYTEST_OUTPUT_TAIL_CHARS: Final = 4_000
# The agent's test code runs in this subprocess: it only gets a minimal environment, never the parent's API keys.
PYTEST_ENV_PASSTHROUGH: Final = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT")
PYTEST_ENV_FLAGS: Final = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",  # no third-party plugin: fast, and isolated from the project's config
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "NO_COLOR": "1",
}
_SUMMARY_PATTERN: Final = re.compile(
    r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b|no tests ran", re.IGNORECASE
)


class PytestOutcome(NamedTuple):
    exit_code: int
    summary: str
    output_tail: str
    duration_s: float


def tree_files(root: Path) -> dict[str, Path]:
    """``{relative POSIX path: path}`` of the files ``compute_tree_sha256`` hashes (empty when ``root`` is absent)."""
    if not root.is_dir():
        return {}
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not _is_hash_ignored(PurePosixPath(relative)):
            files[relative] = path
    return files


def copy_tree(source: Path, destination: Path) -> None:
    """Copy the hashed files of ``source`` into the (existing, empty) ``destination`` directory."""
    for relative, path in tree_files(source).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _diff_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    return [line if line.endswith("\n") else f"{line}\n" for line in lines]


def build_diff_report(prod_dir: Path, staging_dir: Path) -> DiffReport:
    """The prod -> staging change: both tree hashes plus one unified diff per changed file."""
    prod_files, staging_files = tree_files(prod_dir), tree_files(staging_dir)
    files: list[FileDiff] = []
    for relative in sorted(prod_files.keys() | staging_files.keys()):
        before, after = prod_files.get(relative), staging_files.get(relative)
        if before is not None and after is not None and before.read_bytes() == after.read_bytes():
            continue
        lines = list(
            difflib.unified_diff(
                _diff_lines(before),
                _diff_lines(after),
                fromfile=f"a/prod/{relative}",
                tofile=f"b/staging/{relative}",
                n=DIFF_CONTEXT_LINES,
            )
        )
        body = lines[2:]  # skip the ---/+++ header
        files.append(
            FileDiff(
                path=relative,
                additions=sum(line.startswith("+") for line in body),
                deletions=sum(line.startswith("-") for line in body),
                unified_diff="".join(lines),
            )
        )
    return DiffReport(
        prod_tree_sha256=compute_tree_sha256(prod_dir),
        staging_tree_sha256=compute_tree_sha256(staging_dir),
        files=files,
    )


def _summary_line(output: str, exit_code: int) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip().strip("=").strip()
        if stripped and _SUMMARY_PATTERN.search(stripped):
            return stripped
    return f"pytest exited with code {exit_code}"


def _as_text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    return stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else stream


def run_pytest(staging_dir: Path, scratch_dir: Path, *, pytest_k: str | None = None) -> PytestOutcome:
    """Run the staging test suite in a subprocess, isolated from the repository's pytest config and plugins.

    The run uses an empty ini file in ``scratch_dir`` (outside the hashed trees) and runs from there, so a test that
    writes files never changes the staging tree. stdin is closed: a stdio MCP server's protocol stream is never
    inherited.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ini_path = scratch_dir / "pytest.ini"
    ini_path.write_text("[pytest]\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        str(ini_path),
        "--rootdir",
        str(staging_dir),
        str(staging_dir),
    ]
    if pytest_k:
        command += ["-k", pytest_k]
    env = {key: os.environ[key] for key in PYTEST_ENV_PASSTHROUGH if key in os.environ} | PYTEST_ENV_FLAGS
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=scratch_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT_S,
            check=False,
        )
        exit_code, output = completed.returncode, completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as timeout:
        exit_code = PYTEST_TIMEOUT_EXIT_CODE
        output = f"{_as_text(timeout.stdout)}{_as_text(timeout.stderr)}\npytest timed out after {PYTEST_TIMEOUT_S}s"
    duration_s = time.perf_counter() - started
    summary = (
        f"pytest timed out after {PYTEST_TIMEOUT_S}s"
        if exit_code == PYTEST_TIMEOUT_EXIT_CODE
        else _summary_line(output, exit_code)
    )
    return PytestOutcome(exit_code, summary, output[-PYTEST_OUTPUT_TAIL_CHARS:], duration_s)
