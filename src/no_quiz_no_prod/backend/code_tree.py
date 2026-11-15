"""Code-side primitives of the workspace: tree listing, prod -> staging diff, tree copy and the sandboxed test run.

Everything here is pure filesystem/subprocess work with no state of its own; ``SandboxWorkspace`` adds the locking,
the ledger and the persistence around it.

``run_staged_tests`` is the only way agent-written tests are ever executed. It never runs them in place: staging is
copied to a private temporary directory and pytest runs there under OS isolation (``backend.sandbox``). Lock 1 needs
two suites green: the agent's own tests, then the seed's hidden acceptance suite (``seeds/<seed>/acceptance/``),
which is never copied into an area and which no agent tool can read or edit.
"""

import ast
import difflib
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Final, NamedTuple

from no_quiz_no_prod.backend.sandbox import (
    SANDBOX_ROOT_PREFIX,
    TIMEOUT_EXIT_CODE,
    SandboxLimits,
    SandboxResult,
    require_isolation,
    run_isolated,
)
from no_quiz_no_prod.contracts import (
    ACCEPTANCE_DIRNAME,
    ACCEPTANCE_TESTS_TIMEOUT_S,
    STAGED_TESTS_TIMEOUT_S,
    DiffReport,
    FileDiff,
    IsolationLevel,
    _is_hash_ignored,  # the exact ignore rule of compute_tree_sha256: listing and hashing can never disagree
    compute_tree_sha256,
)

DIFF_CONTEXT_LINES: Final = 3
PYTEST_OUTPUT_TAIL_CHARS: Final = 4_000
PYTEST_OUTPUT_BUFFER_BYTES: Final = 64 * 1024  # most the gate ever holds of a run's output, whatever the run prints
PYTEST_ENV_FLAGS: Final = {
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",  # no third-party plugin: fast, and isolated from the project's config
    "NO_COLOR": "1",
}
ACCEPTANCE_FAILED_EXIT_CODE: Final = 1  # pytest's "some tests failed", also used when the suite is missing or tampered
ACCEPTANCE_TEST_GLOB: Final = "test_*.py"
STAGING_COPY_DIRNAME: Final = "staging"
ACCEPTANCE_COPY_DIRNAME: Final = "acceptance"
RUN_DIRNAME: Final = "run"  # pytest's cwd and ini file: next to the copied trees, never inside them
PYTEST_INI_FILENAME: Final = "pytest.ini"
_SUMMARY_PATTERN: Final = re.compile(
    r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b|no tests ran", re.IGNORECASE
)
_COUNT_PATTERN: Final = re.compile(r"\b(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b")


class StagedTestOutcome(NamedTuple):
    """One sandboxed run of staging: the agent's suite, then (full green runs only) the hidden acceptance suite."""

    exit_code: int  # the agent's suite; TIMEOUT_EXIT_CODE when it was killed
    summary: str
    output_tail: str  # the agent's suite only: nothing the acceptance run printed is ever returned
    duration_s: float
    isolation: IsolationLevel
    tested_tree_sha256: str  # compute_tree_sha256 of the private copy that was run
    acceptance_exit_code: int | None  # None: not run (filtered run, red agent suite, or no suite requested)
    acceptance_summary: str  # counts only, written here: never text printed by the test process

    @property
    def passed(self) -> bool:
        """Both suites green. A filtered run can be green too, but never ran the acceptance suite."""
        return self.exit_code == 0 and self.acceptance_exit_code in (None, 0)


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


# ---- the sandboxed test run ------------------------------------------------------------------------------------------
def acceptance_suite_dir(seed_dir: Path) -> Path:
    """``seeds/<seed>/acceptance``: the hidden Lock 1 suite of a seed (``populate_area`` never copies it)."""
    return seed_dir / ACCEPTANCE_DIRNAME


def _summary_line(output: str, exit_code: int) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip().strip("=").strip()
        if stripped and _SUMMARY_PATTERN.search(stripped):
            return stripped
    return f"pytest exited with code {exit_code}"


def _run_suite(
    root: Path,
    tests_dir: Path,
    *,
    isolation: IsolationLevel,
    timeout_s: float,
    pytest_k: str | None = None,
    import_dir: Path | None = None,
) -> SandboxResult:
    """Run pytest on ``tests_dir`` inside the sandbox root, isolated from the repository's pytest config and plugins.

    ``import_dir`` is put on ``sys.path`` (the staged modules, for a suite that lives elsewhere).
    """
    run_dir = root / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    ini_path = run_dir / PYTEST_INI_FILENAME
    ini_path.write_text("[pytest]\n" + (f"pythonpath = {import_dir}\n" if import_dir else ""), encoding="utf-8")
    args = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", str(ini_path), "--rootdir", str(tests_dir)]
    args.append(str(tests_dir))
    if pytest_k:
        args += ["-k", pytest_k]
    limits = SandboxLimits(timeout_s=timeout_s, output_tail_bytes=PYTEST_OUTPUT_BUFFER_BYTES)
    return run_isolated(args, root=root, cwd=run_dir, limits=limits, isolation=isolation, env=PYTEST_ENV_FLAGS)


def _acceptance_test_count(acceptance_dir: Path) -> int:
    """How many tests the hidden suite holds, counted from its source here, outside the sandbox: the count a green
    run must report. The suite keeps to plain module-level ``test_*`` functions so that this is exact."""
    count = 0
    for path in sorted(acceptance_dir.glob(ACCEPTANCE_TEST_GLOB)):
        module = ast.parse(path.read_text(encoding="utf-8"))
        count += sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in module.body)
    return count


def _run_acceptance(
    staging_dir: Path, acceptance_dir: Path, *, tested: str, isolation: IsolationLevel, timeout_s: float
) -> tuple[int, str]:
    """``(exit code, counts-only summary)`` of the hidden suite against a fresh private copy of staging.

    A fresh sandbox root: nothing the agent's suite left behind (files or a stray process) can reach this run. The
    staged modules are imported by the suite, so they run in its process: the verdict is green only when pytest
    exits 0 AND reports exactly the expected number of passed tests and nothing else. That defeats an early exit, a
    module-level skip or a deselection; a staged module written to forge that exact report would be visible in the
    diff the human reviews.
    """
    expected = _acceptance_test_count(acceptance_dir) if acceptance_dir.is_dir() else 0
    if expected == 0:
        return ACCEPTANCE_FAILED_EXIT_CODE, "this seed has no hidden acceptance suite: Lock 1 cannot be earned"
    with tempfile.TemporaryDirectory(prefix=SANDBOX_ROOT_PREFIX) as tmp:
        root = Path(tmp).resolve()
        staging_copy, suite_copy = root / STAGING_COPY_DIRNAME, root / ACCEPTANCE_COPY_DIRNAME
        staging_copy.mkdir()
        copy_tree(staging_dir, staging_copy)
        if compute_tree_sha256(staging_copy) != tested:
            return ACCEPTANCE_FAILED_EXIT_CODE, "staging changed during the test run: run the tests again"
        shutil.copytree(acceptance_dir, suite_copy, ignore=shutil.ignore_patterns("__pycache__"))
        result = _run_suite(root, suite_copy, isolation=isolation, timeout_s=timeout_s, import_dir=staging_copy)
    if result.timed_out:
        return TIMEOUT_EXIT_CODE, f"the hidden acceptance suite timed out after {timeout_s:.0f}s"
    counts = {kind: int(number) for number, kind in _COUNT_PATTERN.findall(_summary_line(result.output, 0))}
    passed = counts.get("passed", 0)
    green = result.exit_code == 0 and counts == {"passed": expected}
    summary = f"{min(passed, expected)} of {expected} hidden acceptance checks passed"
    if green:
        return 0, summary
    hint = "the public behaviour of the seed modules must keep working (functions, signatures, results)"
    return result.exit_code or ACCEPTANCE_FAILED_EXIT_CODE, f"{summary}: {hint}"


def run_staged_tests(
    staging_dir: Path,
    *,
    acceptance_dir: Path | None,
    pytest_k: str | None = None,
    allow_unsafe: bool = False,
    timeout_s: float = STAGED_TESTS_TIMEOUT_S,
    acceptance_timeout_s: float = ACCEPTANCE_TESTS_TIMEOUT_S,
) -> StagedTestOutcome:
    """Run the staged test suite, then the hidden acceptance suite, on private copies under OS isolation.

    The acceptance suite runs only after a full (unfiltered) green run of the agent's suite. Without a working
    isolation backend this raises ``sandbox_unavailable`` before running anything, unless ``allow_unsafe``
    (``RuntimeSettings.unsafe_no_sandbox``); the level used is reported in the outcome.
    """
    isolation = require_isolation(allow_unsafe=allow_unsafe)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=SANDBOX_ROOT_PREFIX) as tmp:
        root = Path(tmp).resolve()
        staging_copy = root / STAGING_COPY_DIRNAME
        staging_copy.mkdir()
        copy_tree(staging_dir, staging_copy)
        tested = compute_tree_sha256(staging_copy)
        result = _run_suite(root, staging_copy, isolation=isolation, timeout_s=timeout_s, pytest_k=pytest_k)
    if result.timed_out:
        summary = f"pytest timed out after {timeout_s:.0f}s"
        output = f"{result.output}\n{summary}"
    else:
        summary, output = _summary_line(result.output, result.exit_code), result.output
    acceptance_exit_code, acceptance_summary = None, ""
    if result.exit_code == 0 and pytest_k is None and acceptance_dir is not None:
        acceptance_exit_code, acceptance_summary = _run_acceptance(
            staging_dir, acceptance_dir, tested=tested, isolation=isolation, timeout_s=acceptance_timeout_s
        )
    return StagedTestOutcome(
        exit_code=result.exit_code,
        summary=summary,
        output_tail=output[-PYTEST_OUTPUT_TAIL_CHARS:],
        duration_s=time.perf_counter() - started,
        isolation=isolation,
        tested_tree_sha256=tested,
        acceptance_exit_code=acceptance_exit_code,
        acceptance_summary=acceptance_summary,
    )


def run_pytest(staging_dir: Path, scratch_dir: Path, *, pytest_k: str | None = None) -> StagedTestOutcome:
    """Transitional entry point of the pre-sandbox ``workspace.py``: already sandboxed, without the acceptance suite.

    ``scratch_dir`` is ignored (the run happens on a private copy). Removed once ``workspace.py`` calls
    ``run_staged_tests``.
    """
    del scratch_dir
    return run_staged_tests(staging_dir, acceptance_dir=None, pytest_k=pytest_k)
