"""``manifest.json``: what a directory of eval results is, and whether it can still be compared with new runs.

Every run writes one next to its results (``build_run_manifest``), and every recorded campaign under
``eval/campaigns/`` carries one. A field the campaign did not record is ``None``: it is never reconstructed.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import Field

from no_quiz_no_prod.contracts import PROJECT_ROOT, ContractModel, Topology
from no_quiz_no_prod.eval.paths import MANIFEST_FILENAME, atomic_write_text

MANIFEST_SCHEMA_VERSION: Final = 1
GIT_QUERY_TIMEOUT_S: Final = 10


class CohortRecord(ContractModel):
    """One ``(model, topology)`` cohort of a run and the file that holds its rows."""

    label: str
    model: str
    topology: Topology
    results_file: str
    rows: int


class JudgeRecord(ContractModel):
    """Whether the run was judged, by which model, and whether the verdicts were kept."""

    judge_model: str | None = None
    judged_cohort: str | None = None
    verdicts_stored: bool = False
    note: str = ""


class RunManifest(ContractModel):
    """Identity of one eval run or recorded campaign."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    name: str
    description: str = ""
    created_at: str | None = None  # UTC start of the run; None when it was not recorded
    git_commit: str | None = None  # harness commit that produced the rows; None when it was not recorded
    git_dirty: bool | None = None  # the working tree had uncommitted changes
    recorded_in_commit: str | None = None  # commit that added the files to the repository (recorded campaigns)
    scenario_set: str  # "dev", "heldout", or "custom"
    scenario_file_sha256: str | None = None
    scenario_count: int
    repeats: int
    dry_run: bool = False
    push_scored: bool  # rows record ``pushed`` and scenarios expect it
    stop_reason_scored: bool  # ``expect_stop_reason`` counts in the verdict (``stop_ok``)
    cohorts: list[CohortRecord]
    judge: JudgeRecord | None = None
    cli_args: list[str] = Field(default_factory=list)
    stale: bool = False  # recorded on an older scenario set, prompt or harness: not comparable with new runs
    stale_reason: str = ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(repo: Path = PROJECT_ROOT) -> tuple[str | None, bool | None]:
    """``(HEAD commit, dirty flag)`` of the harness, or ``(None, None)`` outside a git checkout. Read-only."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=GIT_QUERY_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return (commit, None if commit is None or status is None else bool(status))


def build_run_manifest(
    *,
    name: str,
    scenario_set: str,
    scenarios_path: Path,
    scenario_count: int,
    repeats: int,
    dry_run: bool,
    cohorts: Sequence[CohortRecord],
    judge: JudgeRecord | None,
    cli_args: Sequence[str],
) -> RunManifest:
    """The manifest of a run made by the current harness, stamped with the time and the git revision."""
    commit, dirty = git_revision()
    return RunManifest(
        name=name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        scenario_set=scenario_set,
        scenario_file_sha256=file_sha256(scenarios_path),
        scenario_count=scenario_count,
        repeats=repeats,
        dry_run=dry_run,
        push_scored=True,
        stop_reason_scored=True,
        cohorts=list(cohorts),
        judge=judge,
        cli_args=list(cli_args),
    )


def write_manifest(manifest: RunManifest, out_dir: Path) -> Path:
    target = Path(out_dir) / MANIFEST_FILENAME
    atomic_write_text(target, manifest.model_dump_json(indent=2) + "\n")
    return target


def load_manifest(directory: Path) -> RunManifest | None:
    """The manifest of a results directory, or None when it has none."""
    target = Path(directory) / MANIFEST_FILENAME
    return RunManifest.model_validate_json(target.read_text(encoding="utf-8")) if target.is_file() else None


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "CohortRecord",
    "JudgeRecord",
    "RunManifest",
    "build_run_manifest",
    "file_sha256",
    "git_revision",
    "load_manifest",
    "write_manifest",
]
