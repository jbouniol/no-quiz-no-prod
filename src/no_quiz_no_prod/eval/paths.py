"""Where the evaluation data lives: scenario sets, label files, recorded campaigns and new runs.

Single source of truth for every path and file name under ``eval/``:

- ``eval/scenarios.jsonl``: the **development** set. The prompts and the harness were tuned on it, so a pass rate
  measured on it is optimistic.
- ``eval/scenarios_heldout.jsonl``: the **held-out** set. Never use it to tune a prompt or the harness.
- ``eval/labels/``: the label schema, and the synthetic draft labels used as an offline judge fixture.
- ``eval/campaigns/<name>/``: recorded live campaigns, one directory each, with a ``manifest.json``.
- ``eval/runs/<utc timestamp>_<slug>/``: where a new run writes by default, so it never replaces a recorded campaign.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from no_quiz_no_prod.contracts import PROJECT_ROOT

ScenarioSetName = Literal["dev", "heldout"]

EVAL_DIR: Final = PROJECT_ROOT / "eval"
ACCEPTANCE_CHECKS_DIR: Final = EVAL_DIR / "acceptance"
ACCEPTANCE_REPLAY_PATH: Final = EVAL_DIR / "acceptance_replay.md"
DEFAULT_SCENARIOS_PATH: Final = EVAL_DIR / "scenarios.jsonl"
HELDOUT_SCENARIOS_PATH: Final = EVAL_DIR / "scenarios_heldout.jsonl"
SCENARIO_SET_PATHS: Final[Mapping[ScenarioSetName, Path]] = {
    "dev": DEFAULT_SCENARIOS_PATH,
    "heldout": HELDOUT_SCENARIOS_PATH,
}
DEFAULT_SCENARIO_SET: Final[ScenarioSetName] = "dev"
CUSTOM_SCENARIO_SET: Final = "custom"  # manifest value when --scenarios points at another file

LABELS_DIR: Final = EVAL_DIR / "labels"
SYNTHETIC_LABELS_PATH: Final = LABELS_DIR / "synthetic_draft_labels.json"

CAMPAIGNS_DIR: Final = EVAL_DIR / "campaigns"
LATEST_CAMPAIGN_DIR: Final = CAMPAIGNS_DIR / "04_dev_models_x_topologies"  # the most recent recorded campaign
RUNS_DIR: Final = EVAL_DIR / "runs"

RESULTS_FILENAME: Final = "results.jsonl"  # the primary cohort of a run
COHORT_RESULTS_GLOB: Final = "results_*.jsonl"  # one file per cohort of a multi-cohort run
SUMMARY_FILENAME: Final = "summary.md"
COMPARISON_FILENAME: Final = "comparison.md"
MANIFEST_FILENAME: Final = "manifest.json"
JUDGE_VERDICTS_FILENAME: Final = "judge_verdicts.jsonl"
SWAP_TEST_FILENAME: Final = "swap_test.json"
JUDGE_CALIBRATION_FILENAME: Final = "judge_calibration.json"


def cohort_results_filename(slug: str) -> str:
    """File name of one cohort's rows in a multi-cohort run (matches ``COHORT_RESULTS_GLOB``)."""
    return f"results_{slug}.jsonl"


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` through a temporary file, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def display_path(path: Path) -> str:
    """``path`` relative to the repository root when it is inside it (for reports and manifests)."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(path)


__all__ = [
    "ACCEPTANCE_CHECKS_DIR",
    "ACCEPTANCE_REPLAY_PATH",
    "CAMPAIGNS_DIR",
    "COHORT_RESULTS_GLOB",
    "COMPARISON_FILENAME",
    "CUSTOM_SCENARIO_SET",
    "DEFAULT_SCENARIOS_PATH",
    "DEFAULT_SCENARIO_SET",
    "EVAL_DIR",
    "HELDOUT_SCENARIOS_PATH",
    "JUDGE_CALIBRATION_FILENAME",
    "JUDGE_VERDICTS_FILENAME",
    "LABELS_DIR",
    "LATEST_CAMPAIGN_DIR",
    "MANIFEST_FILENAME",
    "RESULTS_FILENAME",
    "RUNS_DIR",
    "SCENARIO_SET_PATHS",
    "SUMMARY_FILENAME",
    "SWAP_TEST_FILENAME",
    "SYNTHETIC_LABELS_PATH",
    "ScenarioSetName",
    "atomic_write_text",
    "cohort_results_filename",
    "display_path",
]
