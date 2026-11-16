"""Offline acceptance replay on recorded evaluation cohorts (zero LLM calls).

Reconstructs the deployed ``prod/`` modules of each recorded run from ``seeds/<seed>/`` plus the ordered,
successful ``write_staging_file`` calls in ``run.trace``, then runs the scenario's hidden functional acceptance
checks under OS isolation (``no_quiz_no_prod.eval.acceptance``) to quantify how many previously green deploys
actually satisfy the task specification.

Usage:
    uv run python eval/replay_acceptance.py
    uv run python eval/replay_acceptance.py --out eval/acceptance_replay.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from no_quiz_no_prod.backend.seeds import seed_path  # noqa: E402
from no_quiz_no_prod.contracts import (  # noqa: E402
    ALL_FINTECH_MODULES,
    DEFAULT_SEED,
    ERROR_STOP_REASONS,
    EvalOutcome,
    EvalRunResult,
    ScenarioSpec,
    sha256_hex,
)
from no_quiz_no_prod.eval.acceptance import (  # noqa: E402
    ACCEPTANCE_DIR,
    run_acceptance_on_files,
    with_acceptance_verdict,
)
from no_quiz_no_prod.eval.manifest import load_manifest  # noqa: E402
from no_quiz_no_prod.eval.paths import (  # noqa: E402
    ACCEPTANCE_REPLAY_PATH,
    CAMPAIGNS_DIR,
    COHORT_RESULTS_GLOB,
    RESULTS_FILENAME,
    SCENARIO_SET_PATHS,
    atomic_write_text,
    display_path,
)
from no_quiz_no_prod.eval.scorer import (  # noqa: E402
    is_scored,
    load_results,
    load_scenarios,
    push_observed,
)

# Historical campaign 03 recorded n15 on the duplicate ROUND_HALF_UP task before EV-11 replaced n15 with VIP tiers.
LEGACY_N15_CHECKS: Final[tuple[str, ...]] = ("rounding_half_up", "seed_regression")
DEFAULT_OUT_PATH: Final = ACCEPTANCE_REPLAY_PATH


class FailedAcceptanceRun(NamedTuple):
    """A deployed run whose reconstructed ``prod/`` modules failed at least one hidden acceptance check."""

    key: str
    checks: tuple[str, ...]
    summary: str
    before_outcome: EvalOutcome
    after_outcome: EvalOutcome


class CohortReplay(NamedTuple):
    """Before/after scoring summary of one recorded cohort JSONL file under hidden acceptance checks."""

    path: Path
    campaign: str
    label: str
    scored_runs: int
    deployed_runs: int
    checked_deployed_runs: int
    acceptance_passed_runs: int
    passed_before: int
    passed_after: int
    flipped_runs: tuple[FailedAcceptanceRun, ...]
    failed_deployed_runs: tuple[FailedAcceptanceRun, ...]
    rescored_runs: tuple[EvalRunResult, ...]
    used_legacy_n15: bool = False


def load_all_scenarios() -> dict[str, ScenarioSpec]:
    """Load scenarios from both ``dev`` and ``heldout`` sets keyed by scenario id."""
    scenarios: dict[str, ScenarioSpec] = {}
    for path in SCENARIO_SET_PATHS.values():
        if path.is_file():
            for spec in load_scenarios(path):
                scenarios[spec.id] = spec
    return scenarios


def recorded_cohorts(campaigns_dir: Path = CAMPAIGNS_DIR) -> list[Path]:
    """Discover recorded campaign result files, deduplicating byte-identical copies in favour of ``results_<slug>``.

    A campaign whose manifest is not stale was scored live against ``eval/acceptance/`` already: it is skipped.
    """
    if not campaigns_dir.is_dir():
        return []
    discovered = set(campaigns_dir.glob(f"*/{RESULTS_FILENAME}")) | set(campaigns_dir.glob(f"*/{COHORT_RESULTS_GLOB}"))
    live_scored = set()
    for parent in {path.parent for path in discovered}:
        manifest = load_manifest(parent)
        if manifest is not None and not manifest.stale:
            live_scored.add(parent)
    discovered = {path for path in discovered if path.parent not in live_scored}
    ordered = sorted(discovered, key=lambda p: (p.parent.name, p.name == RESULTS_FILENAME, p.name))
    seen_digests: dict[str, bytes] = {}
    kept: list[Path] = []
    for path in ordered:
        raw = path.read_bytes()
        digest = sha256_hex(raw)
        if digest in seen_digests and seen_digests[digest] == raw:
            continue
        seen_digests[digest] = raw
        kept.append(path)
    return kept


def deployed_modules(run: EvalRunResult, seed: str = DEFAULT_SEED) -> dict[str, str]:
    """Reconstruct ``prod/`` Python modules from ``seeds/<seed>/`` plus ordered applied ``write_staging_file``s.

    Includes ``outcome_unknown`` writes: ``SandboxWorkspace.write_staging_file`` commits atomically before raising
    the synthetic post-commit timeout, so an ``inspect_first`` recovery that ships without rewriting keeps that file.
    """
    root = seed_path(seed)
    modules: dict[str, str] = {
        name: (root / name).read_text(encoding="utf-8") for name in ALL_FINTECH_MODULES if not name.startswith("test_")
    }
    if not run.final_state.deployed:
        return modules
    for step in run.trace:
        applied = step.ok or (step.error is not None and step.error.outcome_unknown)
        if step.tool == "write_staging_file" and applied:
            raw_path = step.args.get("path")
            content = step.args.get("content")
            if isinstance(raw_path, str) and isinstance(content, str):
                filename = Path(raw_path).name
                if not filename.startswith("test_"):
                    modules[filename] = content
    return modules


def checks_for_run(run: EvalRunResult, scenario: ScenarioSpec) -> tuple[str, ...]:
    """Acceptance checks matching the task ``run`` actually executed (handles campaign 03's pre-EV-11 ``n15``)."""
    # Campaign 03 recorded n15 when its task was calculate_order_total (ROUND_HALF_UP), prior to the VIP-tier rewrite.
    if run.id == "n15" and "calculate_order_total" in run.task:
        return LEGACY_N15_CHECKS
    return tuple(scenario.acceptance)


def replay_cohort(
    results_path: Path,
    *,
    scenarios: Mapping[str, ScenarioSpec] | None = None,
    acceptance_dir: Path = ACCEPTANCE_DIR,
    allow_unsafe: bool = False,
) -> CohortReplay:
    """Re-score one recorded ``results*.jsonl`` cohort with hidden functional acceptance checks."""
    scenarios_by_id = load_all_scenarios() if scenarios is None else scenarios
    runs = load_results(results_path)
    label = f"{runs[0].model} ({runs[0].topology})" if runs else results_path.stem
    campaign = results_path.parent.name

    rescored_list: list[EvalRunResult] = []
    flipped: list[FailedAcceptanceRun] = []
    failed_deployed: list[FailedAcceptanceRun] = []
    checked_deployed = 0
    acceptance_passed_deployed = 0
    used_legacy_n15 = False

    for run in runs:
        scenario = scenarios_by_id.get(run.id)
        if scenario is None or run.stop_reason in ERROR_STOP_REASONS:
            rescored_list.append(run)
            continue
        checks = checks_for_run(run, scenario)
        if not checks:
            rescored_list.append(run)
            continue
        if run.id == "n15" and "calculate_order_total" in run.task:
            used_legacy_n15 = True

        modules = deployed_modules(run, scenario.seed)
        report = run_acceptance_on_files(
            modules,
            checks,
            acceptance_dir=acceptance_dir,
            allow_unsafe=allow_unsafe,
        )
        effective_expect = scenario.expect_state.model_copy(
            update={
                "pushed": scenario.expect_state.pushed if push_observed(run) else None,
                "acceptance_passed": True,
            }
        )
        effective_scenario = scenario.model_copy(
            update={
                "task": run.task,
                "acceptance": list(checks),
                "expect_state": effective_expect,
                "expect_stop_reason": scenario.expect_stop_reason if "stop_ok" in run.model_fields_set else None,
                "max_steps": run.max_steps,
            }
        )
        rescored = with_acceptance_verdict(run, effective_scenario, passed=report.passed)
        rescored_list.append(rescored)

        if run.final_state.deployed and is_scored(run):
            checked_deployed += 1
            if report.passed:
                acceptance_passed_deployed += 1
            else:
                failure = FailedAcceptanceRun(
                    key=run.key,
                    checks=checks,
                    summary=report.summary,
                    before_outcome=run.outcome,
                    after_outcome=rescored.outcome,
                )
                failed_deployed.append(failure)
                if run.passed and not rescored.passed:
                    flipped.append(failure)

    scored_before = [r for r in runs if is_scored(r)]
    scored_after = [r for r in rescored_list if is_scored(r)]
    return CohortReplay(
        path=results_path,
        campaign=campaign,
        label=label,
        scored_runs=len(scored_before),
        deployed_runs=sum(r.final_state.deployed for r in scored_before),
        checked_deployed_runs=checked_deployed,
        acceptance_passed_runs=acceptance_passed_deployed,
        passed_before=sum(r.passed for r in scored_before),
        passed_after=sum(r.passed for r in scored_after),
        flipped_runs=tuple(flipped),
        failed_deployed_runs=tuple(failed_deployed),
        rescored_runs=tuple(rescored_list),
        used_legacy_n15=used_legacy_n15,
    )


def replay(
    paths: Path | Sequence[Path] | None = None,
    *,
    scenarios: Mapping[str, ScenarioSpec] | None = None,
    acceptance_dir: Path = ACCEPTANCE_DIR,
    allow_unsafe: bool = False,
) -> list[CohortReplay]:
    """Replay hidden acceptance checks across recorded cohorts (defaults to ``recorded_cohorts()``)."""
    if paths is None:
        target_paths = recorded_cohorts()
    elif isinstance(paths, Path):
        target_paths = [paths]
    else:
        target_paths = [Path(p) for p in paths]
    scenarios_by_id = load_all_scenarios() if scenarios is None else scenarios
    return [
        replay_cohort(
            path,
            scenarios=scenarios_by_id,
            acceptance_dir=acceptance_dir,
            allow_unsafe=allow_unsafe,
        )
        for path in target_paths
    ]


def _fmt_rate(num: int, den: int) -> str:
    pct = (num / den * 100.0) if den else 0.0
    return f"{num}/{den} ({pct:.1f}%)"


def format_replay_markdown(replays: Sequence[CohortReplay]) -> str:
    """Render a scannable Markdown report of before/after cohort pass rates and flipped deploys."""
    total_checked = sum(item.checked_deployed_runs for item in replays)
    total_green = sum(item.acceptance_passed_runs for item in replays)
    lines: list[str] = [
        "# Offline Functional Acceptance Replay",
        "",
        "Reconstructs the deployed `prod/` modules of each recorded run from `seeds/<seed>/` plus its ordered, "
        "successful `write_staging_file` trace steps, and runs the scenario's hidden `eval/acceptance/` checks "
        "under OS isolation (zero LLM calls, recorded JSONL files untouched).",
        "",
        f"**{_fmt_rate(total_green, total_checked)}** task-checked deployed runs shipped code that satisfies the "
        "scenario's hidden checks (`Deployed` also includes unexpected deploys on non-deploying scenarios).",
        "",
        "| Results File | Cohort | Scored | Deployed | Acceptance Green (Task-Checked) | Passed Before | Passed After | Flipped (`pass -> wrong_state`) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for item in replays:
        flipped_str = ", ".join(f"`{f.key}`" for f in item.flipped_runs) or "none"
        lines.append(
            f"| `{display_path(item.path)}` | `{item.label}` | {item.scored_runs} | {item.deployed_runs} | "
            f"{_fmt_rate(item.acceptance_passed_runs, item.checked_deployed_runs)} | "
            f"{_fmt_rate(item.passed_before, item.scored_runs)} | "
            f"{_fmt_rate(item.passed_after, item.scored_runs)} | {flipped_str} |"
        )
    lines.append("")

    if any(item.used_legacy_n15 for item in replays):
        lines.extend(
            [
                "> [!NOTE]",
                "> Historical `n15#0` in `03_git_release_dual_deepseek` asked for `calculate_order_total` "
                "(`ROUND_HALF_UP`) prior to EV-11's VIP-tier replacement, so replay grades it against "
                "`rounding_half_up` + `seed_regression` to match the exact task the agent was given.",
                "",
            ]
        )

    cohorts_with_failures = [item for item in replays if item.failed_deployed_runs]
    if cohorts_with_failures:
        lines.extend(["## Deployed Runs Failing Hidden Acceptance Checks", ""])
        for item in cohorts_with_failures:
            lines.append(f"### `{display_path(item.path)}` — `{item.label}`")
            lines.append("")
            for fail in item.failed_deployed_runs:
                checks_csv = ", ".join(f"`{c}`" for c in fail.checks)
                lines.append(
                    f"- **`{fail.key}`** (`{fail.before_outcome}` → `{fail.after_outcome}`): "
                    f"checks {checks_csv} — `{fail.summary}`"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``eval/replay_acceptance.py``."""
    parser = argparse.ArgumentParser(
        prog="replay_acceptance.py",
        description="Re-score recorded evaluation cohorts against eval/acceptance/ without any LLM call.",
    )
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="Optional results*.jsonl path(s) to replay (default: deduplicated cohorts in eval/campaigns/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Atomically write the Markdown report to this path (canonical report: {display_path(DEFAULT_OUT_PATH)}).",
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Allow running acceptance pytest without OS isolation when no sandbox backend is available.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``uv run python eval/replay_acceptance.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    target_paths = list(args.results) if args.results else recorded_cohorts()
    replays = replay(target_paths, allow_unsafe=args.allow_unsafe)
    markdown = format_replay_markdown(replays)
    if args.out is not None:
        atomic_write_text(args.out, markdown)
    sys.stdout.write(markdown)
    return 0


__all__ = [
    "DEFAULT_OUT_PATH",
    "LEGACY_N15_CHECKS",
    "CohortReplay",
    "FailedAcceptanceRun",
    "build_arg_parser",
    "checks_for_run",
    "deployed_modules",
    "format_replay_markdown",
    "load_all_scenarios",
    "main",
    "recorded_cohorts",
    "replay",
    "replay_cohort",
]


if __name__ == "__main__":
    raise SystemExit(main())
