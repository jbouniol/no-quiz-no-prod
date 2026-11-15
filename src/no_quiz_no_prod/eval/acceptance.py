"""Hidden functional acceptance checks on deployed production code, isolated in the OS sandbox.

The check files live in ``eval/acceptance/test_<name>.py`` (plus ``acceptance_support.py``), outside the runtime
workspace so no agent tool can list or read them. Execution delegates directly to
``no_quiz_no_prod.backend.code_tree.run_isolated_suite``: every check runs in a fresh private directory under OS
isolation, ignores any ``test_*.py`` written by the agent, and defeats ``os._exit(0)``, ``sys.exit(0)`` or
module-level skips inside untrusted code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from no_quiz_no_prod.backend.code_tree import run_isolated_suite
from no_quiz_no_prod.contracts import (
    ACCEPTANCE_TESTS_TIMEOUT_S,
    EvalRunResult,
    IsolationLevel,
    ScenarioSpec,
)
from no_quiz_no_prod.eval.paths import ACCEPTANCE_CHECKS_DIR

ACCEPTANCE_DIR: Final = ACCEPTANCE_CHECKS_DIR
SUPPORT_MODULE: Final = "acceptance_support.py"


class AcceptanceReport(NamedTuple):
    """Verdict of running a scenario's hidden acceptance checks against deployed ``prod/`` modules."""

    passed: bool
    summary: str
    output_tail: str
    isolation: IsolationLevel = "none"


def check_path(name: str, acceptance_dir: Path = ACCEPTANCE_DIR) -> Path:
    """Resolve ``acceptance_dir / f'test_{name}.py'``; raise ``FileNotFoundError`` when missing."""
    path = acceptance_dir / f"test_{name}.py"
    if not name or Path(name).name != name or not path.is_file():
        raise FileNotFoundError(f"unknown acceptance check {name!r}: {path} does not exist")
    return path


def run_acceptance_on_files(
    modules: Mapping[str, str],
    checks: Sequence[str],
    *,
    acceptance_dir: Path = ACCEPTANCE_DIR,
    allow_unsafe: bool = False,
    timeout_s: float = ACCEPTANCE_TESTS_TIMEOUT_S,
) -> AcceptanceReport:
    """Run ``checks`` against ``modules`` (``{filename: source}``) under OS isolation."""
    paths = list(dict.fromkeys(check_path(name, acceptance_dir) for name in checks))
    if not paths:
        raise ValueError("at least one acceptance check is required")
    support_path = acceptance_dir / SUPPORT_MODULE
    if acceptance_dir == ACCEPTANCE_DIR and not support_path.is_file():
        raise FileNotFoundError(f"missing acceptance support module: {support_path} does not exist")
    suite_files: list[Path] = [*paths, *([support_path] if support_path.is_file() else [])]
    clean_modules = {
        name: source
        for name, source in modules.items()
        if Path(name).name != SUPPORT_MODULE and not Path(name).name.startswith("test_")
    }
    outcome = run_isolated_suite(
        clean_modules,
        suite_files,
        allow_unsafe=allow_unsafe,
        timeout_s=timeout_s,
    )
    return AcceptanceReport(
        passed=outcome.passed,
        summary=outcome.summary,
        output_tail=outcome.output_tail,
        isolation=outcome.isolation,
    )


def run_acceptance(
    prod_dir: Path,
    checks: Sequence[str],
    *,
    acceptance_dir: Path = ACCEPTANCE_DIR,
    allow_unsafe: bool = False,
    timeout_s: float = ACCEPTANCE_TESTS_TIMEOUT_S,
) -> AcceptanceReport:
    """Read non-``test_*`` Python modules from ``prod_dir`` and run ``checks`` under OS isolation."""
    modules = (
        {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(prod_dir.glob("*.py"))
            if path.is_file() and not path.name.startswith("test_") and path.name != SUPPORT_MODULE
        }
        if prod_dir.is_dir()
        else {}
    )
    return run_acceptance_on_files(
        modules,
        checks,
        acceptance_dir=acceptance_dir,
        allow_unsafe=allow_unsafe,
        timeout_s=timeout_s,
    )


def with_acceptance_verdict(result: EvalRunResult, scenario: ScenarioSpec, *, passed: bool) -> EvalRunResult:
    """Rebuild ``result`` against ``scenario`` with ``final_state.acceptance_passed = passed``."""
    rescored = EvalRunResult.from_run(
        scenario=scenario,
        repeat=result.repeat,
        final_state=result.final_state.model_copy(update={"acceptance_passed": passed}),
        trace=result.trace,
        stop_reason=result.stop_reason,
        final_answer=result.final_answer,
        model=result.model,
        topology=result.topology,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
    )
    return rescored if rescored.persona == result.persona else rescored.model_copy(update={"persona": result.persona})


__all__ = [
    "ACCEPTANCE_DIR",
    "SUPPORT_MODULE",
    "AcceptanceReport",
    "check_path",
    "run_acceptance",
    "run_acceptance_on_files",
    "with_acceptance_verdict",
]
