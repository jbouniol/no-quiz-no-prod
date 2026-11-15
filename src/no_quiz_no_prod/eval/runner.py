"""Isolated-workspace evaluation runner and CLI orchestration for the scenario sets.

Backs ``eval/run_eval.py`` so both ``uv run python eval/run_eval.py ...`` and Python callers import
the same single source of truth from ``no_quiz_no_prod.eval.runner``.

A run writes into its own new directory (``eval/runs/<utc timestamp>_<slug>/`` by default) and refuses a directory
that already holds results, so a smoke test or a dry run can never replace a recorded campaign. Next to the rows it
writes a ``manifest.json`` (scenario set, harness commit, repeats) and, when judged, everything the judge said.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, get_args

from no_quiz_no_prod.contracts import (
    ALL_FINTECH_MODULES,
    LLM_AGENT_ROLES,
    AgentRole,
    EvalRunResult,
    FintechModule,
    RuleCode,
    RunFinalState,
    RuntimeSettings,
    ScenarioSpec,
    StopReason,
    ToolCallTrace,
    ToolErrorPayload,
    Topology,
    idem_key,
    sha256_hex,
)
from no_quiz_no_prod.eval.judge import (
    DEFAULT_LABELING_BATCH_SIZE,
    JudgeCalibrationReport,
    JudgedRunRecord,
    SwapTestResult,
    calibrate_judge_on_human_labels,
    calibrate_judge_on_synthetic_labels,
    export_labeling_batch,
    judge_batch,
    swap_test,
)
from no_quiz_no_prod.eval.manifest import CohortRecord, JudgeRecord, build_run_manifest, write_manifest
from no_quiz_no_prod.eval.paths import (
    COHORT_RESULTS_GLOB,
    CUSTOM_SCENARIO_SET,
    DEFAULT_SCENARIO_SET,
    HELDOUT_SCENARIOS_PATH,
    RESULTS_FILENAME,
    RUNS_DIR,
    SCENARIO_SET_PATHS,
    ScenarioSetName,
    atomic_write_text,
    cohort_results_filename,
    display_path,
)
from no_quiz_no_prod.eval.report import (
    write_comparison_artifacts,
    write_judge_artifacts,
    write_summary_artifacts,
)
from no_quiz_no_prod.eval.scorer import (
    is_scored,
    load_results,
    load_scenario_set,
    load_scenarios,
    scenario_set_overlap,
    summarize_cohort,
    write_results,
)
from no_quiz_no_prod.llm import LLMError, estimate_cost_usd, resolve_provider

logger = logging.getLogger(__name__)

McpTransport = Literal["stdio", "inprocess"]
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
# Three repeats is the smallest k that separates pass^k (reliability) from pass@1 (one lucky run).
DEFAULT_REPEATS: Final = 3
EXIT_OK: Final = 0
EXIT_USAGE: Final = 2  # bad arguments, missing key, or an output directory that already holds results
EXIT_HARNESS_ERROR: Final = 3  # at least one run hit a harness bug: the report is written but must not be trusted


@dataclass(frozen=True)
class CohortConfig:
    """Target ``(model, topology)`` cohort to execute on the Golden Set."""

    model: str
    topology: Topology

    @property
    def label(self) -> str:
        return f"{self.model} ({self.topology})"

    @property
    def slug(self) -> str:
        cleaned = _SLUG_RE.sub("_", f"{self.topology}_{self.model}").strip("_").lower()
        if len(cleaned) <= 48:
            return cleaned or self.topology
        return f"{cleaned[:39]}_{sha256_hex(cleaned)[:8]}"


# ---- deterministic offline --dry-run simulator ---------------------------------------------------------------------
def _infer_primary_module(task: str) -> FintechModule:
    for mod in ALL_FINTECH_MODULES:
        if mod in task:
            return mod
    return "orders.py"


def _infer_rule_code(task: str, scenario: ScenarioSpec) -> RuleCode:
    match = re.search(r"RULE-\d{3}", task)
    if match:
        return match.group(0)
    if scenario.injection and scenario.injection.target.startswith("RULE-"):
        return scenario.injection.target
    return "RULE-001"


def simulate_dry_run_scenario(
    scenario: ScenarioSpec,
    *,
    repeat: int = 0,
    model: str,
    topology: Topology,
) -> EvalRunResult:
    """Build a realistic, strictly validated ``EvalRunResult`` with zero LLM or MCP network calls."""
    coder_role: AgentRole = "coder" if topology == "dual" else "single"
    examiner_role: AgentRole = "examiner" if topology == "dual" else "single"
    thread_id = f"dry_{scenario.id}_r{repeat}"

    seed_hash = sha256_hex(f"seed:{scenario.seed}")
    staged_hash = sha256_hex(f"staged:{scenario.id}:{repeat}:{topology}")
    diff_hash = sha256_hex(f"{seed_hash}:{staged_hash}")

    primary_module = _infer_primary_module(scenario.task)
    rule_code = _infer_rule_code(scenario.task, scenario)
    inj_target = scenario.injection.target if scenario.injection else None

    trace: list[ToolCallTrace] = []
    stop_reason: StopReason

    def add_step(
        agent: AgentRole,
        server: Literal["code-mcp", "notion-quiz-mcp"],
        tool: str,
        args: dict[str, object],
        *,
        is_write: bool = False,
        error: ToolErrorPayload | None = None,
        attempts: int = 1,
        replayed: bool = False,
        contains_injection: bool = False,
    ) -> None:
        trace.append(
            ToolCallTrace(
                step=len(trace) + 1,
                agent=agent,
                server=server,
                tool=tool,
                args=dict(args),
                result=None if error is not None else {"ok": True, "diff_sha256": diff_hash},
                error=error,
                latency_s=0.05 * attempts,
                idempotency_key=idem_key(thread_id, tool, dict(args)) if is_write else None,
                attempts=attempts,
                replayed=replayed,
                contains_injection=contains_injection,
            )
        )

    rate_limit_attempts = 2 if scenario.fault == "rate_limit" else 1
    # push_rejected: the harness repeats the refused release push once, with the same idempotency key
    deploy_attempts = 2 if scenario.fault == "push_rejected" else 1
    release_sha = sha256_hex(f"release:{diff_hash}")[:40]  # stands in for the pushed release commit id
    add_step(
        coder_role,
        "notion-quiz-mcp",
        "search_notion_rules",
        {"query": primary_module.removesuffix(".py")},
        attempts=rate_limit_attempts,
    )
    add_step(
        coder_role,
        "notion-quiz-mcp",
        "read_notion_rule",
        {"rule_code": rule_code},
        contains_injection=(inj_target == rule_code),
    )
    add_step(
        coder_role,
        "code-mcp",
        "read_code_file",
        {"path": primary_module, "area": "staging"},
        contains_injection=(inj_target == primary_module),
    )

    if scenario.kind == "ambiguous":
        final_state = RunFinalState(
            prod_tree_sha256=seed_hash,
            staging_tree_sha256=seed_hash,
            prod_changed=False,
            pushed=False,
        )
        answer = (
            f"Per {rule_code}, a required business parameter is missing in your request for {primary_module}. "
            "Could you clarify the exact value before I modify staging?"
        )
        stop_reason = "clarification_needed"
    elif scenario.kind == "impossible":
        final_state = RunFinalState(
            prod_tree_sha256=seed_hash,
            staging_tree_sha256=seed_hash,
            prod_changed=False,
            pushed=False,
        )
        answer = (
            f"I must refuse this request: modifying {primary_module} as asked violates {rule_code} "
            "or closed sandbox security boundaries. Staging and production remain untouched."
        )
        stop_reason = "refused"
    else:
        write_args: dict[str, object] = {"path": primary_module, "content": f"# patched {scenario.id}\n"}
        if scenario.fault == "write_timeout":
            # retryable: the harness repeats the call with the same key and gets the replay; the model sees success
            add_step(coder_role, "code-mcp", "write_staging_file", write_args, is_write=True, attempts=2, replayed=True)
        elif scenario.fault == "write_outcome_unknown":
            # not retryable: the model reads the error and, here, inspects staging before going on
            add_step(
                coder_role,
                "code-mcp",
                "write_staging_file",
                write_args,
                is_write=True,
                error=ToolErrorPayload(
                    code="timeout",
                    message="staging write timed out before confirmation",
                    retryable=False,
                    outcome_unknown=True,
                ),
            )
            add_step(coder_role, "code-mcp", "get_staging_diff", {})
        else:
            add_step(coder_role, "code-mcp", "write_staging_file", write_args, is_write=True)

        add_step(
            coder_role,
            "code-mcp",
            "write_staging_file",
            {"path": "test_fintech.py", "content": f"# tests for {scenario.id}\n"},
            is_write=True,
        )
        add_step(coder_role, "code-mcp", "run_staging_tests", {}, is_write=True)
        add_step(
            examiner_role,
            "notion-quiz-mcp",
            "upsert_ship_log_doc",
            {
                "title": f"Fix {scenario.id}",
                "diff_sha256": diff_hash,
                "changed_files": [primary_module, "test_fintech.py"],
                "applied_rules": [rule_code],
                "targeted_weaknesses": scenario.target_skill_tags or ["decimal_rounding"],
                "doc_version": "v1",
                "status": "Quiz Pending",
            },
            is_write=True,
        )
        add_step(
            examiner_role,
            "notion-quiz-mcp",
            "create_comprehension_quiz",
            {"diff_sha256": diff_hash, "attempt": 1},
            is_write=True,
        )

        if scenario.fault == "push_unavailable":
            # every push is refused: the gate passed, the release did not, prod stays untouched
            add_step(
                "system",
                "code-mcp",
                "deploy_to_production",
                {"expected_diff_sha256": diff_hash, "release_note": f"Ship {scenario.id}"},
                is_write=True,
                attempts=2,
                error=ToolErrorPayload(
                    code="upstream_error",
                    message="the release remote refused the push",
                    retryable=True,
                ),
            )
            final_state = RunFinalState(
                prod_tree_sha256=seed_hash,
                staging_tree_sha256=staged_hash,
                prod_changed=False,
                tests_passed=True,
                ship_log_status="Quiz Pending",
                doc_version="v1",
                quiz_attempts=1,
                quiz_passed=True,
                deployed=False,
                pushed=False,
            )
            answer = f"The release of {scenario.id} could not be pushed; production is untouched."
            stop_reason = "release_failed"
        elif scenario.persona == "fail_once_then_pass":
            add_step(
                examiner_role,
                "notion-quiz-mcp",
                "upsert_ship_log_doc",
                {
                    "title": f"Fix {scenario.id}",
                    "diff_sha256": diff_hash,
                    "changed_files": [primary_module, "test_fintech.py"],
                    "applied_rules": [rule_code],
                    "targeted_weaknesses": scenario.target_skill_tags or ["decimal_rounding"],
                    "doc_version": "v2",
                    "status": "Remediation v2",
                },
                is_write=True,
            )
            add_step(
                examiner_role,
                "notion-quiz-mcp",
                "create_comprehension_quiz",
                {"diff_sha256": diff_hash, "attempt": 2},
                is_write=True,
            )
            add_step(
                "system",
                "code-mcp",
                "deploy_to_production",
                {"expected_diff_sha256": diff_hash, "release_note": f"Ship {scenario.id}"},
                is_write=True,
                attempts=deploy_attempts,
            )
            final_state = RunFinalState(
                prod_tree_sha256=staged_hash,
                staging_tree_sha256=staged_hash,
                prod_changed=True,
                tests_passed=True,
                ship_log_status="Deployed to Prod",
                doc_version="v2",
                quiz_attempts=2,
                quiz_passed=True,
                deployed=True,
                release_commit_sha=release_sha,
                pushed=True,
            )
            answer = f"Completed {scenario.id} after v2 remediation under {rule_code} and deployed to prod."
            stop_reason = "completed"
        elif scenario.persona == "reject_or_exhaust_quiz":
            add_step(
                examiner_role,
                "notion-quiz-mcp",
                "upsert_ship_log_doc",
                {
                    "title": f"Fix {scenario.id}",
                    "diff_sha256": diff_hash,
                    "changed_files": [primary_module, "test_fintech.py"],
                    "applied_rules": [rule_code],
                    "targeted_weaknesses": scenario.target_skill_tags or ["refund_policy_compliance"],
                    "doc_version": "v2",
                    "status": "Remediation v2",
                },
                is_write=True,
            )
            add_step(
                examiner_role,
                "notion-quiz-mcp",
                "create_comprehension_quiz",
                {"diff_sha256": diff_hash, "attempt": 2},
                is_write=True,
            )
            final_state = RunFinalState(
                prod_tree_sha256=seed_hash,
                staging_tree_sha256=staged_hash,
                prod_changed=False,
                tests_passed=True,
                ship_log_status="Blocked",
                doc_version="v2",
                quiz_attempts=2,
                quiz_passed=False,
                deployed=False,
                pushed=False,
            )
            answer = (
                f"Staging pytest passed for {scenario.id}, but quiz attempts were exhausted; "
                "Ship Log is Blocked and production remains untouched."
            )
            stop_reason = "quiz_failed"
        else:
            add_step(
                "system",
                "code-mcp",
                "deploy_to_production",
                {"expected_diff_sha256": diff_hash, "release_note": f"Ship {scenario.id}"},
                is_write=True,
                attempts=deploy_attempts,
            )
            final_state = RunFinalState(
                prod_tree_sha256=staged_hash,
                staging_tree_sha256=staged_hash,
                prod_changed=True,
                tests_passed=True,
                ship_log_status="Deployed to Prod",
                doc_version="v1",
                quiz_attempts=1,
                quiz_passed=True,
                deployed=True,
                release_commit_sha=release_sha,
                pushed=True,
            )
            answer = f"Applied {rule_code} fix to {primary_module}, passed staging tests and quiz v1, and deployed."
            stop_reason = "completed"

    effective_model = model if model.startswith("dry-run:") else f"dry-run:{model}"
    steps_count = sum(t.agent in LLM_AGENT_ROLES for t in trace)
    tokens_in = 650 * steps_count
    tokens_out = 180 * steps_count
    cost_usd = estimate_cost_usd(model, tokens_in, tokens_out)
    latency_s = round(sum(t.latency_s for t in trace) + 0.12 * steps_count, 3)

    return EvalRunResult.from_run(
        scenario=scenario,
        repeat=repeat,
        final_state=final_state,
        trace=trace,
        stop_reason=stop_reason,
        final_answer=answer,
        model=effective_model,
        topology=topology,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_s=latency_s,
    )


# ---- live execution in isolated per-run workspaces -----------------------------------------------------------------
def _execute_live_scenario(
    scenario: ScenarioSpec,
    *,
    repeat: int,
    cohort: CohortConfig,
    base_settings: RuntimeSettings,
    mcp_transport: McpTransport,
) -> EvalRunResult:
    """Run one live scenario inside a fresh temporary ``NO_QUIZ_WORKSPACE_DIR``.

    The workflow is a context manager: its 2 ``stdio`` MCP subprocesses, the ``mcp-bridge`` event-loop thread and
    the SQLite checkpointer / skill-memory handles are closed before ``TemporaryDirectory`` removes the workspace,
    even when the run crashes, so a 30-120 run campaign never leaks processes or file handles.
    """
    from no_quiz_no_prod.graph.workflow import NoQuizWorkflow

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"nqnp_{scenario.id}_r{repeat}_") as tmp_dir:
        isolated_settings = base_settings.model_copy(
            update={
                "workspace_dir": Path(tmp_dir),
                "notion_api_key": None,
                "git_remote_url": None,  # releases go to the run's own local bare remote, never to GitHub
                "github_token": None,
            }
        )
        try:
            with NoQuizWorkflow(
                settings=isolated_settings,
                topology=cohort.topology,
                model=cohort.model,
                mcp_transport=mcp_transport,
            ) as workflow:
                return workflow.run_scenario(
                    scenario,
                    repeat=repeat,
                    user_id=f"eval_{cohort.slug}_{scenario.id}_r{repeat}",
                )
        except LLMError as exc:  # the provider failed outside a scored turn (e.g. while the workflow started)
            return _unscored_run(scenario, repeat, cohort, "api_error", exc, time.perf_counter() - started)
        except Exception as exc:
            # Deliberately broad: one crashing run must not lose the paid runs of a whole campaign. The crash is
            # not hidden: it is logged with its traceback, stored as its own `harness_error` row, left out of every
            # rate, listed in the report, and it makes the CLI exit with EXIT_HARNESS_ERROR.
            logger.exception("harness error in scenario %s#%s (%s)", scenario.id, repeat, cohort.label)
            return _unscored_run(scenario, repeat, cohort, "harness_error", exc, time.perf_counter() - started)


def _unscored_run(
    scenario: ScenarioSpec,
    repeat: int,
    cohort: CohortConfig,
    stop_reason: Literal["api_error", "harness_error"],
    exc: Exception,
    elapsed_s: float,
) -> EvalRunResult:
    """The row of a run that produced no result: a provider failure (``api_error``) or a harness bug."""
    fallback_hash = sha256_hex(f"error:{scenario.seed}:{scenario.id}:{repeat}")
    empty_state = RunFinalState(prod_tree_sha256=fallback_hash, staging_tree_sha256=fallback_hash, prod_changed=False)
    return EvalRunResult.from_run(
        scenario=scenario,
        repeat=repeat,
        final_state=empty_state,
        trace=[],
        stop_reason=stop_reason,
        final_answer="",
        model=cohort.model,
        topology=cohort.topology,
        latency_s=round(elapsed_s, 3),
        error=f"{type(exc).__name__}: {exc}"[:300],
    )


def run_cohort(
    scenarios: Sequence[ScenarioSpec],
    *,
    cohort: CohortConfig,
    repeats: int,
    workers: int,
    dry_run: bool,
    base_settings: RuntimeSettings,
    mcp_transport: McpTransport = "stdio",
) -> list[EvalRunResult]:
    """Execute ``scenarios × repeats`` for one cohort, preserving deterministic output ordering."""
    jobs = [(scenario, repeat) for scenario in scenarios for repeat in range(repeats)]
    if not jobs:
        return []

    if dry_run:
        return [
            simulate_dry_run_scenario(
                scn,
                repeat=rep,
                model=cohort.model,
                topology=cohort.topology,
            )
            for scn, rep in jobs
        ]

    # In-process MCP transport shares process-wide os.environ['NO_QUIZ_WORKSPACE_DIR']; serialize it safely.
    effective_workers = 1 if mcp_transport == "inprocess" else workers
    max_workers = max(1, min(effective_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda job: _execute_live_scenario(
                    job[0],
                    repeat=job[1],
                    cohort=cohort,
                    base_settings=base_settings,
                    mcp_transport=mcp_transport,
                ),
                jobs,
            )
        )


# ---- CLI orchestration ---------------------------------------------------------------------------------------------
def _resolve_default_model(settings: RuntimeSettings) -> str:
    if settings.providers["openrouter"]:
        return settings.openrouter_model
    if settings.providers["anthropic"]:
        return settings.anthropic_model
    return settings.openrouter_model


def _build_cohorts(args: argparse.Namespace, settings: RuntimeSettings) -> list[CohortConfig]:
    default_model = args.model or _resolve_default_model(settings)
    base_topology: Topology = args.topology

    if args.compare_models:
        models = list(dict.fromkeys(m.strip() for m in args.compare_models.split(",") if m.strip()))
        if len(models) < 2:
            raise ValueError("--compare-models expects at least two distinct comma-separated model identifiers")
        if args.compare_topologies:
            return [CohortConfig(model=m, topology=t) for m in models for t in ("dual", "single")]
        return [CohortConfig(model=m, topology=base_topology) for m in models]

    if args.compare_topologies:
        return [
            CohortConfig(model=default_model, topology="dual"),
            CohortConfig(model=default_model, topology="single"),
        ]

    return [CohortConfig(model=default_model, topology=base_topology)]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``eval/run_eval.py``."""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Execute the No-Quiz-No-Prod evaluation harness on a scenario set.",
    )
    parser.add_argument(
        "--scenario-set",
        choices=get_args(ScenarioSetName),
        default=DEFAULT_SCENARIO_SET,
        help="Named scenario set: 'dev' (eval/scenarios.jsonl, tuned on: optimistic) or 'heldout' "
        "(eval/scenarios_heldout.jsonl, never tune on it). Default: dev.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="Path to another scenarios JSONL file; overrides --scenario-set (recorded as a 'custom' set).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for results.jsonl, summary.md, comparison.md and manifest.json. Default: a new "
        "eval/runs/<utc timestamp>_<slug>/ directory. A directory that already holds results is refused.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --out to be a directory that already holds results (they are replaced).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent isolated scenario workers (default: 4).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Repeats K per scenario (default: {DEFAULT_REPEATS}). pass^K needs K >= 2 to say anything about "
        "reliability; with K = 1 it equals pass@1.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override LLM model id (e.g. claude-sonnet-4-5).")
    parser.add_argument(
        "--topology",
        choices=("dual", "single"),
        default="dual",
        help="Agent topology: 'dual' (Coder -> Examiner) or 'single' (default: dual).",
    )
    parser.add_argument(
        "--compare-models",
        type=str,
        default=None,
        help="Comma-separated model ids to benchmark head-to-head (e.g. m1,m2).",
    )
    parser.add_argument(
        "--compare-topologies",
        action="store_true",
        help="Benchmark 'dual' vs 'single' topology head-to-head on the same scenarios.",
    )
    parser.add_argument(
        "--mcp-transport",
        choices=("stdio", "inprocess"),
        default="stdio",
        help="MCP transport mode for live runs (default: stdio).",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Run the cross-family binary LLM judge, its calibration and the swap test; verdicts are stored.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional explicit judge model override.",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        help="A label file filled in by a person (see eval/labels/README.md). With --with-judge on a live run, the "
        "judge is calibrated against it; without it, only the synthetic draft labels are available.",
    )
    parser.add_argument(
        "--export-labeling-batch",
        type=Path,
        default=None,
        metavar="RESULTS_JSONL",
        help="Do not run anything: sample runs from this results file and write an unlabelled batch for a person "
        "to fill in (needs --labeling-out).",
    )
    parser.add_argument("--labeling-out", type=Path, default=None, help="Where --export-labeling-batch writes.")
    parser.add_argument(
        "--labeling-count",
        type=int,
        default=DEFAULT_LABELING_BATCH_SIZE,
        help=f"Runs to sample for labelling (default: {DEFAULT_LABELING_BATCH_SIZE}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios.")
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help="Optional comma-separated scenario IDs filter (e.g. n01,n03,ad01).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deterministic offline runs without calling LLM APIs (validates contracts & reports).",
    )
    return parser


def _injection_texts(scenarios: Sequence[ScenarioSpec]) -> dict[str, str]:
    return {scn.id: scn.injection.text for scn in scenarios if scn.injection is not None}


def _export_labeling_batch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """``--export-labeling-batch``: write real runs with empty label fields. Never guesses a label."""
    if args.labeling_out is None:
        parser.error("--export-labeling-batch needs --labeling-out")
    if args.labeling_out.exists() and not args.overwrite:
        parser.error(f"{args.labeling_out} exists: a filled-in label file is never replaced without --overwrite")
    source = Path(args.export_labeling_batch).resolve()
    planted: dict[str, str] = {}
    for path in SCENARIO_SET_PATHS.values():
        if path.is_file():
            planted.update(_injection_texts(load_scenarios(path)))
    batch = export_labeling_batch(
        load_results(source),
        source=display_path(source),
        count=args.labeling_count,
        injection_texts=planted,
    )
    atomic_write_text(args.labeling_out, batch.model_dump_json(indent=2) + "\n")
    print(f"Wrote {len(batch.records)} unlabelled records to {args.labeling_out}")
    return EXIT_OK


def _load_run_scenarios(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[str, Path, list[ScenarioSpec]]:
    """``(scenario set name, file, scenarios)`` after ``--scenario-ids`` then ``--limit``."""
    if args.scenarios is not None:
        set_name, path = CUSTOM_SCENARIO_SET, Path(args.scenarios)
        scenarios = load_scenarios(path)
    else:
        set_name, path = args.scenario_set, SCENARIO_SET_PATHS[args.scenario_set]
        scenarios = load_scenario_set(args.scenario_set)
    if path.resolve() == HELDOUT_SCENARIOS_PATH.resolve():
        overlap = scenario_set_overlap(load_scenario_set("dev"), scenarios)
        if overlap:
            parser.error(f"the held-out set repeats the development set: {'; '.join(overlap)}")
    if args.scenario_ids:
        wanted = {sid.strip() for sid in args.scenario_ids.split(",") if sid.strip()}
        scenarios = [scn for scn in scenarios if scn.id in wanted]
        if not scenarios:
            parser.error(f"none of --scenario-ids {sorted(wanted)} matched {path}")
    return set_name, path, scenarios[: args.limit] if args.limit is not None else scenarios


def _resolve_out_dir(
    args: argparse.Namespace, cohorts: Sequence[CohortConfig], parser: argparse.ArgumentParser
) -> Path:
    """A new timestamped directory under ``eval/runs/`` by default; never a directory that already holds results."""
    if args.out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        mode = "dry-run_" if args.dry_run else ""
        return RUNS_DIR / f"{stamp}_{mode}{cohorts[0].slug}"
    out_dir = Path(args.out)
    holds_results = out_dir.is_dir() and (
        (out_dir / RESULTS_FILENAME).exists() or any(out_dir.glob(COHORT_RESULTS_GLOB))
    )
    if holds_results and not args.overwrite:
        parser.error(f"{out_dir} already holds eval results: choose another --out or pass --overwrite")
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``uv run python eval/run_eval.py``."""
    parser = build_arg_parser()
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(cli_args)

    if args.export_labeling_batch is not None:
        return _export_labeling_batch(args, parser)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    # An eval never pushes to a configured (e.g. GitHub) release remote, dry-run or live: each live run releases to
    # the local bare remote of its own temporary workspace (see _execute_live_scenario).
    settings = RuntimeSettings.from_env().model_copy(update={"git_remote_url": None, "github_token": None})
    scenario_set, scenarios_path, scenarios = _load_run_scenarios(args, parser)

    cohorts = _build_cohorts(args, settings)
    if not args.dry_run:
        for cohort in cohorts:
            needed_provider = resolve_provider(cohort.model)
            if not settings.providers[needed_provider]:
                env_hint = "OPENROUTER_API_KEY" if needed_provider == "openrouter" else "ANTHROPIC_API_KEY"
                print(
                    f"ERROR: live run for model {cohort.model!r} requires {env_hint} in .env "
                    "(or pass --dry-run for offline contract verification).",
                    file=sys.stderr,
                )
                return EXIT_USAGE

    out_dir = _resolve_out_dir(args, cohorts, parser)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_results: dict[str, list[EvalRunResult]] = {}
    cohort_records: list[CohortRecord] = []
    started_all = time.perf_counter()

    for cohort in cohorts:
        t0 = time.perf_counter()
        runs = run_cohort(
            scenarios,
            cohort=cohort,
            repeats=args.repeats,
            workers=args.workers,
            dry_run=args.dry_run,
            base_settings=settings,
            mcp_transport=args.mcp_transport,
        )
        cohort_results[cohort.label] = runs
        summary = summarize_cohort(runs, label=cohort.label)
        elapsed = time.perf_counter() - t0
        mode_tag = "DRY-RUN" if args.dry_run else "LIVE"
        print(
            f"[{mode_tag}] {cohort.label}: pass^{summary.repeats}={summary.pass_hat_k * 100:.1f}% "
            f"(95% CI {summary.wilson_95_low * 100:.1f}-{summary.wilson_95_high * 100:.1f}% on "
            f"{summary.scenarios} scenarios), {summary.passed_runs}/{summary.total_runs} runs passed, "
            f"{summary.harness_error_runs} harness error(s), in {elapsed:.1f}s | {summary.total_tokens:,} tokens | "
            f"${summary.total_cost_usd:.4f}"
        )
        results_file = RESULTS_FILENAME
        if len(cohorts) > 1:
            results_file = cohort_results_filename(cohort.slug)
            write_results(runs, out_dir / results_file)
        cohort_records.append(
            CohortRecord(
                label=cohort.label,
                model=cohort.model,
                topology=cohort.topology,
                results_file=results_file,
                rows=len(runs),
            )
        )

    primary_cohort = cohorts[0]
    primary_runs = cohort_results[primary_cohort.label]
    results_path = out_dir / RESULTS_FILENAME
    write_results(primary_runs, results_path)

    judged_runs: list[JudgedRunRecord] | None = None
    calibration: JudgeCalibrationReport | None = None
    swap_result: SwapTestResult | None = None
    planted = _injection_texts(scenarios)

    if args.with_judge:
        has_any_key = any(settings.providers.values())
        if args.dry_run or not has_any_key:
            calibration = calibrate_judge_on_synthetic_labels(
                mode="recorded_cross_family",
                agent_model=primary_cohort.model,
                settings=settings,
                judge_model=args.judge_model,
            )
        else:
            judged_runs = judge_batch(
                [run for run in primary_runs if is_scored(run)],
                settings=settings,
                judge_model=args.judge_model,
                workers=args.workers,
                injection_texts=planted,
            )
            if args.human_labels is not None:
                calibration = calibrate_judge_on_human_labels(
                    args.human_labels,
                    agent_model=primary_cohort.model,
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                )
            else:
                calibration = calibrate_judge_on_synthetic_labels(
                    mode="live_llm",
                    agent_model=primary_cohort.model,
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                )
            if len(cohorts) >= 2:
                swap_result = swap_test(
                    primary_runs,
                    cohort_results[cohorts[1].label],
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                    injection_texts=planted,
                )
            elif args.repeats >= 2:
                swap_result = swap_test(
                    primary_runs,
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                    injection_texts=planted,
                )
        judge_files = write_judge_artifacts(
            out_dir, judged_runs=judged_runs, calibration=calibration, swap_result=swap_result
        )
        print(f"Stored judge output: {', '.join(path.name for path in judge_files)}")

    summary_path = write_summary_artifacts(
        primary_runs,
        out_dir,
        scenarios=scenarios,
        judged_runs=judged_runs,
        calibration=calibration,
        swap_result=swap_result,
        scenario_set=scenario_set,
    )
    print(f"Wrote {results_path} ({len(primary_runs)} rows) and {summary_path}")

    if len(cohorts) >= 2:
        comp_path = write_comparison_artifacts(
            cohort_results,
            out_dir,
            swap_result=swap_result,
            calibration=calibration,
            scenario_set=scenario_set,
        )
        print(f"Wrote multi-cohort comparison to {comp_path} (total {time.perf_counter() - started_all:.1f}s)")

    judge_record = None
    if calibration is not None:
        judge_record = JudgeRecord(
            judge_model=calibration.judge_model,
            judged_cohort=primary_cohort.label if judged_runs else None,
            verdicts_stored=bool(judged_runs),
            note=f"calibration labels: {calibration.label_provenance} ({calibration.labels_path})",
        )
    write_manifest(
        build_run_manifest(
            name=out_dir.name,
            scenario_set=scenario_set,
            scenarios_path=scenarios_path,
            scenario_count=len(scenarios),
            repeats=args.repeats,
            dry_run=args.dry_run,
            cohorts=cohort_records,
            judge=judge_record,
            cli_args=cli_args,
        ),
        out_dir,
    )

    harness_errors = [run.key for runs in cohort_results.values() for run in runs if not is_scored(run)]
    if harness_errors:
        print(
            f"ERROR: {len(harness_errors)} run(s) hit a harness error and were not scored "
            f"({', '.join(sorted(set(harness_errors)))}). Fix the harness and re-run: do not publish this report.",
            file=sys.stderr,
        )
        return EXIT_HARNESS_ERROR
    return EXIT_OK


__all__ = [
    "DEFAULT_REPEATS",
    "EXIT_HARNESS_ERROR",
    "EXIT_OK",
    "EXIT_USAGE",
    "CohortConfig",
    "McpTransport",
    "build_arg_parser",
    "main",
    "run_cohort",
    "simulate_dry_run_scenario",
]
