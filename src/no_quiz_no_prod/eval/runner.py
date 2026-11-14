"""Isolated-workspace evaluation runner and CLI orchestration for the Golden Set.

Backs ``eval/run_eval.py`` so both ``uv run python eval/run_eval.py ...`` and Python callers import
the same single source of truth from ``no_quiz_no_prod.eval.runner``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from no_quiz_no_prod.contracts import (
    ALL_FINTECH_MODULES,
    PROJECT_ROOT,
    AgentRole,
    EvalRunResult,
    FintechModule,
    RuleCode,
    RunFinalState,
    RuntimeSettings,
    ScenarioSpec,
    ToolCallTrace,
    ToolErrorPayload,
    Topology,
    idem_key,
    sha256_hex,
)
from no_quiz_no_prod.eval.judge import (
    JudgeCalibrationReport,
    JudgedRunRecord,
    SwapTestResult,
    calibrate_judge_against_human,
    judge_batch,
    swap_test,
)
from no_quiz_no_prod.eval.report import (
    write_comparison_artifacts,
    write_summary_artifacts,
)
from no_quiz_no_prod.eval.scorer import (
    DEFAULT_SCENARIOS_PATH,
    load_scenarios,
    summarize_cohort,
    write_results,
)
from no_quiz_no_prod.llm import estimate_cost_usd, resolve_provider

McpTransport = Literal["stdio", "inprocess"]
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


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
        return cleaned[:48] or self.topology


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
        )
        answer = (
            f"I must refuse this request: modifying {primary_module} as asked violates {rule_code} "
            "or closed sandbox security boundaries. Staging and production remain untouched."
        )
        stop_reason = "refused"
    else:
        write_args: dict[str, object] = {"path": primary_module, "content": f"# patched {scenario.id}\n"}
        if scenario.fault == "write_timeout":
            add_step(
                coder_role,
                "code-mcp",
                "write_staging_file",
                write_args,
                is_write=True,
                error=ToolErrorPayload(
                    code="timeout",
                    message="staging write timed out before confirmation",
                    retryable=True,
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

        if scenario.persona == "fail_once_then_pass":
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
                examiner_role,
                "code-mcp",
                "deploy_to_production",
                {"expected_diff_sha256": diff_hash, "release_note": f"Ship {scenario.id}"},
                is_write=True,
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
            )
            answer = (
                f"Staging pytest passed for {scenario.id}, but quiz attempts were exhausted; "
                "Ship Log is Blocked and production remains untouched."
            )
            stop_reason = "quiz_failed"
        else:
            add_step(
                examiner_role,
                "code-mcp",
                "deploy_to_production",
                {"expected_diff_sha256": diff_hash, "release_note": f"Ship {scenario.id}"},
                is_write=True,
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
            )
            answer = f"Applied {rule_code} fix to {primary_module}, passed staging tests and quiz v1, and deployed."
            stop_reason = "completed"

    steps_count = len(trace)
    tokens_in = 650 * steps_count
    tokens_out = 180 * steps_count
    cost_usd = estimate_cost_usd(model, tokens_in, tokens_out)
    latency_s = round(sum(t.latency_s for t in trace) + 0.12 * steps_count, 3)

    return EvalRunResult.from_run(
        scenario=scenario,
        repeat=repeat,
        final_state=final_state,
        trace=trace,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        final_answer=answer,
        model=model,
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
    """Run one live scenario inside a fresh temporary ``NO_QUIZ_WORKSPACE_DIR``."""
    from no_quiz_no_prod.graph.workflow import NoQuizWorkflow

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"nqnp_{scenario.id}_r{repeat}_") as tmp_dir:
        isolated_settings = base_settings.model_copy(
            update={
                "workspace_dir": Path(tmp_dir),
                "notion_api_key": None,
            }
        )
        try:
            workflow = NoQuizWorkflow(
                settings=isolated_settings,
                topology=cohort.topology,
                model=cohort.model,
                mcp_transport=mcp_transport,
            )
            return workflow.run_scenario(
                scenario,
                repeat=repeat,
                user_id=f"eval_{cohort.slug}_{scenario.id}_r{repeat}",
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            fallback_hash = sha256_hex(f"error:{scenario.seed}:{scenario.id}:{repeat}")
            empty_state = RunFinalState(
                prod_tree_sha256=fallback_hash,
                staging_tree_sha256=fallback_hash,
                prod_changed=False,
            )
            return EvalRunResult.from_run(
                scenario=scenario,
                repeat=repeat,
                final_state=empty_state,
                trace=[],
                stop_reason="api_error",
                final_answer="",
                model=cohort.model,
                topology=cohort.topology,
                latency_s=elapsed,
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
        models = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        if len(models) < 2:
            raise ValueError("--compare-models expects at least two comma-separated model identifiers")
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
        description="Execute the No-Quiz-No-Prod 30-scenario Golden Set evaluation harness.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the Golden Set scenarios JSONL file (default: eval/scenarios.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval",
        help="Output directory for results.jsonl, summary.md, and comparison.md (default: eval/).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent isolated scenario workers (default: 4).")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Repeats K per scenario for pass@k / pass^k (default: 1)."
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
        help="Run cross-family binary LLM-as-a-Judge, Cohen's kappa calibration, and swap_test.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional explicit judge model override.",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``uv run python eval/run_eval.py``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    settings = RuntimeSettings.from_env()
    scenarios = load_scenarios(args.scenarios, limit=args.limit)
    if args.scenario_ids:
        wanted = {sid.strip() for sid in args.scenario_ids.split(",") if sid.strip()}
        scenarios = [scn for scn in scenarios if scn.id in wanted]
        if not scenarios:
            parser.error(f"none of --scenario-ids {sorted(wanted)} matched {args.scenarios}")

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
                return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_results: dict[str, list[EvalRunResult]] = {}
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
            f"[{mode_tag}] {cohort.label}: {summary.passed_runs}/{summary.total_runs} passed "
            f"({summary.pass_rate * 100:.1f}%, pass^{args.repeats}={summary.pass_hat_k * 100:.1f}%) "
            f"in {elapsed:.1f}s | {summary.total_tokens:,} tokens | ${summary.total_cost_usd:.4f}"
        )
        if len(cohorts) > 1:
            write_results(runs, out_dir / f"results_{cohort.slug}.jsonl")

    primary_cohort = cohorts[0]
    primary_runs = cohort_results[primary_cohort.label]
    results_path = out_dir / "results.jsonl"
    write_results(primary_runs, results_path)

    judged_runs: list[JudgedRunRecord] | None = None
    calibration: JudgeCalibrationReport | None = None
    swap_result: SwapTestResult | None = None

    if args.with_judge:
        has_any_key = any(settings.providers.values())
        if args.dry_run or not has_any_key:
            calibration = calibrate_judge_against_human(
                mode="recorded_cross_family",
                agent_model=primary_cohort.model,
                settings=settings,
                judge_model=args.judge_model,
            )
        else:
            judged_runs = judge_batch(
                primary_runs,
                settings=settings,
                judge_model=args.judge_model,
                workers=args.workers,
            )
            calibration = calibrate_judge_against_human(
                mode="live_llm",
                agent_model=primary_cohort.model,
                settings=settings,
                judge_model=args.judge_model,
                workers=args.workers,
            )
            if len(cohorts) >= 2:
                second_runs = cohort_results[cohorts[1].label]
                swap_result = swap_test(
                    primary_runs,
                    second_runs,
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                )
            elif args.repeats >= 2:
                swap_result = swap_test(
                    primary_runs,
                    settings=settings,
                    judge_model=args.judge_model,
                    workers=args.workers,
                )

    summary_path = write_summary_artifacts(
        primary_runs,
        out_dir,
        scenarios=scenarios,
        judged_runs=judged_runs,
        calibration=calibration,
        swap_result=swap_result,
    )
    print(f"Wrote {results_path} ({len(primary_runs)} rows) and {summary_path}")

    if len(cohorts) >= 2:
        comp_path = write_comparison_artifacts(
            cohort_results,
            out_dir,
            swap_result=swap_result,
            calibration=calibration,
        )
        print(f"Wrote multi-cohort comparison to {comp_path} (total {time.perf_counter() - started_all:.1f}s)")

    return 0


__all__ = [
    "CohortConfig",
    "McpTransport",
    "build_arg_parser",
    "main",
    "run_cohort",
    "simulate_dry_run_scenario",
]
