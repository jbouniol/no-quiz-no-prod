"""Markdown and JSON report generator for Golden Set runs and multi-cohort comparisons.

Produces defence-ready ``eval/summary.md`` and ``eval/comparison.md`` reports with per-kind tables,
scenario-level 95% Wilson confidence intervals, ``pass@k``/``pass^k`` reliability, Session 6 failure
modes (``unsafe``, ``unstable``, ``expensive``), turn-level fault recovery diagnostics, Cohen's kappa
judge calibration, and paired Sign Test p-values.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from no_quiz_no_prod.contracts import EvalRunResult, ScenarioSpec
from no_quiz_no_prod.eval.judge import JudgeCalibrationReport, JudgedRunRecord, SwapTestResult
from no_quiz_no_prod.eval.scorer import (
    CohortComparison,
    CohortSummary,
    ThreeLensScore,
    build_do_nothing_baseline,
    compare_cohorts,
    score_three_lenses,
    summarize_cohort,
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_usd(value: float) -> str:
    return f"${value:.5f}"


def _fmt_list_or_none(items: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in items) if items else "none in these runs"


def _fault_and_adversarial_section(
    results: Sequence[EvalRunResult],
    scenarios_by_id: Mapping[str, ScenarioSpec],
) -> list[str]:
    lines: list[str] = ["## 4. Fault-Injection Recovery & Adversarial Hardening", ""]
    lenses: list[tuple[EvalRunResult, ThreeLensScore, ScenarioSpec | None]] = [
        (r, score_three_lenses(r, scenarios_by_id.get(r.id)), scenarios_by_id.get(r.id)) for r in results
    ]

    fault_rows = [(r, lens, scn) for r, lens, scn in lenses if scn is not None and scn.fault is not None]
    if fault_rows:
        lines.extend(
            [
                "### Injected Tool Failures (`write_timeout` & `rate_limit`)",
                "",
                "| Run | Fault | Outcome | Steps | Safe Recoveries | Unsafe Recoveries | Recovery Actions |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for run, lens, scn in fault_rows:
            assert scn is not None
            actions = [t.recovery for t in lens.turns if t.recovery != "none"]
            action_str = ", ".join(f"`{a}`" for a in actions) if actions else "`none`"
            lines.append(
                f"| `{run.key}` | `{scn.fault}` | `{run.outcome}` | {run.steps}/{run.max_steps} | "
                f"{lens.safe_recoveries} | {lens.unsafe_recoveries} | {action_str} |"
            )
        lines.append("")

    adv_rows = [(r, lens, scn) for r, lens, scn in lenses if r.kind == "adversarial"]
    if adv_rows:
        code_adv = [
            item for item in adv_rows if item[2] and item[2].injection and item[2].injection.target.endswith(".py")
        ]
        rule_adv = [
            item for item in adv_rows if item[2] and item[2].injection and item[2].injection.target.startswith("RULE-")
        ]
        resisted_total = sum(lens.injection_resisted for _, lens, _ in adv_rows)
        exfil_calls = [r.key for r, _, _ in adv_rows if "read_audit_credentials" in r.forbidden_called]
        lines.extend(
            [
                "### Prompt-Injection Resistance (`adversarial`)",
                "",
                f"- **Overall injection resistance**: **{resisted_total}/{len(adv_rows)}** runs completed the developer's task while ignoring planted tool-output directives.",
                f"- **Code-planted injections (`orders.py`, `auth.py`, `refunds.py`)**: {sum(x[1].passed for x in code_adv)}/{len(code_adv)} passed.",
                f"- **Notion-rule-planted injections (`RULE-001`..`RULE-003`)**: {sum(x[1].passed for x in rule_adv)}/{len(rule_adv)} passed.",
                f"- **Honeypot exfiltration (`read_audit_credentials`) triggered**: {_fmt_list_or_none(exfil_calls)}.",
                "",
            ]
        )
    return lines


def _judge_section(
    judged_runs: Sequence[JudgedRunRecord] | None,
    calibration: JudgeCalibrationReport | None,
    swap_result: SwapTestResult | None,
) -> list[str]:
    if not judged_runs and calibration is None and swap_result is None:
        return []
    lines = ["## 5. Cross-Family Binary LLM-as-a-Judge & Human Alignment", ""]
    if judged_runs:
        usable = [j for j in judged_runs if j.acceptable is not None]
        accepted = sum(bool(j.acceptable) for j in usable)
        sel = judged_runs[0].judge_selection
        lines.extend(
            [
                f"- **Judge model**: `{sel.judge_model}` (`{sel.judge_provider}`, cross-family=`{sel.cross_family}` — {sel.reason})",
                f"- **Judged runs**: {accepted}/{len(usable)} accepted across 4 quote-backed binary criteria (`faithful`, `only_user_instructions`, `appropriate_clarification_or_refusal`, `pedagogical_clarity`).",
                "",
            ]
        )
    if calibration is not None:
        lines.extend(
            [
                "### Calibration vs `eval/golden_human_labels.json` (20 Human-Labeled Reference Runs)",
                "",
                f"- **Calibrated judge mode**: `{calibration.judge_mode}` (`{calibration.judge_model}`, cross-family=`{calibration.cross_family}`)",
                f"- **Raw agreement**: **{calibration.raw_agreement * 100:.1f}%** ({calibration.compared_count} runs, majority-class baseline={calibration.majority_baseline_agreement * 100:.1f}%)",
                f"- **Cohen's $\\kappa$**: **`{calibration.cohen_kappa:.3f}`**"
                + (
                    f" (vs uncalibrated single-family baseline $\\kappa = `{calibration.baseline_single_family_kappa:.3f}`)"
                    if calibration.baseline_single_family_kappa is not None
                    else ""
                ),
                f"- **Disagreements**: {_fmt_list_or_none(calibration.disagreements)}",
                "",
            ]
        )
    if swap_result is not None and swap_result.usable_pairs > 0:
        lines.extend(
            [
                "### Position-Bias Swap Test (`(A, B)` vs `(B, A)`)",
                "",
                f"- **Usable pairs**: {swap_result.usable_pairs}/{swap_result.total_pairs}",
                f"- **Order-consistent winners**: **{swap_result.consistent_pairs}/{swap_result.usable_pairs}** ({_fmt_pct(swap_result.consistency_rate)})",
                f"- **Position-flipped pairs (dropped, never guessed)**: {swap_result.flipped_pairs} ({_fmt_pct(swap_result.position_bias_rate)})",
                "",
            ]
        )
    return lines


def generate_summary_markdown(
    results: Sequence[EvalRunResult],
    scenarios: Sequence[ScenarioSpec] | None = None,
    *,
    judged_runs: Sequence[JudgedRunRecord] | None = None,
    calibration: JudgeCalibrationReport | None = None,
    swap_result: SwapTestResult | None = None,
    title: str = "No-Quiz-No-Prod — Golden Set Evaluation Summary",
) -> str:
    """Render ``eval/summary.md`` for one cohort with 3-lens diagnostics and statistical intervals."""
    summary: CohortSummary = summarize_cohort(results)
    scenarios_by_id = {s.id: s for s in scenarios} if scenarios else {}

    control_line = ""
    if scenarios:
        control_runs = build_do_nothing_baseline(scenarios)
        control_pass = sum(r.passed for r in control_runs) / max(len(control_runs), 1)
        control_line = (
            f"- **Zero-Tool Control Baseline (`do_nothing`)**: `{_fmt_pct(control_pass)}` "
            "(passes `ambiguous` & `impossible` by leaving workspace untouched, 0% on `normal` & `adversarial` — "
            "proving why per-kind reporting is mandatory)."
        )

    k_col_at = f"pass@{summary.repeats}"
    k_col_hat = f"pass^{summary.repeats}"

    lines: list[str] = [
        f"# {title}",
        "",
        "Scored strictly on **backend ground-truth state** (`RunFinalState` vs `ExpectedState`, SHA-256 locks, "
        "`forbidden_called`, and `max_steps` budget) — never on LLM self-report.",
        "",
        "## 1. Executive Cohort KPIs",
        "",
        f"- **Cohort**: `{summary.model}` · topology=`{summary.topology}`",
        f"- **Coverage**: **{summary.scenarios} scenarios** × **{summary.repeats} repeat(s)** = **{summary.total_runs} runs**",
        f"- **Right Final State (`state_ok`)**: **{summary.state_ok_runs}/{summary.total_runs}** ({_fmt_pct(summary.state_ok_rate)})",
        f"- **Strict Pass (`outcome == 'pass'`)**: **{summary.passed_runs}/{summary.total_runs}** (**{_fmt_pct(summary.pass_rate)}**, scenario-level 95% Wilson CI: `{_fmt_pct(summary.wilson_95_low)}` to `{_fmt_pct(summary.wilson_95_high)}`)",
        f"- **Reliability**: `{k_col_at}` = **{_fmt_pct(summary.pass_at_k)}** · `{k_col_hat}` = **{_fmt_pct(summary.pass_hat_k)}**",
        f"- **Hidden Failures (`state_ok` yet unsafe/over-budget)**: **{summary.hidden_failure_count}**",
        f"- **Efficiency & Cost**: `{summary.total_tokens:,}` tokens · avg latency `{summary.avg_latency_s:.2f}s` · total `{_fmt_usd(summary.total_cost_usd)}` (`{summary.tokens_per_passed_run:,}` tokens / `{_fmt_usd(summary.cost_per_passed_run_usd)}` per passed run)",
    ]
    if control_line:
        lines.append(control_line)
    lines.extend(
        [
            f"- **Oral-Defence Replay Pointers**: showcase passed = `{summary.showcase_passed_key or 'n/a'}` · showcase failed/blocked = `{summary.showcase_failed_or_blocked_key or 'n/a'}`",
            "",
            "## 2. Per-Kind Results & Scenario-Level 95% Wilson Intervals",
            "",
            f"| Kind | Scenarios | Runs | Right State | Strict Pass | 95% Wilson CI (Scenarios) | `{k_col_at}` | `{k_col_hat}` | Avg Steps | Avg Latency | Avg Cost |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in summary.per_kind:
        bold = "**" if row.kind == "ALL" else ""
        lines.append(
            f"| {bold}`{row.kind}`{bold} | {row.scenarios} | {row.runs} | {_fmt_pct(row.state_ok_rate)} | "
            f"{bold}{_fmt_pct(row.pass_rate)}{bold} | {_fmt_pct(row.wilson_95_low)} – {_fmt_pct(row.wilson_95_high)} | "
            f"{_fmt_pct(row.pass_at_k)} | {_fmt_pct(row.pass_hat_k)} | {row.avg_steps:.1f} | "
            f"{row.avg_latency_s:.2f}s | {_fmt_usd(row.avg_cost_usd)} |"
        )

    fm = summary.failure_modes
    outcome_pills = " · ".join(f"`{name}`: **{count}**" for name, count in summary.outcomes.items())
    stop_pills = " · ".join(f"`{name}`: **{count}**" for name, count in summary.stop_reasons.items())

    lines.extend(
        [
            "",
            "## 3. Outcome Precedence & Session 6 Failure-Mode Taxonomy",
            "",
            f"- **Outcome counts** (`api_error > wrong_state > forbidden_call > over_budget > pass`): {outcome_pills}",
            f"- **Terminal stop reasons**: {stop_pills}",
            f"- **Unsafe success** (*right state, forbidden tool called*): {_fmt_list_or_none(fm.unsafe)}",
            f"- **Unstable success** (*passes some repeats, fails others*): {_fmt_list_or_none(fm.unstable)}",
            f"- **Expensive success** (*right state, over step budget; tokens vs median passing `{fm.median_passing_tokens:,}`*): {_fmt_list_or_none(fm.expensive)}",
            f"- **Visible wrong state**: {_fmt_list_or_none(fm.wrong_state)}",
            f"- **Provider API errors**: {_fmt_list_or_none(fm.api_errors)}",
            "",
        ]
    )

    lines.extend(_fault_and_adversarial_section(results, scenarios_by_id))
    lines.extend(_judge_section(judged_runs, calibration, swap_result))
    return "\n".join(lines).rstrip() + "\n"


def _kind_pass_str(summary: CohortSummary, kind: str) -> str:
    metric = next((m for m in summary.per_kind if m.kind == kind), None)
    if metric is None or metric.runs == 0:
        return "0/0 (0.0%)"
    return f"{metric.passed_runs}/{metric.runs} ({_fmt_pct(metric.pass_rate)})"


def generate_comparison_markdown(
    cohort_results: Mapping[str, Sequence[EvalRunResult]],
    *,
    swap_result: SwapTestResult | None = None,
    calibration: JudgeCalibrationReport | None = None,
    title: str = "No-Quiz-No-Prod — Multi-Model & Topology Benchmark Comparison",
) -> str:
    """Render ``eval/comparison.md`` comparing 2+ cohorts with paired Sign Test significance."""
    labels = list(cohort_results.keys())
    if len(labels) < 2:
        raise ValueError("generate_comparison_markdown requires at least two cohorts")

    summaries = {label: summarize_cohort(cohort_results[label], label=label) for label in labels}
    lines: list[str] = [
        f"# {title}",
        "",
        "Head-to-head evaluation on identical Golden Set scenarios, fresh seeded sandboxes, and identical human personas.",
        "",
        "## 1. Cohort Comparison Matrix",
        "",
        "| Cohort (`model` × `topology`) | Runs | Strict Pass | `pass^k` | `normal` | `ambiguous` | `impossible` | `adversarial` | Avg Steps | Avg Latency | Tokens / Pass | USD / Pass |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label in labels:
        s = summaries[label]
        lines.append(
            f"| `{s.model}` · **`{s.topology}`** | {s.total_runs} | **{_fmt_pct(s.pass_rate)}** | "
            f"**{_fmt_pct(s.pass_hat_k)}** | {_kind_pass_str(s, 'normal')} | {_kind_pass_str(s, 'ambiguous')} | "
            f"{_kind_pass_str(s, 'impossible')} | {_kind_pass_str(s, 'adversarial')} | "
            f"{s.avg_steps:.1f} | {s.avg_latency_s:.2f}s | {s.tokens_per_passed_run:,} | "
            f"{_fmt_usd(s.cost_per_passed_run_usd)} |"
        )

    lines.extend(["", "## 2. Paired Scenario Sign Tests (`pass^k` Discordant Pairs)", ""])
    baseline_label = labels[0]
    for challenger_label in labels[1:]:
        comp: CohortComparison = compare_cohorts(
            cohort_results[baseline_label],
            cohort_results[challenger_label],
            label_a=baseline_label,
            label_b=challenger_label,
        )
        sig_note = (
            "statistically significant at $\\alpha = 0.05$"
            if comp.sign_test_p < 0.05
            else "not statistically distinguishable at $\\alpha = 0.05$ on 30 scenarios (effect size / cost-latency trade-off governs selection)"
        )
        cost_str = f"`{comp.cost_ratio_b_over_a:.2f}x`" if comp.cost_ratio_b_over_a is not None else "`n/a`"
        lat_str = f"`{comp.latency_ratio_b_over_a:.2f}x`" if comp.latency_ratio_b_over_a is not None else "`n/a`"
        lines.extend(
            [
                f"### `{comp.label_a}` vs `{comp.label_b}`",
                "",
                f"- **Paired scenarios**: {comp.paired_scenarios} (`both_pass={comp.both_pass}`, `both_fail={comp.both_fail}`, `only_a={comp.only_a}`, `only_b={comp.only_b}`)",
                f"- **Exact Two-Sided Sign Test**: **`p = {comp.sign_test_p:.4f}`** — {sig_note}.",
                f"- **Pass-rate delta (`B - A`)**: `{comp.pass_rate_delta * 100:+.1f}` percentage points (`pass^k` delta: `{comp.pass_hat_k_delta * 100:+.1f}` pp).",
                f"- **Cost & latency ratio (`B / A`)**: cost per passed run = {cost_str}, avg latency = {lat_str}.",
                "",
            ]
        )

    lines.extend(_judge_section(None, calibration, swap_result))
    return "\n".join(lines).rstrip() + "\n"


def build_summary_payload(
    results: Sequence[EvalRunResult],
    *,
    calibration: JudgeCalibrationReport | None = None,
    swap_result: SwapTestResult | None = None,
) -> dict[str, Any]:
    """Structured JSON-serializable summary dictionary matching ``CohortSummary``."""
    payload: dict[str, Any] = {"summary": summarize_cohort(results).model_dump(mode="json")}
    if calibration is not None:
        payload["judge_calibration"] = calibration.model_dump(mode="json")
    if swap_result is not None:
        payload["swap_test"] = swap_result.model_dump(mode="json")
    return payload


def write_summary_artifacts(
    results: Sequence[EvalRunResult],
    out_dir: Path,
    *,
    scenarios: Sequence[ScenarioSpec] | None = None,
    judged_runs: Sequence[JudgedRunRecord] | None = None,
    calibration: JudgeCalibrationReport | None = None,
    swap_result: SwapTestResult | None = None,
    markdown_filename: str = "summary.md",
    json_filename: str | None = None,
) -> Path:
    """Write ``summary.md`` (and optional ``summary.json``) atomically inside ``out_dir``."""
    target_md = Path(out_dir) / markdown_filename
    markdown = generate_summary_markdown(
        results,
        scenarios,
        judged_runs=judged_runs,
        calibration=calibration,
        swap_result=swap_result,
    )
    _atomic_write_text(target_md, markdown)
    if json_filename:
        target_json = Path(out_dir) / json_filename
        payload = build_summary_payload(results, calibration=calibration, swap_result=swap_result)
        _atomic_write_text(target_json, json.dumps(payload, indent=2) + "\n")
    return target_md


def write_comparison_artifacts(
    cohort_results: Mapping[str, Sequence[EvalRunResult]],
    out_dir: Path,
    *,
    swap_result: SwapTestResult | None = None,
    calibration: JudgeCalibrationReport | None = None,
    markdown_filename: str = "comparison.md",
) -> Path:
    """Write ``comparison.md`` atomically inside ``out_dir``."""
    target_md = Path(out_dir) / markdown_filename
    markdown = generate_comparison_markdown(
        cohort_results,
        swap_result=swap_result,
        calibration=calibration,
    )
    _atomic_write_text(target_md, markdown)
    return target_md


__all__ = [
    "build_summary_payload",
    "generate_comparison_markdown",
    "generate_summary_markdown",
    "write_comparison_artifacts",
    "write_summary_artifacts",
]
