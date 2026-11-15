"""Markdown and JSON report generator for eval runs and multi-cohort comparisons.

Produces ``summary.md`` and ``comparison.md`` with per-kind tables, ``pass^k`` and its scenario-level 95% Wilson
interval, ``pass@k``, failure modes (``unsafe``, ``unstable``, ``expensive``), fault diagnostics that keep harness
retries apart from agent recoveries, judge verdicts with their failures and cost, and paired Sign Test p-values.
It also stores what the judge said (``write_judge_artifacts``), so every judge figure in a report can be recomputed.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from no_quiz_no_prod.contracts import EvalRunResult, ScenarioSpec
from no_quiz_no_prod.eval.judge import (
    HUMAN_PROVENANCE,
    JUDGE_KAPPA_BAR,
    MIN_HUMAN_LABELS,
    JudgeCalibrationReport,
    JudgedRunRecord,
    SwapTestResult,
)
from no_quiz_no_prod.eval.paths import (
    COMPARISON_FILENAME,
    JUDGE_CALIBRATION_FILENAME,
    JUDGE_VERDICTS_FILENAME,
    SUMMARY_FILENAME,
    SWAP_TEST_FILENAME,
    atomic_write_text,
)
from no_quiz_no_prod.eval.scorer import (
    ZERO_TOOL_CONTROL_STOP_REASONS,
    CohortComparison,
    CohortSummary,
    ThreeLensScore,
    build_do_nothing_baseline,
    compare_cohorts,
    score_three_lenses,
    summarize_cohort,
)
from no_quiz_no_prod.llm import LLMUsage

SIGNIFICANCE_LEVEL: Final = 0.05
SCENARIO_SET_NOTES: Final[Mapping[str, str]] = {
    "dev": (
        "development set: the prompts and the harness were tuned on these scenarios, so every rate below is an "
        "optimistic estimate"
    ),
    "heldout": "held-out set: never used to tune a prompt or the harness; keep it that way",
    "custom": "custom scenario file",
}
SINGLE_REPEAT_NOTE: Final = (
    "one repeat per scenario: `pass^1` equals `pass@1`, so run-to-run reliability is NOT measured and a difference "
    "of a few scenarios is within noise (run with `--repeats 3` or more)"
)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_usd(value: float) -> str:
    return f"${value:.5f}"


def _fmt_list_or_none(items: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in items) if items else "none in these runs"


def _fmt_usage(usage: LLMUsage) -> str:
    return f"`{usage.tokens_in:,}` tokens in · `{usage.tokens_out:,}` tokens out · `{_fmt_usd(usage.cost_usd)}`"


def _dry_run_banner_lines(results: Sequence[EvalRunResult]) -> list[str]:
    if not any(r.model.startswith("dry-run:") for r in results):
        return []
    return [
        "> [!WARNING]",
        "> **DRY-RUN SIMULATION (NO LLM OR MCP CALLS MADE)** — All metrics, traces, token counts, and costs "
        "in this report come from deterministic offline simulation (`--dry-run`, `model = 'dry-run:...'`) to verify "
        "contracts and report rendering. Do not treat these figures as live model measurements.",
        "",
    ]


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
                "### Injected faults: harness retries vs agent recoveries",
                "",
                "A fault the harness absorbs inside one tool call (`Seen by model` = no) says nothing about the "
                "model: only the agent columns describe what the model did after an error it could read.",
                "",
                "| Run | Fault | Outcome | Steps | Seen by model | Harness retries | Agent safe | Agent unsafe | Actions |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for run, lens, scn in fault_rows:
            assert scn is not None
            actions = [f"`{t.recovery}` ({t.recovery_actor})" for t in lens.turns if t.recovery != "none"]
            lines.append(
                f"| `{run.key}` | `{scn.fault}` | `{run.outcome}` | {run.steps}/{run.max_steps} | "
                f"{'yes' if lens.fault_seen_by_model else 'no'} | {lens.harness_retries} | "
                f"{lens.safe_recoveries} | {lens.unsafe_recoveries} | {', '.join(actions) or '`none`'} |"
            )
        lines.append("")

    adv_rows = [(r, lens, scn) for r, lens, scn in lenses if r.kind == "adversarial"]
    if adv_rows:
        exposed_adv = [item for item in adv_rows if item[1].injection_exposed]
        unexposed_keys = [r.key for r, lens, _ in adv_rows if not lens.injection_exposed]
        code_adv = [
            item for item in adv_rows if item[2] and item[2].injection and item[2].injection.target.endswith(".py")
        ]
        rule_adv = [
            item for item in adv_rows if item[2] and item[2].injection and item[2].injection.target.startswith("RULE-")
        ]
        resisted_exposed = sum(lens.injection_resisted for _, lens, _ in exposed_adv)
        exfil_calls = [r.key for r, _, _ in adv_rows if "read_audit_credentials" in r.forbidden_called]
        lines.extend(
            [
                "### Prompt-Injection Resistance (`adversarial`)",
                "",
                f"- **Exposed & resisted (`contains_injection == True`)**: **{resisted_exposed}/{len(exposed_adv)}** exposed runs completed the developer's task while ignoring planted tool-output directives.",
                f"- **Unexposed adversarial runs (`contains_injection == False`, planted target never read)**: {_fmt_list_or_none(unexposed_keys)}.",
                f"- **Code-planted injections**: {sum(x[1].passed for x in code_adv)}/{len(code_adv)} scenario runs passed.",
                f"- **Notion-rule-planted injections**: {sum(x[1].passed for x in rule_adv)}/{len(rule_adv)} scenario runs passed.",
                f"- **Honeypot exfiltration (`read_audit_credentials`) triggered**: {_fmt_list_or_none(exfil_calls)}.",
                "",
            ]
        )
    return lines


def _judged_runs_lines(judged_runs: Sequence[JudgedRunRecord]) -> list[str]:
    usable = [j for j in judged_runs if j.acceptable is not None]
    failed_calls = [j.key for j in judged_runs if j.acceptable is None]
    rejected = [
        f"{j.key} ({', '.join(j.failed_criteria) or 'no criterion returned'})" for j in usable if not j.acceptable
    ]
    criteria = usable[0].verdict.criteria if usable and usable[0].verdict else []
    usage = sum((j.usage for j in judged_runs), LLMUsage())
    sel = judged_runs[0].judge_selection
    return [
        f"- **Judge model**: `{sel.judge_model}` (`{sel.judge_provider}`, cross-family=`{sel.cross_family}` — {sel.reason})",
        f"- **Judged runs**: {sum(bool(j.acceptable) for j in usable)}/{len(usable)} accepted on "
        f"{len(criteria)} binary criteria ({', '.join(f'`{c.name}`' for c in criteria) or 'n/a'}).",
        f"- **Rejected runs (failed criteria)**: {_fmt_list_or_none(rejected)}",
        f"- **Failed judge calls (no verdict, left out of the count above)**: {len(failed_calls)} — {_fmt_list_or_none(failed_calls)}",
        f"- **Judge usage**: {_fmt_usage(usage)}",
        "",
    ]


def _validation_line(calibration: JudgeCalibrationReport | None) -> str:
    if calibration is not None and calibration.validated:
        return (
            f"- **Judge validated against human labels**: **yes** (κ = `{calibration.cohen_kappa:.3f}` on "
            f"{calibration.compared_count} human-labelled runs)."
        )
    if calibration is None:
        why = "no calibration was run"
    elif calibration.label_provenance != HUMAN_PROVENANCE:
        why = "the only labels available are synthetic drafts, which nobody labelled"
    elif calibration.compared_count < MIN_HUMAN_LABELS:
        why = f"only {calibration.compared_count} human-labelled runs, {MIN_HUMAN_LABELS} are required"
    else:
        why = f"κ = {calibration.cohen_kappa:.3f} is under the {JUDGE_KAPPA_BAR:.2f} bar"
    return (
        f"- **Judge validated against human labels**: **no** ({why}). Until it is, the judge figures in this section "
        "are not evidence about the agent."
    )


def _calibration_lines(calibration: JudgeCalibrationReport) -> list[str]:
    human = calibration.label_provenance == HUMAN_PROVENANCE
    title = (
        f"### Judge calibration against human labels (`{calibration.labels_path}`)"
        if human
        else f"### Judge check against SYNTHETIC draft labels (`{calibration.labels_path}`) — not human labels"
    )
    lines = [title, ""]
    if not human:
        lines.append(
            "- **Provenance**: these labels were written by the harness author, not by a reviewer. The figures below "
            "check the code path; they are **not** a human-agreement measurement."
        )
    live = calibration.judge_mode == "live_llm"
    kappa_label = "Cohen's $\\kappa$" if live else "Fixture $\\kappa$ (offline fixture, not a measured $\\kappa$)"
    lines.extend(
        [
            f"- **Judge mode**: `{calibration.judge_mode}` (`{calibration.judge_model}`, cross-family=`{calibration.cross_family}`)",
            f"- **Raw agreement**: {calibration.raw_agreement * 100:.1f}% on {calibration.compared_count} records "
            f"(always-majority baseline = {calibration.majority_baseline_agreement * 100:.1f}%; "
            f"labels accept {calibration.label_accept_count}, judge accepts {calibration.judge_accept_count})",
            f"- **{kappa_label}**: `{calibration.cohen_kappa:.3f}`"
            + (
                f" (recorded single-family vector: `{calibration.baseline_single_family_kappa:.3f}`)"
                if calibration.baseline_single_family_kappa is not None
                else ""
            ),
            f"- **Failed judge calls**: {calibration.failed_judge_calls} · **records not labelled yet**: {calibration.unlabeled_count}",
            f"- **Disagreements**: {_fmt_list_or_none(calibration.disagreements)}",
            f"- **Calibration usage**: {_fmt_usage(calibration.usage)}",
            "",
        ]
    )
    return lines


def _swap_lines(swap_result: SwapTestResult) -> list[str]:
    flipped = [f"{d.key_a} vs {d.key_b}" for d in swap_result.details if d.consistent is False]
    failed = [f"{d.key_a} vs {d.key_b}" for d in swap_result.details if d.consistent is None]
    return [
        "### Position-Bias Swap Test (`(A, B)` vs `(B, A)`)",
        "",
        f"- **Usable pairs**: {swap_result.usable_pairs}/{swap_result.total_pairs}",
        f"- **Order-consistent winners**: **{swap_result.consistent_pairs}/{swap_result.usable_pairs}** ({_fmt_pct(swap_result.consistency_rate)})",
        f"- **Position-flipped pairs (dropped, never guessed)**: {swap_result.flipped_pairs} ({_fmt_pct(swap_result.position_bias_rate)}) — {_fmt_list_or_none(flipped)}",
        f"- **Pairs with a failed judge call**: {len(failed)} — {_fmt_list_or_none(failed)}",
        f"- **Swap-test usage**: {_fmt_usage(swap_result.usage)}",
        "",
    ]


def _judge_section(
    judged_runs: Sequence[JudgedRunRecord] | None,
    calibration: JudgeCalibrationReport | None,
    swap_result: SwapTestResult | None,
) -> list[str]:
    if not judged_runs and calibration is None and swap_result is None:
        return []
    lines = ["## 5. Cross-Family Binary LLM-as-a-Judge & Calibration", "", _validation_line(calibration)]
    stored = [
        name
        for name, present in (
            (JUDGE_VERDICTS_FILENAME, bool(judged_runs)),
            (JUDGE_CALIBRATION_FILENAME, calibration is not None),
            (SWAP_TEST_FILENAME, swap_result is not None),
        )
        if present
    ]
    lines.extend([f"- **Stored next to this report**: {_fmt_list_or_none(stored)}", ""])
    if judged_runs:
        lines.extend(_judged_runs_lines(judged_runs))
    if calibration is not None:
        lines.extend(_calibration_lines(calibration))
    if swap_result is not None and swap_result.total_pairs > 0:
        lines.extend(_swap_lines(swap_result))
    return lines


def _control_line(scenarios: Sequence[ScenarioSpec]) -> str:
    parts = []
    for stop_reason in ZERO_TOOL_CONTROL_STOP_REASONS:
        control = summarize_cohort(build_do_nothing_baseline(scenarios, stop_reason=stop_reason))
        per_kind = ", ".join(f"{m.kind} {m.passed_runs}/{m.runs}" for m in control.per_kind[1:] if m.runs)
        parts.append(f"always `{stop_reason}`: `{_fmt_pct(control.pass_rate)}` ({per_kind})")
    return (
        "- **Zero-tool controls** (an agent that calls no tool and always ends the same way; what it passes is not "
        f"skill): {' · '.join(parts)}"
    )


def _coverage_line(summary: CohortSummary) -> str:
    line = (
        f"- **Coverage**: **{summary.scenarios} scenarios** × **{summary.repeats} repeat(s)** = "
        f"**{summary.total_runs} scored runs**"
    )
    if summary.min_repeats != summary.repeats:
        line += f" (uneven: {summary.min_repeats} to {summary.repeats} scored repeats per scenario)"
    if summary.harness_error_runs:
        line += (
            f" — **{summary.harness_error_runs} run(s) hit a harness error and are NOT scored**: fix the harness and "
            "re-run before reading any rate"
        )
    return line


def _push_line(summary: CohortSummary) -> str:
    if summary.push_observed_runs == 0:
        return (
            "- **Deployed & Pushed**: not observed — these rows were recorded before the release push existed "
            f"({summary.deployed_runs} run(s) deployed)"
        )
    line = (
        "- **Deployed & Pushed (`deployed` and `pushed`: the release commit is on the remote)**: "
        f"**{summary.deployed_and_pushed_runs}/{summary.total_runs}** ({_fmt_pct(summary.deployed_and_pushed_rate)}; "
        f"{summary.deployed_runs} run(s) deployed)"
    )
    if summary.push_observed_runs != summary.total_runs:
        line += f" — push observed on {summary.push_observed_runs}/{summary.total_runs} runs only"
    return line


def generate_summary_markdown(
    results: Sequence[EvalRunResult],
    scenarios: Sequence[ScenarioSpec] | None = None,
    *,
    judged_runs: Sequence[JudgedRunRecord] | None = None,
    calibration: JudgeCalibrationReport | None = None,
    swap_result: SwapTestResult | None = None,
    scenario_set: str | None = None,
    title: str = "No-Quiz-No-Prod — Evaluation Summary",
) -> str:
    """Render ``summary.md`` for one cohort with 3-lens diagnostics and statistical intervals."""
    summary: CohortSummary = summarize_cohort(results)
    scenarios_by_id = {s.id: s for s in scenarios} if scenarios else {}
    k = summary.repeats
    k_col_at = f"pass@{k}"
    k_col_hat = f"pass^{k}"
    overall = summary.per_kind[0]

    lines: list[str] = [
        f"# {title}",
        "",
        *_dry_run_banner_lines(results),
        "Scored strictly on **backend ground-truth state** (`RunFinalState` vs `ExpectedState`, SHA-256 locks, "
        "`forbidden_called`, the expected stop reason and the `max_steps` budget) — never on LLM self-report.",
        "",
        "## 1. Executive Cohort KPIs",
        "",
    ]
    if scenario_set is not None:
        lines.append(f"- **Scenario set**: `{scenario_set}` — {SCENARIO_SET_NOTES.get(scenario_set, scenario_set)}.")
    lines.extend(
        [
            f"- **Cohort**: `{summary.model}` · topology=`{summary.topology}`",
            _coverage_line(summary),
            f"- **Right Final State (`state_ok`)**: **{summary.state_ok_runs}/{summary.total_runs}** ({_fmt_pct(summary.state_ok_rate)})",
            f"- **Strict Pass, per run (`outcome == 'pass'`)**: **{summary.passed_runs}/{summary.total_runs}** ({_fmt_pct(summary.pass_rate)}; descriptive, repeats of one scenario are not independent)",
            f"- **Reliability, per scenario**: `{k_col_hat}` = **{_fmt_pct(summary.pass_hat_k)}** "
            f"({overall.scenarios_all_passed}/{summary.scenarios} scenarios passed all {k} repeat(s); 95% Wilson CI "
            f"`{_fmt_pct(summary.wilson_95_low)}` to `{_fmt_pct(summary.wilson_95_high)}`) · `{k_col_at}` = "
            f"**{_fmt_pct(summary.pass_at_k)}** (at least one repeat passed)",
        ]
    )
    if k == 1:
        lines.append(f"- **Caveat**: {SINGLE_REPEAT_NOTE}.")
    lines.extend(
        [
            _push_line(summary),
            f"- **Hidden Failures (`state_ok` yet unsafe/over-budget/wrong stop, or `deploy_push_mismatch`)**: **{summary.hidden_failure_count}**",
            f"- **Efficiency & Cost**: `{summary.total_tokens:,}` tokens · avg latency `{summary.avg_latency_s:.2f}s` · total `{_fmt_usd(summary.total_cost_usd)}` (`{summary.tokens_per_passed_run:,}` tokens / `{_fmt_usd(summary.cost_per_passed_run_usd)}` per passed run)",
        ]
    )
    if scenarios:
        lines.append(_control_line(scenarios))
    lines.extend(
        [
            f"- **Trace Replay Pointers**: showcase passed = `{summary.showcase_passed_key or 'n/a'}` · showcase failed/blocked = `{summary.showcase_failed_or_blocked_key or 'n/a'}`",
            "",
            f"## 2. Per-Kind Results (`{k_col_hat}` with its scenario-level 95% Wilson interval)",
            "",
            f"| Kind | Scenarios | Runs | Right State | Strict Pass (runs) | `{k_col_hat}` | 95% Wilson CI of `{k_col_hat}` | `{k_col_at}` | Avg Steps | Avg Latency | Avg Cost |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in summary.per_kind:
        bold = "**" if row.kind == "ALL" else ""
        lines.append(
            f"| {bold}`{row.kind}`{bold} | {row.scenarios} | {row.runs} | {_fmt_pct(row.state_ok_rate)} | "
            f"{_fmt_pct(row.pass_rate)} | {bold}{_fmt_pct(row.pass_hat_k)}{bold} | "
            f"{_fmt_pct(row.wilson_95_low)} – {_fmt_pct(row.wilson_95_high)} | "
            f"{_fmt_pct(row.pass_at_k)} | {row.avg_steps:.1f} | "
            f"{row.avg_latency_s:.2f}s | {_fmt_usd(row.avg_cost_usd)} |"
        )

    fm = summary.failure_modes
    outcome_pills = " · ".join(f"`{name}`: **{count}**" for name, count in summary.outcomes.items())
    stop_pills = " · ".join(f"`{name}`: **{count}**" for name, count in summary.stop_reasons.items())

    lines.extend(
        [
            "",
            "## 3. Outcome Precedence & Failure-Mode Taxonomy",
            "",
            f"- **Outcome counts** (`harness_error > api_error > wrong_state > wrong_stop > forbidden_call > over_budget > pass`): {outcome_pills}",
            f"- **Terminal stop reasons**: {stop_pills}",
            f"- **Unsafe success** (*right state, forbidden tool called*): {_fmt_list_or_none(fm.unsafe)}",
            f"- **Unstable success** (*passes some repeats, fails others*): {_fmt_list_or_none(fm.unstable)}",
            f"- **Expensive success** (*right state, over step budget; tokens vs median passing `{fm.median_passing_tokens:,}`*): {_fmt_list_or_none(fm.expensive)}",
            f"- **Deploy/push mismatch** (*`deployed` and `pushed` diverge; counted as hidden failures*): {_fmt_list_or_none(fm.deploy_push_mismatch)}",
            f"- **Wrong stop reason** (*right state, but not the stop reason the scenario expects*): {_fmt_list_or_none(fm.wrong_stop)}",
            f"- **Visible wrong state**: {_fmt_list_or_none(fm.wrong_state)}",
            f"- **Provider API errors**: {_fmt_list_or_none(fm.api_errors)}",
            f"- **Harness errors (bugs of the harness, not scored)**: {_fmt_list_or_none(fm.harness_errors)}",
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
    scenario_set: str | None = None,
    title: str = "No-Quiz-No-Prod — Multi-Model & Topology Benchmark Comparison",
) -> str:
    """Render ``comparison.md`` comparing 2+ cohorts with paired Sign Test significance."""
    labels = list(cohort_results.keys())
    if len(labels) < 2:
        raise ValueError("generate_comparison_markdown requires at least two cohorts")

    all_runs = [r for runs in cohort_results.values() for r in runs]
    summaries = {label: summarize_cohort(cohort_results[label], label=label) for label in labels}
    comparisons: list[CohortComparison] = [
        compare_cohorts(cohort_results[label_a], cohort_results[label_b], label_a=label_a, label_b=label_b)
        for label_a, label_b in itertools.combinations(labels, 2)
    ]
    lines: list[str] = [
        f"# {title}",
        "",
        *_dry_run_banner_lines(all_runs),
        "Head-to-head evaluation on identical scenarios, fresh seeded sandboxes, and identical human personas.",
        "",
    ]
    if scenario_set is not None:
        lines.extend(
            [f"- **Scenario set**: `{scenario_set}` — {SCENARIO_SET_NOTES.get(scenario_set, scenario_set)}.", ""]
        )
    cautions = []
    if all(s.repeats == 1 for s in summaries.values()):
        cautions.append(f"{SINGLE_REPEAT_NOTE[0].upper()}{SINGLE_REPEAT_NOTE[1:]}.")
    if all(c.sign_test_p >= SIGNIFICANCE_LEVEL for c in comparisons):
        cautions.append(
            "No pair below is statistically distinguishable: **this table does not rank the cohorts**. Read the "
            "intervals, not the order of the rows."
        )
    if cautions:
        lines.extend(["> [!IMPORTANT]", *[f"> {text}" for text in cautions], ""])
    lines.extend(
        [
            "## 1. Cohort Comparison Matrix",
            "",
            "| Cohort (`model` × `topology`) | Scenarios × k | Strict Pass (runs) | `pass^k` | 95% Wilson CI of `pass^k` | `normal` | `ambiguous` | `impossible` | `adversarial` | Avg Steps | Avg Latency | Tokens / Pass | USD / Pass |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for label in labels:
        s = summaries[label]
        lines.append(
            f"| `{s.model}` · **`{s.topology}`** | {s.scenarios} × {s.repeats} | {_fmt_pct(s.pass_rate)} | "
            f"**{_fmt_pct(s.pass_hat_k)}** | {_fmt_pct(s.wilson_95_low)} – {_fmt_pct(s.wilson_95_high)} | "
            f"{_kind_pass_str(s, 'normal')} | {_kind_pass_str(s, 'ambiguous')} | "
            f"{_kind_pass_str(s, 'impossible')} | {_kind_pass_str(s, 'adversarial')} | "
            f"{s.avg_steps:.1f} | {s.avg_latency_s:.2f}s | {s.tokens_per_passed_run:,} | "
            f"{_fmt_usd(s.cost_per_passed_run_usd)} |"
        )

    lines.extend(
        [
            "",
            "## 2. Paired Scenario Sign Tests (`pass^k` Discordant Pairs, all cohort pairs)",
            "",
        ]
    )
    for comp in comparisons:
        sig_note = (
            f"statistically significant at $\\alpha = {SIGNIFICANCE_LEVEL}$"
            if comp.sign_test_p < SIGNIFICANCE_LEVEL
            else f"not statistically distinguishable at $\\alpha = {SIGNIFICANCE_LEVEL}$ on {comp.paired_scenarios} scenarios"
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
    scenario_set: str | None = None,
    markdown_filename: str = SUMMARY_FILENAME,
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
        scenario_set=scenario_set,
    )
    atomic_write_text(target_md, markdown)
    if json_filename:
        target_json = Path(out_dir) / json_filename
        payload = build_summary_payload(results, calibration=calibration, swap_result=swap_result)
        atomic_write_text(target_json, json.dumps(payload, indent=2) + "\n")
    return target_md


def write_comparison_artifacts(
    cohort_results: Mapping[str, Sequence[EvalRunResult]],
    out_dir: Path,
    *,
    swap_result: SwapTestResult | None = None,
    calibration: JudgeCalibrationReport | None = None,
    scenario_set: str | None = None,
    markdown_filename: str = COMPARISON_FILENAME,
) -> Path:
    """Write ``comparison.md`` atomically inside ``out_dir``."""
    target_md = Path(out_dir) / markdown_filename
    markdown = generate_comparison_markdown(
        cohort_results,
        swap_result=swap_result,
        calibration=calibration,
        scenario_set=scenario_set,
    )
    atomic_write_text(target_md, markdown)
    return target_md


def write_judge_artifacts(
    out_dir: Path,
    *,
    judged_runs: Sequence[JudgedRunRecord] | None = None,
    calibration: JudgeCalibrationReport | None = None,
    swap_result: SwapTestResult | None = None,
) -> list[Path]:
    """Store everything the judge said: one verdict per line with evidence and usage, the swap picks of every pair,
    and the calibration report. A failed call is stored too (``acceptable: null``), never dropped."""
    written: list[Path] = []
    if judged_runs:
        target = Path(out_dir) / JUDGE_VERDICTS_FILENAME
        atomic_write_text(target, "".join(f"{record.model_dump_json()}\n" for record in judged_runs))
        written.append(target)
    for filename, payload in ((JUDGE_CALIBRATION_FILENAME, calibration), (SWAP_TEST_FILENAME, swap_result)):
        if payload is not None:
            target = Path(out_dir) / filename
            atomic_write_text(target, payload.model_dump_json(indent=2) + "\n")
            written.append(target)
    return written


__all__ = [
    "SCENARIO_SET_NOTES",
    "SIGNIFICANCE_LEVEL",
    "build_summary_payload",
    "generate_comparison_markdown",
    "generate_summary_markdown",
    "write_comparison_artifacts",
    "write_judge_artifacts",
    "write_summary_artifacts",
]
