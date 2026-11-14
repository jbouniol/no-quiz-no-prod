"""The Pedagogical Examiner: documents the staged change and writes the comprehension quiz, in a fresh context.

The harness hands it ``(task, diff_report, applied_rules_text, memory_profile_block, attempt, prior_failed_grade)``
and nothing else: no Coder transcript (context isolation, the point of the dual topology). Attempt 1 publishes doc
v1 + quiz 1. Attempt 2 (remediation) publishes doc v2 with the ``### Why Option X Was a Trap (Remediation v2)``
callout and a quiz asked from a fresh angle.

The binders below are harness-side: they keep the model's content, drop anything else, and inject the authoritative
binding (diff hash, thread, attempt, doc version, status, changed files, Layer 4 weaknesses). The model can neither
bind a quiz to another diff or thread, nor forget the remediation callout or a detected weakness.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from no_quiz_no_prod.agents.prompts import EXAMINER_SYSTEM_PROMPT
from no_quiz_no_prod.agents.tools import RULE_CODE_PATTERN, THREAD_ARG, TOOLS, AgentSpec
from no_quiz_no_prod.contracts import (
    ALL_FINTECH_MODULES,
    ALL_SKILL_TAGS,
    DiffReport,
    PytestReport,
    QuizGradeResult,
    SkillTag,
    doc_version_for_attempt,
    ship_log_status_for_attempt,
)

EXAMINER_TOOLS: Final = ("search_notion_rules", "read_notion_rule", "upsert_ship_log_doc", "create_comprehension_quiz")
EXAMINER_MAX_TOKENS: Final = 8000  # the whole Markdown doc travels inside one tool call
DIFF_CHARS_FOR_EXAMINER: Final = 12_000  # the brief stays small even for a multi-file change
RULE_CHARS_FOR_EXAMINER: Final = 3_000
REMEDIATION_HEADING: Final = "### Why Option {option} Was a Trap (Remediation v2)"
_REMEDIATION_HEADING_RE: Final = re.compile(r"^###\s+Why Option [A-D] Was a Trap \(Remediation v2\)", re.M)
_RULE_CODE_RE: Final = re.compile(RULE_CODE_PATTERN)

EXAMINER: Final = AgentSpec(
    role="examiner", system_prompt=EXAMINER_SYSTEM_PROMPT, tools=EXAMINER_TOOLS, max_tokens=EXAMINER_MAX_TOKENS
)


def _untrusted(source: str, body: str) -> str:
    return f'<untrusted_data source="{source}">\n{body.strip() or "(empty)"}\n</untrusted_data>'


def _unique[T](items: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(items))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


# ---- the brief (the Examiner's whole context) ----------------------------------------------------------------------
def render_diff(diff: DiffReport, *, limit: int = DIFF_CHARS_FOR_EXAMINER) -> str:
    """Changed files with +/- counts and the unified diffs, truncated to ``limit`` characters."""
    header = ", ".join(f"{file.path} (+{file.additions} -{file.deletions})" for file in diff.files) or "(none)"
    body = "\n".join(file.unified_diff for file in diff.files)
    if len(body) > limit:
        body = f"{body[:limit]}\n... [diff truncated at {limit} characters]"
    return f"Changed files: {header}\n```diff\n{body}\n```"


def render_applied_rules(rules: Mapping[str, Mapping[str, str]], *, limit: int = RULE_CHARS_FOR_EXAMINER) -> str:
    """The rule pages the agents read, one section per rule code (each body truncated to ``limit``)."""
    if not rules:
        return "(the Coder read no rule page: use search_notion_rules / read_notion_rule if you need one)"
    sections = []
    for code in sorted(rules):
        title, body = rules[code].get("title", ""), rules[code].get("body", "")
        sections.append(f"### {code} {title}".rstrip() + f"\n{body[:limit]}")
    return "\n\n".join(sections)


def remediation_brief(grade: QuizGradeResult) -> str:
    """What went wrong at the previous attempt, and the exact v2 deliverables (shared by both topologies)."""
    heading = REMEDIATION_HEADING.format(option=grade.selected_option)
    return (
        f"Quiz attempt {grade.attempt} of {grade.max_attempts} was answered WRONG: the developer picked option "
        f"{grade.selected_option} (skill `{grade.skill_tag}`).\n"
        f"The grader's explanation of that option, i.e. the misconception: {grade.explanation}\n"
        f"Publish doc v2 with the exact heading `{heading}`, then create a NEW question for attempt "
        f"{grade.attempt + 1}: a fresh angle on the same or a sibling edge case (different stem, values and letter)."
    )


def build_examiner_brief(
    task: str,
    diff_report: DiffReport,
    applied_rules_text: str,
    memory_profile_block: str,
    attempt: int,
    prior_failed_grade: QuizGradeResult | None = None,
    *,
    pytest: PytestReport | None = None,
) -> str:
    """The isolated Examiner context: task, diff, test result, rules, Layer 4 profile and the assignment."""
    tests = f"{pytest.summary} (exit code {pytest.exit_code})" if pytest else "(no pytest report)"
    if prior_failed_grade is None:
        assignment = (
            f"Attempt {attempt}: publish doc {doc_version_for_attempt(attempt)} with `upsert_ship_log_doc`, then "
            f"create the attempt-{attempt} question with `create_comprehension_quiz`."
        )
    else:
        assignment = remediation_brief(prior_failed_grade)
    return "\n\n".join(
        [
            "# Developer task (verbatim)",
            _untrusted("developer_task", task),
            f"# Staged change under review\ndiff_sha256: {diff_report.diff_sha256} (bound by the harness)",
            _untrusted("staged_diff", render_diff(diff_report)),
            f"# Staging tests\n{tests}",
            "# Engineering rules applied",
            _untrusted("notion_rules", applied_rules_text),
            "# Developer profile (Layer 4 long-term memory)",
            memory_profile_block.strip(),
            f"# Your assignment\n{assignment}",
        ]
    )


def build_examiner_messages(brief: str) -> list[dict[str, Any]]:
    """Fresh Examiner conversation for one attempt: system prompt + the brief, nothing else."""
    return [{"role": "system", "content": EXAMINER.system_prompt}, {"role": "user", "content": brief}]


# ---- harness-side binders and checks -------------------------------------------------------------------------------
def remediation_callout(grade: QuizGradeResult) -> str:
    """Deterministic v2 callout, appended when the model forgets the mandatory heading."""
    heading = REMEDIATION_HEADING.format(option=grade.selected_option)
    return f"{heading}\n\n> **Option {grade.selected_option} was a trap.** {grade.explanation}\n"


def has_remediation_callout(doc_markdown: str) -> bool:
    return bool(_REMEDIATION_HEADING_RE.search(doc_markdown))


def bind_ship_log_args(
    args: Mapping[str, Any],
    *,
    thread_id: str,
    diff: DiffReport,
    attempt: int,
    weaknesses: Iterable[SkillTag] = (),
    rules_read: Iterable[str] = (),
    prior_failed_grade: QuizGradeResult | None = None,
) -> dict[str, Any]:
    """``upsert_ship_log_doc`` arguments: the model's content + the harness binding (see the module docstring)."""
    bound = {key: args[key] for key in TOOLS["upsert_ship_log_doc"].parameters if key in args}
    doc = str(bound.get("doc_markdown") or "")
    if prior_failed_grade is not None and doc.strip() and not has_remediation_callout(doc):
        doc = f"{doc.rstrip()}\n\n{remediation_callout(prior_failed_grade)}"
    changed = [path for path in diff.changed_files if path in ALL_FINTECH_MODULES]
    model_changed = [path for path in _as_list(bound.get("changed_files")) if path in ALL_FINTECH_MODULES]
    rules = [
        code for code in _as_list(bound.get("applied_rules")) if isinstance(code, str) and _RULE_CODE_RE.match(code)
    ]
    targeted = [tag for tag in [*_as_list(bound.get("targeted_weaknesses")), *weaknesses] if tag in ALL_SKILL_TAGS]
    return {
        **bound,
        "doc_markdown": doc,
        "changed_files": _unique(changed or model_changed),
        "applied_rules": _unique(rules or sorted(rules_read)),
        "targeted_weaknesses": _unique(targeted),
        "diff_sha256": diff.diff_sha256,
        "doc_version": doc_version_for_attempt(attempt),
        "status": ship_log_status_for_attempt(attempt),
        "quiz_attempts": attempt - 1,
        THREAD_ARG: thread_id,
    }


def bind_quiz_args(args: Mapping[str, Any], *, thread_id: str, diff_sha256: str, attempt: int) -> dict[str, Any]:
    """``create_comprehension_quiz`` flat arguments (#14 §3B) + the harness binding (diff, thread, attempt)."""
    bound = {key: args[key] for key in TOOLS["create_comprehension_quiz"].parameters if args.get(key) is not None}
    if isinstance(bound.get("correct_option"), str):
        bound["correct_option"] = bound["correct_option"].strip().upper()
    return {**bound, "diff_sha256": diff_sha256, THREAD_ARG: thread_id, "attempt": attempt}


def missing_deliverables(
    *, ship_log: Mapping[str, Any] | None, quiz: Mapping[str, Any] | None, diff_sha256: str, attempt: int
) -> list[str]:
    """What the documentation phase still owes for the current diff and attempt (empty: ready for the quiz gate)."""
    version = doc_version_for_attempt(attempt)
    missing = []
    if not ship_log or ship_log.get("diff_sha256") != diff_sha256 or ship_log.get("doc_version") != version:
        missing.append(f"call upsert_ship_log_doc for the current diff (doc {version})")
    if not quiz or quiz.get("diff_sha256") != diff_sha256 or quiz.get("attempt") != attempt:
        missing.append(f"call create_comprehension_quiz for the current diff (attempt {attempt})")
    return missing


__all__ = [
    "DIFF_CHARS_FOR_EXAMINER",
    "EXAMINER",
    "EXAMINER_MAX_TOKENS",
    "EXAMINER_TOOLS",
    "REMEDIATION_HEADING",
    "RULE_CHARS_FOR_EXAMINER",
    "bind_quiz_args",
    "bind_ship_log_args",
    "build_examiner_brief",
    "build_examiner_messages",
    "has_remediation_callout",
    "missing_deliverables",
    "remediation_brief",
    "remediation_callout",
    "render_applied_rules",
    "render_diff",
]
