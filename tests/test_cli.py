"""CLI contract (`no-quiz-no-prod`): the deploy prompt never turns a stray key into a rejection, and exit codes tell
scripts how a run ended.

The workflow is a scripted double: these tests are about what the CLI asks, records and returns, not about the graph.
"""

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from no_quiz_no_prod import cli
from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import (
    ALL_STOP_REASONS,
    DeployDecision,
    EnvVar,
    PendingRelease,
    QuizAnswer,
    RuntimeSettings,
    StopReason,
    ToolErrorPayload,
)

RunCli = Callable[..., tuple[int, str]]
DIFF = "a" * 64
THREAD = "cli_test"


def outcome(stop_reason: StopReason | None = None, interrupt: Any = None) -> SimpleNamespace:
    """The fields of a ``WorkflowOutcome`` the CLI reads."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        interrupt=interrupt,
        final_answer="",
        final_state=SimpleNamespace(deployed=stop_reason == "completed", ship_log_status="Staging"),
        budget=SimpleNamespace(steps=1, max_steps=14, cost_usd=0.0),
    )


def deploy_gate(release: PendingRelease | None = None) -> SimpleNamespace:
    gate = SimpleNamespace(
        diff_sha256=DIFF, tests_passed=True, sha256_matches=True, quiz_passed=True, human_approved=False
    )
    kind = "release_retry" if release is not None else "deploy_approval"
    return SimpleNamespace(kind=kind, gate=gate, quiz=None, release=release)


def quiz_gate() -> SimpleNamespace:
    quiz = SimpleNamespace(
        attempt=1,
        max_attempts=2,
        skill_tag="decimal_rounding",
        doc_version="v1",
        stem="Why?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
    )
    return SimpleNamespace(kind="quiz", quiz=quiz, gate=None, release=None)


class ScriptedWorkflow:
    """Starts on ``first`` and answers a deploy decision with ``after_decision``; records what the CLI submitted."""

    def __init__(self, first: SimpleNamespace, after_decision: SimpleNamespace | None = None) -> None:
        self.first = first
        self.after_decision = after_decision or outcome("completed")
        self.decisions: list[DeployDecision] = []
        self.quiz_answers: list[QuizAnswer] = []
        self.started = False

    def start_run(self, task: str, **_: Any) -> SimpleNamespace:
        self.started = True
        return self.first

    def submit_quiz(self, thread_id: str, answer: QuizAnswer, **_: Any) -> tuple[SimpleNamespace, SimpleNamespace]:
        self.quiz_answers.append(answer)
        grade = SimpleNamespace(passed=True, explanation="Right.", remediation_required=False)
        return grade, outcome("completed")

    def submit_deploy_decision(self, thread_id: str, decision: DeployDecision, **_: Any) -> SimpleNamespace:
        self.decisions.append(decision)
        return self.after_decision


def typed(*answers: str) -> Callable[[str], str]:
    """An ``input`` double: the given lines, then a closed stdin (``EOFError``)."""
    lines: Iterator[str] = iter(answers)

    def read(_prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    return read


@pytest.fixture
def settings(tmp_path: Path) -> RuntimeSettings:
    """Explicit settings: a temp workspace whose release remote is its own local bare repository."""
    return RuntimeSettings(workspace_dir=tmp_path / "workspace")


@pytest.fixture
def run_cli(settings: RuntimeSettings) -> RunCli:
    def run(workflow: ScriptedWorkflow, *answers: str) -> tuple[int, str]:
        args = cli.build_parser().parse_args(["run", "Fix VAT rounding", "--thread-id", THREAD])
        out = io.StringIO()
        code = cli._cmd_run(args, settings=settings, workflow=workflow, input_fn=typed(*answers), out=out)
        return code, out.getvalue()

    return run


# ---- BE-09: only an explicit answer is a decision ------------------------------------------------------------------
def test_empty_enter_and_typos_at_the_deploy_prompt_ask_again_instead_of_rejecting(run_cli: RunCli) -> None:
    workflow = ScriptedWorkflow(outcome(interrupt=deploy_gate()))
    code, transcript = run_cli(workflow, "", "maybe", "N0", "y")
    assert [decision.action for decision in workflow.decisions] == [
        "approve"
    ]  # the three stray answers recorded nothing
    assert workflow.decisions[0].expected_diff_sha256 == DIFF
    assert transcript.count("No decision recorded") == 3
    assert code == cli.EXIT_OK


def test_closed_stdin_at_the_deploy_prompt_records_no_decision(run_cli: RunCli) -> None:
    workflow = ScriptedWorkflow(outcome(interrupt=deploy_gate()))
    code, transcript = run_cli(workflow, "")  # one stray Enter, then EOF
    assert workflow.decisions == []  # before the fix: a "reject", which blocks the change
    assert code == cli.EXIT_NO_DECISION
    assert THREAD in transcript and "still pending" in transcript


@pytest.mark.parametrize("answer", sorted(cli.REJECT_ANSWERS))
def test_an_explicit_rejection_is_recorded_and_has_its_own_exit_code(answer: str, run_cli: RunCli) -> None:
    workflow = ScriptedWorkflow(outcome(interrupt=deploy_gate()), after_decision=outcome("human_rejected"))
    code, _ = run_cli(workflow, answer.upper())
    assert [decision.action for decision in workflow.decisions] == ["reject"]
    assert code == cli.EXIT_CODE_BY_STOP_REASON["human_rejected"] != cli.EXIT_OK


def test_closed_stdin_at_the_quiz_prompt_submits_no_answer(run_cli: RunCli) -> None:
    workflow = ScriptedWorkflow(outcome(interrupt=quiz_gate()))
    code, _ = run_cli(workflow, "x")  # an invalid choice, then EOF
    assert workflow.quiz_answers == [] and code == cli.EXIT_NO_DECISION


# ---- C8: a failed push comes back as a new gate, never as a silent end ----------------------------------------------
def test_a_failed_release_is_shown_and_must_be_approved_again(run_cli: RunCli) -> None:
    error = ToolErrorPayload(code="timeout", message="git push did not answer", outcome_unknown=True)
    failed = PendingRelease(expected_diff_sha256=DIFF, attempts=1, error=error)
    workflow = ScriptedWorkflow(outcome(interrupt=deploy_gate(failed)))
    code, transcript = run_cli(workflow, "", "y")  # a stray Enter does not re-approve either
    assert "Release attempt 1 was not published (timeout: git push did not answer)" in transcript
    assert "may have reached the remote" in transcript and "approval was revoked" in transcript
    assert [(d.action, d.expected_diff_sha256) for d in workflow.decisions] == [("approve", DIFF)]
    assert code == cli.EXIT_OK

    unanswered = ScriptedWorkflow(outcome(interrupt=deploy_gate(failed)))
    assert run_cli(unanswered)[0] == cli.EXIT_NO_DECISION and unanswered.decisions == []


# ---- C9: the release configuration is checked before the run starts ------------------------------------------------
def test_a_bad_release_configuration_stops_before_any_model_call(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = settings.model_copy(update={"git_branch": "not a branch"})
    workflow = ScriptedWorkflow(outcome("completed"))
    args = cli.build_parser().parse_args(["run", "Fix VAT rounding"])
    with pytest.raises(WorkspaceError) as refused:
        cli._cmd_run(args, settings=bad, workflow=workflow, input_fn=typed(), out=io.StringIO())
    assert refused.value.code == "invalid_argument" and not workflow.started

    monkeypatch.setattr(cli.RuntimeSettings, "from_env", classmethod(lambda _cls: bad))
    assert cli.main(["run", "Fix VAT rounding"]) == cli.EXIT_ERROR  # the user sees one line, not a traceback
    assert capsys.readouterr().err.startswith("Error (invalid_argument): ")


def test_a_valid_release_configuration_is_announced(run_cli: RunCli, settings: RuntimeSettings) -> None:
    _, transcript = run_cli(ScriptedWorkflow(outcome("completed")))
    assert f"Release target: {settings.default_release_remote_dir} (local) · branch main · git " in transcript


# ---- BE-10: exit codes and one-line errors -------------------------------------------------------------------------
def test_every_stop_reason_has_an_exit_code_and_only_completed_is_zero() -> None:
    assert set(cli.EXIT_CODE_BY_STOP_REASON) == set(ALL_STOP_REASONS)
    reserved = {cli.EXIT_USAGE, cli.EXIT_NO_DECISION, cli.EXIT_INTERRUPTED}
    assert not reserved & set(cli.EXIT_CODE_BY_STOP_REASON.values())
    zero = [reason for reason, code in cli.EXIT_CODE_BY_STOP_REASON.items() if code == cli.EXIT_OK]
    assert zero == ["completed"]
    codes = list(cli.EXIT_CODE_BY_STOP_REASON.values())
    assert len(set(codes)) == len(codes)  # a script can tell every ending apart


@pytest.mark.parametrize("stop_reason", ALL_STOP_REASONS)
def test_run_returns_the_exit_code_of_its_stop_reason(stop_reason: StopReason, run_cli: RunCli) -> None:
    code, transcript = run_cli(ScriptedWorkflow(outcome(stop_reason)))  # stdin closed: no clarification reply
    assert code == cli.EXIT_CODE_BY_STOP_REASON[stop_reason]
    assert f"stop_reason={stop_reason}" in transcript


def test_run_help_documents_every_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--help"])
    text = capsys.readouterr().out
    for reason, code in cli.EXIT_CODE_BY_STOP_REASON.items():
        assert f"{code}  stop_reason={reason}" in text
    assert f"{cli.EXIT_NO_DECISION}  stdin closed" in text


@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_max_steps_below_one_is_a_usage_error(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(["run", "task", "--max-steps", value])
    assert stopped.value.code == cli.EXIT_USAGE
    assert "expected an integer >= 1" in capsys.readouterr().err


def test_an_unknown_seed_is_one_line_on_stderr_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(EnvVar.NO_QUIZ_WORKSPACE_DIR.value, str(tmp_path / "workspace"))
    assert cli.main(["reset", "--seed", "no-such-seed"]) == cli.EXIT_ERROR  # before the fix: WorkspaceError escaped
    captured = capsys.readouterr()
    assert captured.err.startswith("Error (invalid_argument): ") and captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


def test_typed_failures_and_ctrl_c_map_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(_args: object) -> int:
        raise WorkspaceError("internal_error", "git is missing")

    monkeypatch.setattr(cli, "_dispatch", refuse)
    assert cli.main(["reset"]) == cli.EXIT_ERROR
    assert capsys.readouterr().err == "Error (internal_error): git is missing\n"

    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_dispatch", interrupt)
    assert cli.main(["reset"]) == cli.EXIT_INTERRUPTED
