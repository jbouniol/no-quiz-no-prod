"""End-to-end integration: the real Studio API on the real workflow, MCP servers and sandbox (no backend mocks).

``TestClient`` -> ``create_app`` -> ``StudioManager`` -> ``NoQuizWorkflow`` (LangGraph + SQLite checkpointer) ->
``StdioMCPBridge`` -> code-mcp and notion-quiz-mcp as real stdio subprocesses -> ``SandboxWorkspace`` (seeded fintech
code, real staging pytest, SHA-256 gate ledger, private quizzes, Layer 4 skill profile). Only the LLM is scripted
(``testkit.ScriptedLLM``), so the whole pedagogical loop replays deterministically and offline:

    doc v1 + quiz 1 -> wrong answer (B) -> Layer 4 "Weakness Detected" -> fresh Examiner: doc v2 + quiz 2
    -> right answer (C) -> human approval -> deploy_to_production (system role, over stdio) -> prod == staging
    -> the release commit (content == prod/, gate trailers) is on the workspace's local bare release remote

while every premature, mismatched or replayed approval gets ``409 gate_locked``, leaves prod untouched and pushes
nothing.
"""

import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from fastapi.testclient import TestClient
from testkit import (
    FIXED_FINTECH_ORDERS,
    HERMETIC_RELEASE_ENV,
    SCRIPTED_MODEL,
    ScriptedLLM,
    call,
    parse_sse_events,
    turn,
)

from no_quiz_no_prod.agents.examiner import REMEDIATION_HEADING
from no_quiz_no_prod.agents.tools import SEALED
from no_quiz_no_prod.backend import SandboxWorkspace
from no_quiz_no_prod.contracts import ALL_DEPLOY_LOCKS, RuntimeSettings, Topology, compute_tree_sha256
from no_quiz_no_prod.graph.mcp_bridge import MCP_SERVER_LOG_FILENAME, StdioMCPBridge
from no_quiz_no_prod.graph.workflow import NoQuizWorkflow
from no_quiz_no_prod.llm import ChatTurnResponse
from no_quiz_no_prod.ui.server import create_app

USER: Final = "dev_e2e"
THREAD: Final = "thr_e2e_remediation"
SKILL: Final = "decimal_rounding"
TASK: Final = "Fix calculate_order_total in orders.py so totals use Decimal('0.01') with ROUND_HALF_UP per RULE-001."
WRONG_SHA: Final = "f" * 64
SEEDED_RULES: Final = ["RULE-001", "RULE-002", "RULE-003", "RULE-004"]
TRAP_B: Final = "Trap: a binary float stores 2.675 just below the tie, so round() returns 2.67 instead of 2.68."
QUIZ_1: Final = {
    "skill_tag": SKILL,
    "rule_code": "RULE-001",
    "stem": "What does the fixed calculate_order_total return for one item priced 2.675 at vat_rate 0?",
    "option_a": "Decimal('2.68')",
    "option_b": "Decimal('2.67')",
    "option_c": "Decimal('2.675')",
    "option_d": "Decimal('2.7')",
    "correct_option": "A",
    "explanation_a": "Right: ROUND_HALF_UP on the exact Decimal tie 2.675 rounds the cent up to 2.68.",
    "explanation_b": TRAP_B,
    "explanation_c": "Trap: RULE-001 quantizes every amount to Decimal('0.01').",
    "explanation_d": "Trap: that rounds to one decimal place instead of the cent.",
}
QUIZ_2: Final = {
    "skill_tag": SKILL,
    "rule_code": "RULE-001",
    "stem": "Three items at 19.99 with 20% VAT: what total does the fixed code charge?",
    "option_a": "71.964 (VAT left unquantized)",
    "option_b": "71.97 (VAT 11.994 rounded up)",
    "option_c": "Decimal('71.96')",
    "option_d": "71.96399999999998 (float sum)",
    "correct_option": "C",
    "explanation_a": "Trap: the VAT amount must be quantized to the cent before it is added.",
    "explanation_b": "Trap: 11.994 is below the half-cent tie, so ROUND_HALF_UP keeps 11.99.",
    "explanation_c": "Right: 59.97 net plus VAT 11.99 (11.994 quantized half-up) is 71.96.",
    "explanation_d": "Trap: binary float arithmetic, exactly what RULE-001 forbids for money.",
}


# ---- the scripted agents (real-seed fix, real rule codes, flat quiz arguments of #14 §3B) ---------------------------
def coder_script() -> list[ChatTurnResponse]:
    """Rules first, read before write, the RULE-001 fix of orders.py, the FULL staging suite, then done."""
    return [
        turn(call("search_notion_rules", query="VAT rounding Decimal")),
        turn(call("read_notion_rule", rule_code="RULE-001")),
        turn(call("read_code_file", path="orders.py")),
        turn(call("write_staging_file", path="orders.py", content=FIXED_FINTECH_ORDERS)),
        turn(call("run_staging_tests")),
        turn(text="STATUS: DONE\nTotals are Decimal with ROUND_HALF_UP (RULE-001); the full staging suite is green."),
    ]


def examiner_script(attempt: int, quiz: dict[str, str]) -> list[ChatTurnResponse]:
    """One fresh Examiner attempt. The v2 doc omits the remediation callout on purpose: the harness binder adds it."""
    doc = (
        f"## Technical Feature Documentation\nMoney is Decimal end to end, quantized to the cent (RULE-001), "
        f"doc v{attempt}.\n\n## Pedagogical Lesson\nROUND_HALF_UP rounds an exact tie up; a float may never hold it."
    )
    upsert = call(
        "upsert_ship_log_doc",
        title="Decimal ROUND_HALF_UP totals in orders.py",
        doc_markdown=doc,
        changed_files=["orders.py"],
        applied_rules=["RULE-001"],
        targeted_weaknesses=[SKILL],
    )
    return [
        turn(upsert),
        turn(call("create_comprehension_quiz", **quiz)),
        turn(text=f"STATUS: DONE\nDoc v{attempt} and quiz attempt {attempt} are published."),
    ]


# ---- the live Studio -----------------------------------------------------------------------------------------------
@dataclass
class LiveStudio:
    """The Studio app wired to real collaborators, plus the handles the assertions need."""

    client: TestClient
    llm: ScriptedLLM
    settings: RuntimeSettings
    workflows: list[NoQuizWorkflow]

    @property
    def workflow(self) -> NoQuizWorkflow:
        assert len(self.workflows) == 1, "one thread is driven by one workflow across every request"
        return self.workflows[0]

    def prod_tree(self) -> str:
        return compute_tree_sha256(self.settings.area_dir("prod"))

    def prod_file(self, name: str) -> str:
        return (self.settings.area_dir("prod") / name).read_text(encoding="utf-8")

    def remote_git(self, *args: str) -> str:
        """git against the default local bare release remote, isolated from the developer's git config."""
        env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
        env |= {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, **HERMETIC_RELEASE_ENV}
        argv = ["git", "--git-dir", str(self.settings.default_release_remote_dir), *args]
        return subprocess.run(argv, env=env, capture_output=True, text=True, check=True).stdout.strip()

    def release_tip(self) -> str | None:
        """Tip of the release branch on the local remote; None while nothing was ever pushed."""
        if not (self.settings.default_release_remote_dir / "HEAD").is_file():
            return None
        ref = f"refs/heads/{self.settings.git_branch}"
        listing = self.remote_git("for-each-ref", "--format=%(objectname)", ref)  # reads the bare remote's own refs
        return listing or None

    def stream_run(self) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        payload = {"prompt": TASK, "thread_id": THREAD, "user_id": USER, "model": SCRIPTED_MODEL, "topology": "dual"}
        response = self.client.post("/api/runs/stream", json=payload)
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        return response.text, parse_sse_events(response.text)

    def answer(self, option: str) -> dict[str, Any]:
        response = self.client.post(f"/api/runs/{THREAD}/quiz", json={"selected_option": option, "user_id": USER})
        assert response.status_code == 200, response.text
        return response.json()

    def approve(self, expected_diff_sha256: str | None = None) -> httpx.Response:
        body = {"action": "approve", "user_id": USER, "reason": "e2e reviewer"}
        if expected_diff_sha256 is not None:
            body["expected_diff_sha256"] = expected_diff_sha256
        return self.client.post(f"/api/runs/{THREAD}/deploy", json=body)


@pytest.fixture
def live_studio(tmp_path: Path) -> Iterator[LiveStudio]:
    settings = RuntimeSettings(workspace_dir=tmp_path / "workspace")  # no Notion key: no live DB write-through
    llm = ScriptedLLM(coder=coder_script(), examiner=examiner_script(1, QUIZ_1) + examiner_script(2, QUIZ_2))
    workflows: list[NoQuizWorkflow] = []

    def workflow_factory(run_settings: RuntimeSettings, topology: Topology, model: str) -> NoQuizWorkflow:
        workflow = NoQuizWorkflow(
            settings=run_settings, llm_client=llm, topology=topology, model=model, mcp_transport="stdio"
        )
        workflows.append(workflow)
        return workflow

    app = create_app(settings=settings, workspace=SandboxWorkspace(settings), workflow_factory=workflow_factory)
    try:
        with TestClient(app) as client:
            yield LiveStudio(client=client, llm=llm, settings=settings, workflows=workflows)
    finally:
        for workflow in workflows:  # both stdio MCP servers stop before tmp_path is removed
            workflow.close()


def skill_record(records: list[dict[str, Any]], tag: str = SKILL) -> dict[str, Any]:
    return next(record for record in records if record["skill_tag"] == tag)


def assert_gate_locked(response: httpx.Response, *missing: str) -> dict[str, Any]:
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "gate_locked" and body.get("deployed") is not True
    assert set(missing) <= set(body.get("missing_locks", [])), body
    return body


# ---- the tests -----------------------------------------------------------------------------------------------------
def test_studio_drives_remediation_to_production_over_real_stdio_mcp(live_studio: LiveStudio) -> None:
    client, llm = live_studio.client, live_studio.llm

    # 0. A pristine seeded sandbox through the API (real SandboxWorkspace.reset, Layer 4 wiped)
    reset = client.post("/api/reset", json={"user_id": USER, "reset_skill_memory": True}).json()
    seed_prod = reset["prod_tree_sha256"]
    assert reset["reset"] and seed_prod == reset["staging_tree_sha256"] == live_studio.prod_tree()
    assert [rule["rule_code"] for rule in reset["state"]["notion"]["rules"]] == SEEDED_RULES

    # 1. Coder -> fresh Examiner (doc v1 + quiz 1) streamed over SSE, paused on the quiz gate
    raw, frames = live_studio.stream_run()
    assert frames[0][0] == "run_started" and frames[-1][0] == "run_completed"
    seqs = [payload["seq"] for _, payload in frames]
    assert seqs == sorted(set(seqs))
    steps = [payload["step"] for kind, payload in frames if kind == "step"]
    assert [(step["agent"], step["tool"]) for step in steps] == [
        ("coder", "search_notion_rules"),
        ("coder", "read_notion_rule"),
        ("coder", "read_code_file"),
        ("coder", "write_staging_file"),
        ("coder", "run_staging_tests"),
        ("examiner", "upsert_ship_log_doc"),
        ("examiner", "create_comprehension_quiz"),
    ]
    assert all(step["error"] is None for step in steps)
    assert steps[4]["result"]["exit_code"] == 0  # the real seeded suite ran green on the staged fix
    paused = frames[-1][1]
    assert paused["stop_reason"] is None and paused["interrupt"]["kind"] == "quiz"
    quiz_1 = paused["interrupt"]["quiz"]
    diff_sha = quiz_1["diff_sha256"]
    assert quiz_1["attempt"] == 1 and "correct_option" not in quiz_1
    assert SEALED in raw and TRAP_B not in raw and QUIZ_1["explanation_a"] not in raw  # the key never streams
    assert isinstance(live_studio.workflow.bridge, StdioMCPBridge)
    assert (live_studio.settings.workspace_dir / MCP_SERVER_LOG_FILENAME).is_file()

    # 2. A premature approval is refused server-side: Lock 3 (quiz) is missing
    assert_gate_locked(live_studio.approve(), "quiz_passed")
    assert live_studio.prod_tree() == seed_prod and live_studio.release_tip() is None  # nothing pushed

    # 3. Wrong answer -> Layer 4 "Weakness Detected" -> a fresh Examiner publishes doc v2 + a new quiz 2
    wrong = live_studio.answer("B")
    grade = wrong["grade"]
    assert not grade["passed"] and grade["remediation_required"] and grade["explanation"] == TRAP_B
    assert grade["correct_option"] is None  # the key stays sealed while an attempt remains
    memory = skill_record(wrong["skill_profile"])
    assert (memory["fail_count"], memory["pass_count"], memory["mastery"]) == (1, 0, "Weakness Detected")
    assert memory["last_misconception"] == TRAP_B
    ship_log = wrong["ship_log"]
    assert (ship_log["doc_version"], ship_log["status"], ship_log["quiz_attempts"]) == ("v2", "Remediation v2", 1)
    assert REMEDIATION_HEADING.format(option="B") in ship_log["doc_markdown"]  # bound by the harness, stored by MCP
    assert SKILL in ship_log["targeted_weaknesses"]
    quiz_2 = wrong["next_quiz"]
    assert quiz_2["attempt"] == 2 and quiz_2["question_id"] != quiz_1["question_id"] and "correct_option" not in quiz_2
    assert wrong["interrupt"]["kind"] == "quiz" and wrong["locks"]["missing_locks"] == ["quiz_passed", "human_approved"]
    first_brief, remediation_brief = llm.briefs()
    assert "Weakness Detected" not in first_brief
    assert "Weakness Detected" in remediation_brief and "picked option B" in remediation_brief
    assert TRAP_B in remediation_brief  # the Layer 4 misconception reached the v2 Examiner's fresh context

    # 4. Right answer -> Lock 3 holds, the run pauses on the human deploy approval
    right = live_studio.answer("C")
    assert right["grade"]["passed"] and right["grade"]["correct_option"] == "C"
    assert right["interrupt"]["kind"] == "deploy_approval" and right["interrupt"]["gate"]["ready_for_approval"]
    assert right["locks"]["missing_locks"] == ["human_approved"]
    memory = skill_record(right["skill_profile"])
    assert (memory["fail_count"], memory["pass_count"], memory["mastery"]) == (1, 1, "Practicing")

    # 5. A mismatched hash is refused (Lock 2), and even the harness's own stdio route cannot skip the human (Lock 4)
    assert_gate_locked(live_studio.approve(WRONG_SHA), "sha256_matches")
    bypass = live_studio.workflow.bridge.call(
        "deploy_to_production", {"expected_diff_sha256": diff_sha, "release_note": "bypass attempt"}
    )
    assert bypass.error is not None and bypass.error.code == "gate_locked" and not bypass.error.retryable
    assert bypass.error.details["missing_locks"] == ["human_approved"]
    assert live_studio.prod_tree() == seed_prod and live_studio.release_tip() is None  # nothing pushed

    # 6. The human approves the reviewed hash -> system deploy over stdio -> prod == staging, Ship Log deployed
    response = live_studio.approve(diff_sha)
    assert response.status_code == 200, response.text
    deployed = response.json()
    assert deployed["deployed"] is True and deployed["stop_reason"] == "completed" and deployed["deployed_at"]
    assert deployed["diff"]["trees_identical"] and deployed["ship_log"]["status"] == "Deployed to Prod"
    assert all(deployed["locks"][lock] for lock in ALL_DEPLOY_LOCKS) and not deployed["locks"]["blocked"]
    assert live_studio.prod_file("orders.py") == FIXED_FINTECH_ORDERS and live_studio.prod_tree() != seed_prod

    # 6b. The deploy is a real git release on the local bare remote: content == prod/, gate evidence in its trailers
    release = live_studio.release_tip()
    assert release is not None and deployed["ship_log"]["release_commit_sha"] == release
    prod_dir = live_studio.settings.area_dir("prod")
    files = live_studio.remote_git("ls-tree", "-r", "--name-only", release).splitlines()
    assert sorted(files) == sorted(path.name for path in prod_dir.iterdir() if path.is_file())
    for name in files:
        assert live_studio.remote_git("show", f"{release}:{name}") == (prod_dir / name).read_text().strip()
    trailers = live_studio.remote_git("log", "-1", "--format=%(trailers:unfold,only)", release).splitlines()
    assert {
        f"Diff-SHA256: {diff_sha}",
        "Quiz-Attempts: 2",
        "Approved-By: human",
        "Rules: RULE-001",
        f"Thread: {THREAD}",
    } <= set(trailers)
    assert live_studio.workflow.workspace.snapshot_final_state(THREAD).pushed

    # 7. Replaying the approval cannot ship anything twice
    assert_gate_locked(live_studio.approve(diff_sha), "tests_passed")
    assert live_studio.release_tip() == release  # nothing pushed on top

    # 8. Trace: only the system role deployed, no model ever saw the deploy tool, every scripted turn was used
    trace = live_studio.workflow.app.get_state({"configurable": {"thread_id": THREAD}}).values["trace"]
    deploys = [entry for entry in trace if entry["tool"] == "deploy_to_production"]
    assert len(deploys) == 1 and deploys[0]["agent"] == "system" and deploys[0]["error"] is None
    assert deploys[0]["idempotency_key"] and [entry["agent"] for entry in trace].count("human") == 3
    assert all("deploy_to_production" not in seen.tools for seen in llm.calls)
    assert llm.remaining() == {"coder": 0, "examiner": 0}

    # 9. /api/state replays the final artifacts
    state = client.get("/api/state", params={"user_id": USER, "thread_id": THREAD}).json()
    run = state["active_run"]
    assert run["stop_reason"] == "completed" and run["ship_log"]["status"] == "Deployed to Prod"
    assert all(run["locks"][lock] for lock in ALL_DEPLOY_LOCKS) and state["workspace"]["diff"]["trees_identical"]
    assert skill_record(state["skill_profile"])["mastery"] == "Practicing"


def test_gate_answers_409_when_nothing_is_staged_or_resumable(live_studio: LiveStudio) -> None:
    client = live_studio.client
    seed_prod = client.post("/api/reset", json={"user_id": USER, "reset_skill_memory": True}).json()["prod_tree_sha256"]

    assert_gate_locked(live_studio.approve(), "tests_passed")  # staging == prod: nothing to ship
    unknown_quiz = client.post(f"/api/runs/{THREAD}/quiz", json={"selected_option": "A", "user_id": USER})
    assert_gate_locked(unknown_quiz)  # no paused quiz interrupt on this thread

    assert live_studio.prod_tree() == seed_prod and live_studio.release_tip() is None  # nothing pushed
    assert live_studio.llm.calls == []  # refused before any agent ran
