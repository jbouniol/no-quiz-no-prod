"""Non-regression: a GitHub remote and token in ``.env`` never reach a test, an eval run or a spawned MCP server.

Every test here points the project at a fake ``.env`` (``NO_QUIZ_ENV_FILE``) that configures a real-looking GitHub
remote and token, then proves each launch path still releases to the workspace's local bare remote only. Without the
hermetic environment the spawned server would resolve ``github.com`` and the deploy would fail (or, online, push).
"""

import os
from pathlib import Path
from typing import Final

import pytest
from mcp import Client
from testkit import FIXED_FINTECH_ORDERS, HERMETIC_RELEASE_ENV

from no_quiz_no_prod.backend import SandboxWorkspace, catalog
from no_quiz_no_prod.contracts import (
    DEFAULT_GIT_BRANCH,
    DeployDecision,
    EnvVar,
    OptionTexts,
    QuizDraft,
    ReleaseCommit,
    RuntimeSettings,
    load_project_env,
    project_env_file,
)
from no_quiz_no_prod.graph.mcp_bridge import StdioMCPBridge
from no_quiz_no_prod.mcp_servers.client import decode_tool_result, server_parameters

FORBIDDEN_URL: Final = "https://github.com/example/must-not-be-used.git"
FAKE_TOKEN: Final = "ghp_FAKEtokenMUSTNOTBEUSED0123456789abcd"
PROBE_MODEL: Final = "fake/model-from-the-fake-env-file"
CLIENT_TIMEOUT_S: Final = 60.0
QUIZ: Final = QuizDraft(
    skill_tag="decimal_rounding",
    rule_code="RULE-001",
    stem="Why does round(1.045, 2) give 1.04 with floats?",
    options=OptionTexts(A="Below the tie", B="Truncation", C="Law", D="Half even"),
    correct_option="A",
    explanation_by_option=OptionTexts(A="Right", B="Trap B", C="Trap C", D="Trap D"),
)


@pytest.fixture
def fake_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``.env`` with a GitHub remote and token, made the project's ``.env`` for this test."""
    path = tmp_path / "fake.env"
    path.write_text(
        f"NO_QUIZ_GIT_REMOTE_URL={FORBIDDEN_URL}\n"
        f"NO_QUIZ_GITHUB_TOKEN={FAKE_TOKEN}\n"
        "NO_QUIZ_GIT_BRANCH=must-not-be-used\n"
        f"OPENROUTER_MODEL={PROBE_MODEL}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EnvVar.NO_QUIZ_ENV_FILE.value, str(path))
    # set-then-delete: monkeypatch restores the original state, so loading the probe below cannot leak to other tests
    monkeypatch.setenv("OPENROUTER_MODEL", "placeholder")
    monkeypatch.delenv("OPENROUTER_MODEL")
    return path


def gated_workspace(tmp_path: Path) -> tuple[SandboxWorkspace, str]:
    """A harness-side workspace (explicit settings, no remote) whose staged fix holds Locks 1-4."""
    harness = SandboxWorkspace(RuntimeSettings(workspace_dir=tmp_path / "workspace"))
    harness.reset()
    diff = harness.write_staging_file("orders.py", FIXED_FINTECH_ORDERS)["diff_sha256"]
    assert harness.run_staging_tests().passed
    harness.upsert_ship_log(
        title="Fix VAT float drift",
        diff_sha256=diff,
        changed_files=["orders.py"],
        applied_rules=["RULE-001"],
        targeted_weaknesses=["decimal_rounding"],
        doc_markdown="## Lesson\nMoney is Decimal.",
    )
    question = harness.create_quiz(QUIZ, diff_sha256=diff)
    assert harness.grade_quiz_answer(question.question_id, "A").passed
    harness.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))
    return harness, diff


def assert_released_locally(harness: SandboxWorkspace, payload: dict[str, object]) -> None:
    release = ReleaseCommit.model_validate(payload["release"])
    remote = harness.settings.default_release_remote_dir
    assert (release.remote_kind, release.remote, release.branch) == ("local", str(remote), DEFAULT_GIT_BRANCH)
    assert release.pushed and release.commit_url is None
    final = harness.snapshot_final_state()
    assert final.deployed and final.pushed and final.release_commit_sha == release.commit_sha


# ---- the .env precedence rule everything relies on -----------------------------------------------------------------
def test_the_process_environment_wins_over_dotenv_even_when_empty(fake_env_file: Path) -> None:
    assert project_env_file() == fake_env_file
    assert os.environ[EnvVar.NO_QUIZ_GIT_REMOTE_URL.value] == ""  # conftest's hermetic value
    assert load_project_env()
    assert os.environ["OPENROUTER_MODEL"] == PROBE_MODEL  # the fake file really is the one being read...
    for name, value in HERMETIC_RELEASE_ENV.items():
        assert os.environ[name] == value  # ...yet an already-set variable is never replaced, even an empty one
    settings = RuntimeSettings.from_env(notion_ids_file=None)
    assert (settings.git_remote_url, settings.github_token, settings.git_branch) == (None, None, DEFAULT_GIT_BRANCH)


def test_every_launch_path_blanks_the_release_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EnvVar.NO_QUIZ_GIT_REMOTE_URL.value, FORBIDDEN_URL)  # as if a .env had been loaded already
    monkeypatch.setenv(EnvVar.NO_QUIZ_GITHUB_TOKEN.value, FAKE_TOKEN)
    for env in (
        server_parameters(catalog.CODE_SERVER, workspace_dir=tmp_path).env,
        StdioMCPBridge(RuntimeSettings(workspace_dir=tmp_path)).server_parameters(catalog.CODE_SERVER).env,
    ):
        assert env is not None
        assert env[EnvVar.NO_QUIZ_GIT_REMOTE_URL.value] == "" and env[EnvVar.NO_QUIZ_GITHUB_TOKEN.value] == ""
        assert env[EnvVar.NO_QUIZ_GIT_BRANCH.value] == DEFAULT_GIT_BRANCH
    explicit = server_parameters(catalog.CODE_SERVER, env={EnvVar.NO_QUIZ_GIT_REMOTE_URL.value: "/srv/release.git"})
    assert explicit.env is not None and explicit.env[EnvVar.NO_QUIZ_GIT_REMOTE_URL.value] == "/srv/release.git"


# ---- the real stdio servers, reading the fake .env, only push to the workspace's local remote ----------------------
async def test_stdio_client_deploy_ignores_a_github_remote_in_dotenv(tmp_path: Path, fake_env_file: Path) -> None:
    harness, diff = gated_workspace(tmp_path)
    env = {"NOTION_API_KEY": "", EnvVar.NO_QUIZ_ENV_FILE.value: str(fake_env_file)}  # no release override given
    params = server_parameters(catalog.CODE_SERVER, workspace_dir=harness.root, env=env)
    async with Client(params, read_timeout_seconds=CLIENT_TIMEOUT_S) as code:
        args = {"expected_diff_sha256": diff, "release_note": "Fix VAT"}
        shipped = decode_tool_result(await code.call_tool(catalog.DEPLOY_TO_PRODUCTION, args))
    assert shipped.ok, shipped.error
    assert_released_locally(harness, shipped.payload)


def test_stdio_bridge_deploy_ignores_a_github_remote_in_dotenv(
    tmp_path: Path, fake_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EnvVar.NO_QUIZ_GIT_REMOTE_URL.value, FORBIDDEN_URL)  # even loaded into the harness process
    monkeypatch.setenv(EnvVar.NO_QUIZ_GITHUB_TOKEN.value, FAKE_TOKEN)
    harness, diff = gated_workspace(tmp_path)
    with StdioMCPBridge(harness.settings) as bridge:  # NO_QUIZ_ENV_FILE passes through: the servers read the fake
        outcome = bridge.call(catalog.DEPLOY_TO_PRODUCTION, {"expected_diff_sha256": diff, "release_note": "Fix VAT"})
    assert outcome.ok, outcome.error
    assert_released_locally(harness, outcome.result)
