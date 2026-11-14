"""The deploy as a real git release: commit + fast-forward push, gated by the 4 locks, never forced, token-safe.

Offline only: every remote is a local bare repository under ``tmp_path``; the one HTTPS (GitHub-like) case fakes the
three network git commands (``ls-remote``, ``fetch``, ``push``) and runs every local git command for real.
"""

import base64
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import SecretStr
from testkit import FIXED_FINTECH_ORDERS

from no_quiz_no_prod.agents import AGENTS, TOOLS
from no_quiz_no_prod.backend import SandboxWorkspace, WorkspaceError, catalog
from no_quiz_no_prod.backend import release as release_module
from no_quiz_no_prod.contracts import (
    DEFAULT_GIT_BRANCH,
    DeployDecision,
    EnvVar,
    OptionTexts,
    QuizDraft,
    ReleaseCommit,
    RuntimeSettings,
    ToolErrorPayload,
)
from no_quiz_no_prod.graph import workflow as workflow_module
from no_quiz_no_prod.graph.mcp_bridge import StdioMCPBridge

TOKEN: Final = "ghp_TESTONLYnotarealtoken0123456789abcd"
TOKEN_B64: Final = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
GITHUB_URL: Final = "https://github.com/example-owner/example-release.git"
NETWORK_COMMANDS: Final = frozenset({"ls-remote", "fetch", "push"})
QUIZ: Final = QuizDraft(
    skill_tag="decimal_rounding",
    rule_code="RULE-001",
    stem="Why does round(1.045, 2) give 1.04 with floats?",
    options=OptionTexts(
        A="The float is stored just below 1.045",
        B="round() always truncates",
        C="VAT must be rounded down by law",
        D="Decimal rounds half to even",
    ),
    correct_option="A",
    explanation_by_option=OptionTexts(A="Right", B="Trap B", C="Trap C", D="Trap D"),
)
SECOND_FIX: Final = FIXED_FINTECH_ORDERS + "\n\n# Release 2: no behaviour change, documentation only.\n"


# ---- helpers ---------------------------------------------------------------------------------------------------------
def git(*args: str, cwd: Path | None = None) -> str:
    """Test-side git, isolated from the developer's config like the gate itself."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env |= {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Someone Else",
        "GIT_AUTHOR_EMAIL": "someone@example.invalid",
        "GIT_COMMITTER_NAME": "Someone Else",
        "GIT_COMMITTER_EMAIL": "someone@example.invalid",
    }
    done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def bare_repo(path: Path) -> Path:
    git("init", "--quiet", "--bare", str(path))
    return path


def tip(remote: Path, branch: str = DEFAULT_GIT_BRANCH) -> str | None:
    listing = git("ls-remote", "--heads", str(remote), f"refs/heads/{branch}")
    return listing.split("\t", 1)[0] if listing else None


def refusal(action: Callable[[], Any]) -> ToolErrorPayload:
    with pytest.raises(WorkspaceError) as caught:
        action()
    return caught.value.payload


def ship(sandbox: SandboxWorkspace, content: str = FIXED_FINTECH_ORDERS, *, approve: bool = True) -> str:
    """Stage ``content`` as orders.py and earn Locks 1-3 (and Lock 4 when ``approve``) on it."""
    diff = sandbox.write_staging_file("orders.py", content)["diff_sha256"]
    assert sandbox.run_staging_tests().passed
    sandbox.upsert_ship_log(
        title="Fix VAT float drift in orders.py",
        diff_sha256=diff,
        changed_files=["orders.py"],
        applied_rules=["RULE-001"],
        targeted_weaknesses=["decimal_rounding"],
        doc_markdown="## What changed\nMoney is Decimal end to end.",
    )
    question = sandbox.create_quiz(QUIZ, diff_sha256=diff)
    assert sandbox.grade_quiz_answer(question.question_id, "A").passed
    if approve:
        sandbox.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))
    return diff


def workspace_for(tmp_path: Path, **settings: Any) -> SandboxWorkspace:
    sandbox = SandboxWorkspace(RuntimeSettings(workspace_dir=tmp_path / "workspace", **settings))
    sandbox.reset()
    return sandbox


# ---- default: a local bare remote ------------------------------------------------------------------------------------
def test_deploy_commits_the_approved_tree_and_pushes_it_to_the_local_bare_remote(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    seed_orders = (sandbox.prod_dir / "orders.py").read_text()
    diff = ship(sandbox)
    payload = sandbox.deploy_to_production(diff, "Fix VAT float drift", idempotency_key="deploy-1")

    release = ReleaseCommit.model_validate(payload["release"])
    remote = sandbox.settings.default_release_remote_dir
    assert (release.remote_kind, release.remote, release.branch) == ("local", str(remote), DEFAULT_GIT_BRANCH)
    assert release.pushed and release.diff_sha256 == diff and release.prod_tree_sha256 == payload["prod_tree_sha256"]
    assert tip(remote) == release.commit_sha
    # the release commit holds exactly the promoted prod tree, on top of a baseline commit holding the seed prod tree
    assert git("--git-dir", str(remote), "show", f"{release.commit_sha}:orders.py") == FIXED_FINTECH_ORDERS.strip()
    assert git("--git-dir", str(remote), "show", f"{release.parent_sha}:orders.py") == seed_orders.strip()
    assert git("--git-dir", str(remote), "rev-list", "--count", release.commit_sha) == "2"
    files = git("--git-dir", str(remote), "ls-tree", "-r", "--name-only", release.commit_sha).splitlines()
    assert sorted(files) == sorted(sandbox.list_files("prod"))  # no __pycache__ / .pyc from the pytest run
    assert git("--git-dir", str(remote), "diff", "--name-only", release.parent_sha, release.commit_sha) == "orders.py"
    message = git("--git-dir", str(remote), "log", "-1", "--format=%an%n%B", release.commit_sha)
    assert message.startswith("No-Quiz-No-Prod Gate\ndeploy: Fix VAT float drift") and f"Diff-SHA256: {diff}" in message
    # the release is recorded everywhere the deploy is
    entry = sandbox.get_latest_ship_log()
    assert entry is not None and entry.status == "Deployed to Prod" and entry.release_commit_sha == release.commit_sha
    final = sandbox.snapshot_final_state()
    assert final.deployed and final.release_commit_sha == release.commit_sha
    # an idempotent replay answers from the store: no second commit
    assert sandbox.deploy_to_production(diff, "Fix VAT float drift", idempotency_key="deploy-1")["replayed"]
    assert tip(remote) == release.commit_sha
    # the gate never configures a remote and never writes a tracking ref
    assert "remote" not in (sandbox.settings.release_git_dir / "config").read_text()


def test_successive_deploys_fast_forward_and_a_reset_adds_a_baseline_commit(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    remote = sandbox.settings.default_release_remote_dir
    first = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 1")["release"])
    second = ReleaseCommit.model_validate(
        sandbox.deploy_to_production(ship(sandbox, SECOND_FIX), "Release 2")["release"]
    )
    assert second.parent_sha == first.commit_sha  # prod == remote tip: no baseline in between
    sandbox.reset()  # prod is back to the seed, the remote keeps its history
    third = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 3")["release"])
    assert git("--git-dir", str(remote), "rev-parse", f"{third.parent_sha}^") == second.commit_sha
    assert git("--git-dir", str(remote), "merge-base", "--is-ancestor", first.commit_sha, third.commit_sha) == ""
    assert git("--git-dir", str(remote), "rev-list", "--count", third.commit_sha) == "5"  # linear, nothing rewritten
    assert git("--git-dir", str(remote), "diff", "--name-only", third.parent_sha, third.commit_sha) == "orders.py"


def test_a_locked_or_rejected_deploy_pushes_nothing(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    remote = sandbox.settings.default_release_remote_dir
    diff = ship(sandbox, approve=False)
    assert refusal(lambda: sandbox.deploy_to_production(diff, "too early")).code == "gate_locked"
    sandbox.apply_deploy_decision(DeployDecision(action="reject", expected_diff_sha256=diff))
    assert refusal(lambda: sandbox.deploy_to_production(diff, "rejected")).code == "gate_locked"
    assert not remote.exists() or tip(remote) is None
    assert not sandbox.snapshot_final_state().prod_changed


def test_a_retried_release_is_never_committed_or_pushed_twice(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    diff = ship(sandbox)
    publish = {
        "source_dir": sandbox.staging_dir,
        "baseline_dir": sandbox.prod_dir,
        "diff_sha256": diff,
        "prod_tree_sha256": "0" * 64,
        "release_note": "Fix VAT",
    }
    first = sandbox.release.publish(**publish)  # e.g. the push landed, then the answer was lost
    again = sandbox.release.publish(**publish)
    assert first.pushed and not again.pushed
    assert (again.commit_sha, again.parent_sha) == (first.commit_sha, first.parent_sha)
    assert tip(sandbox.settings.default_release_remote_dir) == first.commit_sha


# ---- a configured remote that moved: never forced --------------------------------------------------------------------
def test_a_moved_remote_is_never_force_pushed_and_prod_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = bare_repo(tmp_path / "team_release.git")
    clone = tmp_path / "someone_else"
    git("init", "--quiet", "-b", DEFAULT_GIT_BRANCH, str(clone))
    (clone / "HOTFIX.md").write_text("pushed by someone else\n")
    git("add", "HOTFIX.md", cwd=clone)
    git("commit", "--quiet", "-m", "hotfix", cwd=clone)
    git("push", "--quiet", str(remote), f"{DEFAULT_GIT_BRANCH}:refs/heads/{DEFAULT_GIT_BRANCH}", cwd=clone)
    theirs = tip(remote)

    sandbox = workspace_for(tmp_path, git_remote_url=f"file://{remote}")
    diff = ship(sandbox)
    # the gate reads a stale view of the branch (it moved between the read and the push): the push must be refused
    monkeypatch.setattr(release_module.GitReleasePublisher, "_remote_tip", lambda self, remote, branch: None)
    error = refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT"))
    assert error.code == "conflict" and error.retryable
    assert tip(remote) == theirs  # nothing overwritten
    final = sandbox.snapshot_final_state()
    assert not final.prod_changed and not final.deployed and final.release_commit_sha is None
    assert sandbox.get_gate_status(diff).can_deploy  # the locks still hold: a retry is allowed

    monkeypatch.undo()
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(diff, "Fix VAT")["release"])
    assert release.remote == str(remote) and tip(remote) == release.commit_sha
    assert git("--git-dir", str(remote), "merge-base", "--is-ancestor", theirs, release.commit_sha) == ""
    assert sandbox.snapshot_final_state().deployed


@pytest.mark.parametrize("branch", ["-delete", "main..x", "feature/", "a b", "x.lock"])
def test_an_invalid_release_branch_is_refused_before_any_write(tmp_path: Path, branch: str) -> None:
    sandbox = workspace_for(tmp_path, git_branch=branch)
    diff = ship(sandbox)
    assert refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT")).code == "invalid_argument"
    assert not sandbox.snapshot_final_state().prod_changed


# ---- GitHub (HTTPS) option: the token only ever reaches git's environment --------------------------------------------
class FakeNetworkGit:
    """Runs local git commands for real and answers the network ones (``ls-remote``/``fetch``/``push``)."""

    def __init__(self, *, push_stderr: str | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.push_stderr = push_stderr
        self._real_run = subprocess.run

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if not argv or argv[0] != "git":  # e.g. the staging pytest run: not ours to record
            return self._real_run(argv, **kwargs)
        self.calls.append((list(argv), dict(kwargs.get("env") or {})))
        command = next((arg for arg in argv[1:] if not arg.startswith("--")), "")
        if command not in NETWORK_COMMANDS:
            return self._real_run(argv, **kwargs)
        if command == "push" and self.push_stderr is not None:
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr=self.push_stderr)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")  # ls-remote: the branch does not exist yet

    def network_envs(self) -> list[dict[str, str]]:
        return [env for argv, env in self.calls if any(arg in NETWORK_COMMANDS for arg in argv)]

    def local_envs(self) -> list[dict[str, str]]:
        return [env for argv, env in self.calls if not any(arg in NETWORK_COMMANDS for arg in argv)]


def assert_token_absent(*texts: str) -> None:
    for text in texts:
        assert TOKEN not in text and TOKEN_B64 not in text


def files_text(root: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in root.rglob("*") if p.is_file())


def github_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeNetworkGit) -> SandboxWorkspace:
    monkeypatch.setattr(release_module.subprocess, "run", fake)
    return workspace_for(tmp_path, git_remote_url=GITHUB_URL, github_token=SecretStr(TOKEN))


def test_github_push_sends_the_token_only_as_a_scoped_git_env_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNetworkGit()
    sandbox = github_workspace(tmp_path, monkeypatch, fake)
    payload = sandbox.deploy_to_production(ship(sandbox), "Fix VAT")

    release = ReleaseCommit.model_validate(payload["release"])
    assert (release.remote_kind, release.remote, release.pushed) == ("https", GITHUB_URL, True)
    push = next(argv for argv, _ in fake.calls if "push" in argv)
    assert push[-2:] == [GITHUB_URL, f"{release.commit_sha}:refs/heads/{DEFAULT_GIT_BRANCH}"]  # no "+", no --force
    assert not {"--force", "-f", "--force-with-lease", "--mirror", "--delete"} & set(push)
    assert all(TOKEN not in arg and TOKEN_B64 not in arg for argv, _ in fake.calls for arg in argv)
    header = f"Authorization: Basic {TOKEN_B64}"
    for env in fake.network_envs():
        keys = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
        assert keys[f"http.{GITHUB_URL}.extraHeader"] == header and keys["credential.helper"] == ""
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull and env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert all(header not in env.values() for env in fake.local_envs())  # commits and trees never see it
    inherited = {name for _, env in fake.calls for name in env if not name.startswith(("GIT_", "LC_", "LANG"))}
    assert inherited <= release_module.GIT_ENV_ALLOWLIST  # no API key of this process reaches git
    # nowhere on disk (.release.git/config, ledger, state.sqlite, logs) and nowhere in what the deploy returns
    assert_token_absent(files_text(sandbox.root), repr(payload), repr(sandbox.settings))
    assert_token_absent(repr(sandbox.snapshot_final_state()), repr(sandbox.get_latest_ship_log()))


def test_a_failed_github_push_never_echoes_the_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    leaky = f"fatal: Authentication failed (header Basic {TOKEN_B64}; token {TOKEN})"
    sandbox = github_workspace(tmp_path, monkeypatch, FakeNetworkGit(push_stderr=leaky))
    diff = ship(sandbox)
    with pytest.raises(WorkspaceError) as caught:
        sandbox.deploy_to_production(diff, "Fix VAT")
    error = caught.value
    assert error.code == "upstream_error" and "***" in error.payload.details["git_output"]
    assert_token_absent(str(error), error.payload.model_dump_json(), repr(error.__cause__), files_text(sandbox.root))
    assert not sandbox.snapshot_final_state().prod_changed  # a failed push never promotes prod


def test_a_remote_url_with_embedded_credentials_is_refused_without_echoing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNetworkGit()
    monkeypatch.setattr(release_module.subprocess, "run", fake)
    url = f"https://x-access-token:{TOKEN}@github.com/example-owner/example-release.git"
    sandbox = workspace_for(tmp_path, git_remote_url=url)
    diff = ship(sandbox)
    error = refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT"))
    assert error.code == "invalid_argument" and "NO_QUIZ_GITHUB_TOKEN" in error.message
    assert_token_absent(error.model_dump_json(), repr(sandbox.settings))
    started = len(fake.calls)  # a count only: a failing assertion must never print a recorded environment
    assert started == 0  # git never even started
    assert not sandbox.snapshot_final_state().prod_changed


# ---- settings, harness wiring and agents -----------------------------------------------------------------------------
def test_settings_read_the_release_remote_and_keep_secrets_out_of_the_repr(tmp_path: Path) -> None:
    environ = {
        EnvVar.NO_QUIZ_WORKSPACE_DIR.value: str(tmp_path),
        EnvVar.NO_QUIZ_GIT_REMOTE_URL.value: GITHUB_URL,
        EnvVar.NO_QUIZ_GIT_BRANCH.value: "release/prod",
        EnvVar.NO_QUIZ_GITHUB_TOKEN.value: TOKEN,
    }
    settings = RuntimeSettings.from_env(environ, notion_ids_file=None)
    assert (settings.git_remote_url, settings.git_branch) == (GITHUB_URL, "release/prod")
    assert settings.github_token is not None and settings.github_token.get_secret_value() == TOKEN
    assert_token_absent(repr(settings), str(settings))
    assert GITHUB_URL not in repr(settings)
    blank = RuntimeSettings.from_env({EnvVar.NO_QUIZ_GIT_REMOTE_URL.value: " "}, notion_ids_file=None)
    assert (blank.git_remote_url, blank.git_branch, blank.github_token) == (None, DEFAULT_GIT_BRANCH, None)


def test_stdio_servers_never_inherit_a_remote_the_harness_settings_do_not_have(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EnvVar.NO_QUIZ_GIT_REMOTE_URL.value, GITHUB_URL)
    monkeypatch.setenv(EnvVar.NO_QUIZ_GITHUB_TOKEN.value, TOKEN)
    env = StdioMCPBridge(RuntimeSettings(workspace_dir=tmp_path)).server_parameters(catalog.CODE_SERVER).env or {}
    assert env[EnvVar.NO_QUIZ_GIT_REMOTE_URL.value] == "" and env[EnvVar.NO_QUIZ_GITHUB_TOKEN.value] == ""
    assert env[EnvVar.NO_QUIZ_GIT_BRANCH.value] == DEFAULT_GIT_BRANCH


def test_no_agent_can_push() -> None:
    risky = ("git", "push", "commit", "release")
    assert not [name for name in TOOLS if any(word in name for word in risky)]
    for spec in AGENTS.values():
        assert catalog.DEPLOY_TO_PRODUCTION not in spec.tools
        assert not [name for name in spec.tools if any(word in name for word in risky)]


def test_the_completed_run_names_its_release_commit() -> None:
    release = ReleaseCommit(
        commit_sha="a" * 40,
        branch="main",
        remote="/tmp/release_remote.git",
        remote_kind="local",
        diff_sha256="b" * 64,
        prod_tree_sha256="c" * 64,
    )
    suffix = workflow_module._release_suffix({"ok": True, "release": release.model_dump(mode="json")})
    assert suffix == " Release commit aaaaaaaaaaaa is on main (/tmp/release_remote.git)."
    assert workflow_module._release_suffix({"ok": True}) == ""  # e.g. a scripted backend without git
