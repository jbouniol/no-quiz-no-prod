"""The deploy as a real git release: commit + fast-forward push, gated by the 4 locks, never forced, token-safe.

Offline only: every remote is a local bare repository under ``tmp_path``; the one HTTPS (GitHub-like) case fakes the
three network git commands (``ls-remote``, ``fetch``, ``push``) and runs every local git command for real.
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import SecretStr
from testkit import FIXED_FINTECH_ORDERS, HERMETIC_RELEASE_ENV

from no_quiz_no_prod.agents import AGENTS, TOOLS
from no_quiz_no_prod.backend import SandboxWorkspace, WorkspaceError, catalog
from no_quiz_no_prod.backend import release as release_module
from no_quiz_no_prod.cli import _cmd_init_github, build_parser
from no_quiz_no_prod.contracts import (
    DEFAULT_GIT_BRANCH,
    DeployDecision,
    EnvVar,
    ExpectedState,
    OptionTexts,
    QuizDraft,
    ReleaseCommit,
    RuntimeSettings,
    ToolErrorPayload,
    compute_tree_sha256,
    diff_sha256_from_trees,
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
OTHER_HOST_URL: Final = "https://git.attacker.example/example-owner/example-release.git"


# ---- helpers ---------------------------------------------------------------------------------------------------------
def git(*args: str, cwd: Path | None = None) -> str:
    """Test-side git, isolated from the developer's config like the gate itself."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env |= HERMETIC_RELEASE_ENV | {
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


def ship(
    sandbox: SandboxWorkspace, content: str = FIXED_FINTECH_ORDERS, *, approve: bool = True, **ship_log: Any
) -> str:
    """Stage ``content`` as orders.py and earn Locks 1-3 (and Lock 4 when ``approve``) on it."""
    diff = sandbox.write_staging_file("orders.py", content)["diff_sha256"]
    assert sandbox.run_staging_tests().passed
    fields: dict[str, Any] = {
        "title": "Fix VAT float drift in orders.py",
        "changed_files": ["orders.py"],
        "applied_rules": ["RULE-001"],
        "targeted_weaknesses": ["decimal_rounding"],
        "doc_markdown": "## What changed\nMoney is Decimal end to end.",
    }
    sandbox.upsert_ship_log(diff_sha256=diff, **(fields | ship_log))
    question = sandbox.create_quiz(QUIZ, diff_sha256=diff)
    assert sandbox.grade_quiz_answer(question.question_id, "A").passed
    if approve:
        sandbox.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))
    return diff


def workspace_for(tmp_path: Path, **settings: Any) -> SandboxWorkspace:
    sandbox = SandboxWorkspace(RuntimeSettings(workspace_dir=tmp_path / "workspace", **settings))
    sandbox.reset()
    return sandbox


def prepared_release(sandbox: SandboxWorkspace, **overrides: Any) -> release_module.PreparedRelease:
    """``prepare`` for what is staged now, as the gate calls it."""
    prod, staging = compute_tree_sha256(sandbox.prod_dir), compute_tree_sha256(sandbox.staging_dir)
    arguments: dict[str, Any] = {
        "source_dir": sandbox.staging_dir,
        "baseline_dir": sandbox.prod_dir,
        "diff_sha256": diff_sha256_from_trees(prod, staging),
        "approved_tree_sha256": staging,
        "baseline_tree_sha256": prod,
        "release_note": "Fix VAT",
    }
    return sandbox.release.prepare(**(arguments | overrides))


def remote_files(remote: Path, commit: str) -> list[str]:
    return git("--git-dir", str(remote), "ls-tree", "-r", "--name-only", commit).splitlines()


def teammate_pushes(remote: Path, clone: Path, name: str = "HOTFIX.md") -> str:
    """Someone else commits one more file on top of the release branch; returns the new tip."""
    git("clone", "--quiet", "--branch", DEFAULT_GIT_BRANCH, str(remote), str(clone))
    (clone / name).write_text("pushed by someone else\n")
    git("add", name, cwd=clone)
    git("commit", "--quiet", "-m", "hotfix", cwd=clone)
    git("push", "--quiet", str(remote), f"HEAD:refs/heads/{DEFAULT_GIT_BRANCH}", cwd=clone)
    theirs = tip(remote)
    assert theirs is not None
    return theirs


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
    assert (
        git("--git-dir", str(remote), "diff", "--name-only", str(release.parent_sha), release.commit_sha) == "orders.py"
    )
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


def test_successive_deploys_fast_forward_and_a_reset_starts_the_default_remote_again(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    remote = sandbox.settings.default_release_remote_dir
    first = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 1")["release"])
    second = ReleaseCommit.model_validate(
        sandbox.deploy_to_production(ship(sandbox, SECOND_FIX), "Release 2")["release"]
    )
    assert second.parent_sha == first.commit_sha  # prod == remote tip: nothing in between
    assert git("--git-dir", str(remote), "rev-list", "--count", second.commit_sha) == "3"
    sandbox.reset()  # prod is back to the seed and the workspace's own bare remote is discarded
    third = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 3")["release"])
    assert git("--git-dir", str(remote), "rev-list", "--count", third.commit_sha) == "2"  # baseline + release
    assert git("--git-dir", str(remote), "diff", "--name-only", str(third.parent_sha), third.commit_sha) == "orders.py"


def test_a_configured_remote_is_never_discarded(tmp_path: Path) -> None:
    remote = bare_repo(tmp_path / "team_release.git")
    sandbox = workspace_for(tmp_path, git_remote_url=str(remote))
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 1")["release"])
    sandbox.release.discard_default_remote()
    assert tip(remote) == release.commit_sha


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
    ship(sandbox)
    prepared = prepared_release(sandbox, thread_id="thr-1")
    announced: list[ReleaseCommit] = []
    first = sandbox.release.publish(prepared, before_push=announced.append)  # the push landed, the answer was lost
    assert announced == [first] and first.pushed  # the gate journals the commit before it is pushed
    again = sandbox.release.publish(prepared, before_push=announced.append)
    assert not again.pushed and len(announced) == 1  # recognised: nothing committed, announced or pushed again
    assert (again.commit_sha, again.parent_sha) == (first.commit_sha, first.parent_sha)
    assert tip(sandbox.settings.default_release_remote_dir) == first.commit_sha


def test_a_redeploy_is_recognised_by_identity_not_by_an_equal_tree(tmp_path: Path) -> None:
    """Audit PoC D at the publisher: the tip holds the same tree, shipped by another thread."""
    sandbox = workspace_for(tmp_path)
    ship(sandbox)
    first = sandbox.release.publish(prepared_release(sandbox, thread_id="thr-1"))
    other_thread = prepared_release(sandbox, thread_id="thr-2")
    assert other_thread.target_tree == git(
        "--git-dir", str(sandbox.settings.release_git_dir), "rev-parse", f"{first.commit_sha}^{{tree}}"
    )
    error = refusal(lambda: sandbox.release.publish(other_thread))
    assert error.code == "remote_diverged"  # not "already released": prod is not what the branch holds
    assert tip(sandbox.settings.default_release_remote_dir) == first.commit_sha
    forged_note = f"Looks released\n\nDiff-SHA256: {'0' * 64}\nThread: thr-9"  # trailers in the note do not count
    noted = prepared_release(sandbox, thread_id="thr-1", release_note=forged_note)
    assert not sandbox.release.publish(noted).pushed and release_module._trailers(noted.message)["Thread"] == "thr-1"


def test_an_equal_tree_redeploy_after_a_reset_is_not_reported_as_pushed(tmp_path: Path) -> None:
    """Audit PoC D end to end: run 2 ships the same fix after a reset; run 1's commit must not stand in for it."""
    remote = bare_repo(tmp_path / "team_release.git")
    sandbox = workspace_for(tmp_path, git_remote_url=str(remote))
    first = ReleaseCommit.model_validate(
        sandbox.deploy_to_production(ship(sandbox, thread_id="run-1"), "Run 1")["release"]
    )
    sandbox.reset()
    diff = ship(sandbox, thread_id="run-2")
    error = refusal(lambda: sandbox.deploy_to_production(diff, "Run 2"))
    assert error.code == "remote_diverged" and error.details["remote_tip"] == first.commit_sha
    final = sandbox.snapshot_final_state()
    assert not final.pushed and not final.deployed and final.release_commit_sha is None
    assert tip(remote) == first.commit_sha


# ---- a configured remote that moved: never forced --------------------------------------------------------------------
def test_a_moved_remote_is_never_force_pushed_and_prod_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = bare_repo(tmp_path / "team_release.git")
    sandbox = workspace_for(tmp_path, git_remote_url=f"file://{remote}")
    first = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Release 1")["release"])
    diff = ship(sandbox, SECOND_FIX)
    # the gate reads the branch, then someone else pushes before the gate does: the push must be refused
    real_remote_tip = release_module.GitReleasePublisher._remote_tip

    def read_then_race(self: Any, *args: Any) -> str | None:
        seen = real_remote_tip(self, *args)
        teammate_pushes(remote, tmp_path / "someone_else")
        return seen

    monkeypatch.setattr(release_module.GitReleasePublisher, "_remote_tip", read_then_race)
    error = refusal(lambda: sandbox.deploy_to_production(diff, "Release 2"))
    assert error.code == "conflict" and error.retryable
    theirs = tip(remote)
    assert theirs != first.commit_sha and "HOTFIX.md" in remote_files(remote, str(theirs))  # nothing overwritten
    final = sandbox.snapshot_final_state()
    assert not final.prod_changed or final.prod_tree_sha256 == first.prod_tree_sha256  # release 2 never promoted
    gate = sandbox.get_gate_status(diff)
    assert gate.missing_locks == ["human_approved"] and gate.ready_for_approval  # failed push revokes Lock 4
    assert refusal(lambda: sandbox.deploy_to_production(diff, "Release 2")).code == "gate_locked"

    sandbox.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))
    monkeypatch.undo()  # the retry now sees their commit: the branch no longer holds prod, a human must reconcile
    assert refusal(lambda: sandbox.deploy_to_production(diff, "Release 2")).code == "remote_diverged"
    assert tip(remote) == theirs


def test_a_diverged_remote_is_refused_and_no_sync_commit_reverts_a_teammate(tmp_path: Path) -> None:
    """Audit PoC C: release 1, a teammate's hotfix, a reset, then release 2 on another file."""
    remote = bare_repo(tmp_path / "team_release.git")
    sandbox = workspace_for(tmp_path, git_remote_url=str(remote))
    sandbox.deploy_to_production(ship(sandbox), "Release 1: fix orders.py")
    theirs = teammate_pushes(remote, tmp_path / "teammate")
    sandbox.reset()
    discounts = sandbox.read_file("discounts.py") + "\n# reviewed\n"
    diff = sandbox.write_staging_file("discounts.py", discounts)["diff_sha256"]
    assert sandbox.run_staging_tests().passed
    question = sandbox.create_quiz(QUIZ, diff_sha256=diff)
    assert sandbox.grade_quiz_answer(question.question_id, "A").passed
    sandbox.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))

    error = refusal(lambda: sandbox.deploy_to_production(diff, "Release 2: discounts only"))
    assert error.code == "remote_diverged" and not error.retryable
    assert error.details == {"branch": DEFAULT_GIT_BRANCH, "remote": str(remote), "remote_tip": theirs}
    assert tip(remote) == theirs  # not one commit was added: no "sync with production", no release
    assert "HOTFIX.md" in remote_files(remote, theirs)
    assert git("--git-dir", str(remote), "show", f"{theirs}:orders.py") == FIXED_FINTECH_ORDERS.strip()
    subjects = git("--git-dir", str(remote), "log", "--format=%s", theirs)
    assert "sync the release branch" not in subjects
    assert not sandbox.snapshot_final_state().prod_changed


# ---- the pushed tree is exactly the approved tree ---------------------------------------------------------------------
def test_a_global_ignore_file_cannot_change_the_pushed_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit PoC E: ``auth.py`` in the user's global git ignore used to vanish from the release."""
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".config" / "git" / "ignore").write_text("auth.py\n")
    (home / ".config" / "git" / "attributes").write_text("*.py text eol=crlf\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    sandbox = workspace_for(tmp_path)
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Fix VAT")["release"])
    remote = sandbox.settings.default_release_remote_dir
    assert sorted(remote_files(remote, release.commit_sha)) == sorted(sandbox.list_files("prod"))
    exported = tmp_path / "exported"
    exported.mkdir()
    sandbox.release.export_tree(release.commit_sha, exported)
    assert compute_tree_sha256(exported) == release.prod_tree_sha256 == compute_tree_sha256(sandbox.prod_dir)


def test_ignore_and_attributes_files_inside_the_tree_are_committed_not_obeyed(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    ship(sandbox)
    (sandbox.staging_dir / ".gitignore").write_text("*.py\n")
    (sandbox.staging_dir / ".gitattributes").write_text("*.py text eol=crlf\n* export-ignore\n")
    approved = compute_tree_sha256(sandbox.staging_dir)
    release = sandbox.release.publish(prepared_release(sandbox))
    remote = sandbox.settings.default_release_remote_dir
    assert sorted(remote_files(remote, release.commit_sha)) == sorted(sandbox.list_files("staging"))
    exported = tmp_path / "exported"
    exported.mkdir()
    sandbox.release.export_tree(release.commit_sha, exported)
    assert compute_tree_sha256(exported) == approved  # byte for byte: no end-of-line rewrite


def test_a_tree_that_changed_after_approval_is_never_committed(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    ship(sandbox)
    approved = compute_tree_sha256(sandbox.staging_dir)
    (sandbox.staging_dir / "orders.py").write_text("# slipped in after the approval\n")
    error = refusal(lambda: prepared_release(sandbox, approved_tree_sha256=approved))
    assert error.code == "invalid_argument"  # the diff hash no longer binds these trees
    prod = compute_tree_sha256(sandbox.prod_dir)
    stale = {"approved_tree_sha256": approved, "diff_sha256": diff_sha256_from_trees(prod, approved)}
    error = refusal(lambda: prepared_release(sandbox, **stale))
    assert error.code == "conflict" and error.details["tree"] == "staging"
    outside = tmp_path / "outside.txt"
    outside.write_text("never follow a symlink out of staging\n")
    (sandbox.staging_dir / "escape.py").symlink_to(outside)
    assert refusal(lambda: prepared_release(sandbox)).code == "forbidden"
    assert not sandbox.settings.default_release_remote_dir.exists()  # nothing was even started
    assert not list(sandbox.settings.release_git_dir.glob("snapshot-*"))  # no private copy is left behind


def test_export_tree_and_remote_has_commit_let_the_gate_complete_a_pushed_release(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    ship(sandbox)
    approved = compute_tree_sha256(sandbox.staging_dir)
    announced: list[ReleaseCommit] = []
    refused = refusal(
        lambda: sandbox.release.publish(
            prepared_release(sandbox), before_push=announced.append, reject_push=lambda: True
        )
    )
    assert refused.code == "upstream_error" and not sandbox.release.remote_has_commit(announced[0].commit_sha)
    release = sandbox.release.publish(prepared_release(sandbox))  # the push lands; the process dies before the swap
    (sandbox.staging_dir / "orders.py").write_text("# staging moved on\n")
    assert sandbox.release.remote_has_commit(release.commit_sha)
    assert not sandbox.release.remote_has_commit("f" * 40)
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    sandbox.release.export_tree(release.commit_sha, promoted)  # prod comes from the commit, not from staging
    assert compute_tree_sha256(promoted) == approved == release.prod_tree_sha256


def test_a_git_command_that_outlives_its_budget_is_killed_with_its_children(tmp_path: Path) -> None:
    marker = tmp_path / "survivor"
    script = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', \"import time, pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('x')\"]); time.sleep(30)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        release_module._run_process(
            [sys.executable, "-c", script], stdin=None, timeout=0.5, env=dict(os.environ), cwd=tmp_path
        )
    assert time.monotonic() - started < 10
    time.sleep(2.5)
    assert not marker.exists()  # the grandchild died with the group


@pytest.mark.parametrize("branch", ["-delete", "main..x", "feature/", "a b", "x.lock"])
def test_an_invalid_release_branch_is_refused_before_any_write(tmp_path: Path, branch: str) -> None:
    sandbox = workspace_for(tmp_path, git_branch=branch)
    diff = ship(sandbox)
    assert refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT")).code == "invalid_argument"
    assert not sandbox.snapshot_final_state().prod_changed


# ---- GitHub (HTTPS) option: the token only ever reaches git's environment --------------------------------------------
class FakeNetworkGit:
    """Runs local git commands for real and answers the network ones (``ls-remote``/``fetch``/``push``)."""

    def __init__(self, *, push_stderr: str | None = None, ls_remote: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.push_stderr = push_stderr
        self.ls_remote = ls_remote  # "" : the branch does not exist yet
        self._real_run = release_module._run_process

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(argv), dict(kwargs["env"])))
        command = next((arg for arg in argv[1:] if not arg.startswith("--")), "")
        if command not in NETWORK_COMMANDS:
            return self._real_run(argv, **kwargs)
        if command == "push" and self.push_stderr is not None:
            return subprocess.CompletedProcess(argv, 128, stdout=b"", stderr=self.push_stderr.encode())
        stdout = self.ls_remote if command == "ls-remote" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout.encode(), stderr=b"")

    def network_envs(self) -> list[dict[str, str]]:
        return [env for argv, env in self.calls if any(arg in NETWORK_COMMANDS for arg in argv)]

    def local_envs(self) -> list[dict[str, str]]:
        return [env for argv, env in self.calls if not any(arg in NETWORK_COMMANDS for arg in argv)]


def git_config(env: dict[str, str]) -> dict[str, str]:
    return {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}


def assert_token_absent(*texts: str) -> None:
    for text in texts:
        assert TOKEN not in text and TOKEN_B64 not in text


def files_text(root: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in root.rglob("*") if p.is_file())


def github_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeNetworkGit) -> SandboxWorkspace:
    monkeypatch.setattr(release_module, "_run_process", fake)
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
        keys = git_config(env)
        assert keys[f"http.{GITHUB_URL}.extraHeader"] == header and keys["credential.helper"] == ""
        assert keys["http.followRedirects"] == "false"  # the header never follows a redirect to another host
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull and env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert all(header not in env.values() for env in fake.local_envs())  # commits and trees never see it
    names = {name for _, env in fake.calls for name in env if not name.startswith(("GIT_CONFIG_", "GIT_INDEX_FILE"))}
    assert names <= release_module.GIT_ENV_ALLOWLIST | set(release_module.GIT_FIXED_ENV)  # no API key reaches git
    assert {env["HOME"] for _, env in fake.calls} == {os.devnull}  # no ~/.netrc, no ~/.config/git
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
    monkeypatch.setattr(release_module, "_run_process", fake)
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


# ---- task 2: observable push state, commit links, gate trailers ------------------------------------------------------
def trailers(remote: Path, commit: str) -> set[str]:
    return set(git("--git-dir", str(remote), "log", "-1", "--format=%(trailers:unfold,only)", commit).splitlines())


def test_the_release_commit_carries_the_gate_evidence_as_trailers(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    diff = ship(sandbox, thread_id="thr-release-1", applied_rules=["RULE-001", "RULE-003"])
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(diff, "Fix VAT")["release"])
    found = trailers(sandbox.settings.default_release_remote_dir, release.commit_sha)
    assert {
        f"Diff-SHA256: {diff}",
        "Quiz-Attempts: 1",
        "Approved-By: human",
        "Rules: RULE-001, RULE-003",
        "Thread: thr-release-1",
    } <= found


def test_an_unbound_thread_gets_no_thread_trailer(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Fix VAT")["release"])
    found = trailers(sandbox.settings.default_release_remote_dir, release.commit_sha)
    assert "Rules: RULE-001" in found and not [line for line in found if line.startswith("Thread:")]
    assert release.commit_url is None  # a local remote has no web page


def test_the_push_rejected_fault_refuses_one_push_then_ships_a_single_release_commit(tmp_path: Path) -> None:
    sandbox = SandboxWorkspace(RuntimeSettings(workspace_dir=tmp_path / "workspace"))
    sandbox.reset(fault="push_rejected")
    remote = sandbox.settings.default_release_remote_dir
    diff = ship(sandbox)
    seed_prod = sandbox.snapshot_final_state().prod_tree_sha256

    error = refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT", idempotency_key="deploy-1"))
    assert (error.code, error.retryable, error.outcome_unknown) == ("upstream_error", True, False)
    assert error.details["fault"] == "push_rejected"
    assert tip(remote) is None  # nothing reached the remote
    before = sandbox.snapshot_final_state()
    assert before.prod_tree_sha256 == seed_prod and not before.deployed and not before.pushed
    gate = sandbox.get_gate_status(diff)
    assert gate.missing_locks == ["human_approved"] and gate.ready_for_approval  # failed push revokes Lock 4
    sandbox.apply_deploy_decision(DeployDecision(action="approve", expected_diff_sha256=diff))

    payload = sandbox.deploy_to_production(diff, "Fix VAT", idempotency_key="deploy-1")  # re-approved retry
    release = ReleaseCommit.model_validate(payload["release"])
    assert release.pushed and tip(remote) == release.commit_sha
    subjects = git("--git-dir", str(remote), "log", "--format=%s", release.commit_sha).splitlines()
    assert [subject for subject in subjects if subject.startswith("deploy:")] == ["deploy: Fix VAT"]
    assert len(subjects) == 2  # the seed baseline, then the one release commit
    final = sandbox.snapshot_final_state()
    assert final.deployed and final.pushed and final.release_commit_sha == release.commit_sha


def test_pushed_is_read_from_the_local_remote_and_scored_by_expected_state(tmp_path: Path) -> None:
    sandbox = workspace_for(tmp_path)
    assert not sandbox.snapshot_final_state().pushed  # nothing deployed yet
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Fix VAT")["release"])
    final = sandbox.snapshot_final_state()
    assert final.deployed and final.pushed and final.release_commit_sha == release.commit_sha
    assert ExpectedState(deployed=True, pushed=True).mismatches(final) == {}
    # someone rewrites the remote branch behind the gate's back (test-side, no push): the release is no longer on it
    remote = sandbox.settings.default_release_remote_dir
    git("--git-dir", str(remote), "update-ref", f"refs/heads/{DEFAULT_GIT_BRANCH}", release.parent_sha or "")
    final = sandbox.snapshot_final_state()
    assert final.deployed and not final.pushed
    assert ExpectedState(pushed=True).mismatches(final) == {"pushed": (True, False)}


def test_github_release_links_its_commit_and_pushed_uses_the_deploy_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNetworkGit()
    sandbox = github_workspace(tmp_path, monkeypatch, fake)
    payload = sandbox.deploy_to_production(ship(sandbox), "Fix VAT")
    release = ReleaseCommit.model_validate(payload["release"])
    url = f"https://github.com/example-owner/example-release/commit/{release.commit_sha}"
    assert release.commit_url == payload["commit_url"] == url
    entry = sandbox.get_latest_ship_log()
    assert entry is not None and entry.release_commit_url == url
    calls_before = len(fake.network_envs())
    assert sandbox.snapshot_final_state().pushed
    assert len(fake.network_envs()) == calls_before  # judged on the deploy record: no network call
    (sandbox.prod_dir / "orders.py").write_text("# drifted after the deploy\n")
    assert not sandbox.snapshot_final_state().pushed  # prod no longer holds the pushed tree


@pytest.mark.parametrize(
    ("url", "repo"),
    [
        ("https://github.com/acme/release.git", "acme/release"),
        ("https://GitHub.com/acme/release/", "acme/release"),
        ("https://github.com/acme/release/extra", None),
        ("https://gitlab.com/acme/release.git", None),
        (f"https://x:{TOKEN}@github.com/acme/release.git", None),
        ("/srv/git/release.git", None),
        (None, None),
    ],
)
def test_github_repo_is_parsed_only_from_a_clean_github_url(url: str | None, repo: str | None) -> None:
    assert release_module.github_repo(url) == repo
    expected = f"https://github.com/{repo}/commit/{'a' * 40}" if repo else None
    assert release_module.github_commit_url(url, "a" * 40) == expected


# ---- task 2: `no-quiz-no-prod init-github [--verify]` ---------------------------------------------------------------
def init_github(settings: RuntimeSettings, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    args: argparse.Namespace = build_parser().parse_args(["init-github", *argv])
    return _cmd_init_github(args, settings=settings, out=out), out.getvalue()


def test_init_github_pushes_the_seed_once_and_deploys_build_on_it(tmp_path: Path) -> None:
    remote = bare_repo(tmp_path / "team_release.git")
    settings = RuntimeSettings(workspace_dir=tmp_path / "workspace", git_remote_url=str(remote))
    code, before = init_github(settings, "--verify")
    assert code == 0 and "head   : absent" in before and "✓ git" in before
    code, created = init_github(settings)
    seed_commit = tip(remote)
    assert code == 0 and seed_commit is not None and f"pushed the fintech seed as {seed_commit}" in created
    assert git("--git-dir", str(remote), "rev-list", "--count", seed_commit) == "1"
    code, again = init_github(settings)
    assert code == 0 and "left untouched" in again and tip(remote) == seed_commit  # never re-pushed, never forced
    code, verified = init_github(settings, "--verify")
    assert code == 0 and f"head   : {seed_commit}" in verified and f"branch : {DEFAULT_GIT_BRANCH}" in verified

    sandbox = SandboxWorkspace(settings)
    sandbox.reset()
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Fix VAT")["release"])
    assert release.parent_sha == seed_commit  # the seed already matches prod: no baseline commit in between


def test_init_github_verify_checks_the_token_and_never_prints_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "c" * 40
    fake = FakeNetworkGit(ls_remote=f"{head}\trefs/heads/{DEFAULT_GIT_BRANCH}\n")
    monkeypatch.setattr(release_module, "_run_process", fake)
    no_token = RuntimeSettings(workspace_dir=tmp_path / "workspace", git_remote_url=GITHUB_URL)
    code, missing = init_github(no_token, "--verify")
    assert code == 1 and "NO_QUIZ_GITHUB_TOKEN is not set" in missing
    assert not fake.network_envs()  # no access attempt without a token

    settings = no_token.model_copy(update={"github_token": SecretStr(TOKEN)})
    code, verified = init_github(settings, "--verify")
    assert code == 0 and "example-owner/example-release" in verified and f"head   : {head}" in verified
    assert "NO_QUIZ_GITHUB_TOKEN : configured" in verified
    leaky = settings.model_copy(update={"git_remote_url": f"https://x:{TOKEN}@github.com/example-owner/r.git"})
    code, refused = init_github(leaky, "--verify")
    assert code == 1 and "must not embed credentials" in refused
    assert_token_absent(missing, verified, refused)


# ---- the token: allow-listed hosts only ---------------------------------------------
def test_the_token_is_never_sent_to_a_host_outside_the_allow_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNetworkGit()
    monkeypatch.setattr(release_module, "_run_process", fake)
    sandbox = workspace_for(tmp_path, git_remote_url=OTHER_HOST_URL, github_token=SecretStr(TOKEN))
    diff = ship(sandbox)
    error = refusal(lambda: sandbox.deploy_to_production(diff, "Fix VAT"))
    assert error.code == "invalid_argument" and error.details["allowed_hosts"] == ["github.com"]
    assert "attacker" not in error.model_dump_json()  # the refused URL is never echoed
    assert_token_absent(error.model_dump_json())
    assert len(fake.calls) == 0  # git never even started (a count only: never print a recorded environment)
    assert not sandbox.snapshot_final_state().prod_changed
    assert refusal(lambda: release_module.validate_release_config(sandbox.settings)).code == "invalid_argument"
    for url in (
        "https://github.com:8443/acme/r.git",
        "https://github.com.attacker.example/acme/r.git",
        "https://github.com/acme/r.git\nhttp.evil.extraHeader=x",
        "file:///tmp/repo.git?x=1",
    ):
        other = RuntimeSettings(workspace_dir=tmp_path / "w", git_remote_url=url, github_token=SecretStr(TOKEN))
        assert refusal(release_module.GitReleasePublisher(other).resolve_remote).code == "invalid_argument"
    # defence in depth: even a remote object built around the check gets no header
    rogue = release_module.ReleaseRemote(target=OTHER_HOST_URL, display=OTHER_HOST_URL, kind="https")
    assert refusal(lambda: sandbox.release._environment(rogue, None)).code == "invalid_argument"


def test_an_extra_allowed_host_gets_the_token_scoped_to_its_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNetworkGit()
    monkeypatch.setattr(release_module, "_run_process", fake)
    hosts = ("github.com", "git.attacker.example")
    sandbox = workspace_for(
        tmp_path, git_remote_url=OTHER_HOST_URL, github_token=SecretStr(TOKEN), git_allowed_hosts=hosts
    )
    release = ReleaseCommit.model_validate(sandbox.deploy_to_production(ship(sandbox), "Fix VAT")["release"])
    assert release.pushed and release.commit_url is None  # not github.com: no web link is invented
    header = f"Authorization: Basic {TOKEN_B64}"
    assert all(git_config(env)[f"http.{OTHER_HOST_URL}.extraHeader"] == header for env in fake.network_envs())


def test_validate_release_config_reports_each_problem_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = RuntimeSettings(workspace_dir=tmp_path / "workspace")
    config = release_module.validate_release_config(base)
    assert (config.remote.kind, config.remote.create, config.branch) == ("local", True, DEFAULT_GIT_BRANCH)
    assert tuple(int(part) for part in config.git_version.split(".")[:2]) >= release_module.GIT_MIN_VERSION
    assert not base.default_release_remote_dir.exists() and not base.release_git_dir.exists()  # nothing was written

    def problem(**settings: Any) -> ToolErrorPayload:
        return refusal(lambda: release_module.validate_release_config(base.model_copy(update=settings)))

    no_token = problem(git_remote_url=GITHUB_URL)
    assert no_token.code == "invalid_argument" and "NO_QUIZ_GITHUB_TOKEN" in no_token.message
    assert problem(git_branch="main..x").code == "invalid_argument"
    assert problem(git_remote_url="relative/path.git").code == "invalid_argument"
    assert problem(git_remote_url=OTHER_HOST_URL, github_token=SecretStr(TOKEN)).code == "invalid_argument"
    ok = release_module.validate_release_config(
        base.model_copy(update={"git_remote_url": GITHUB_URL, "github_token": SecretStr(TOKEN)})
    )
    assert ok.remote.kind == "https"
    monkeypatch.setattr(release_module, "GIT_MIN_VERSION", (99, 0))
    assert problem().code == "internal_error"
