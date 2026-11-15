"""GitReleasePublisher: a production deploy is a real git commit, pushed fast-forward to the release remote.

Who calls it: ``SandboxWorkspace.deploy_to_production``, once the four locks hold. No agent toolset holds
``deploy_to_production`` and there is no git tool. That describes the tool surface only: nothing in this module can
stop code running in the same process, or as the same OS user, from importing it. Keeping agent-written test code
away from the publisher is the job of the test sandbox; what this module guarantees is listed under "Secrets".

How one release is built, with the git CLI (no Python dependency):

1. ``prepare`` (local, under the workspace lock). Staging and prod are each copied to a private snapshot whose
   ``compute_tree_sha256`` must equal the hash the gate approved. The git tree is built from that snapshot file by
   file (``hash-object --no-filters`` and an index written entry by entry), so no ignore file, attributes file or
   filter, the user's or the tree's own, can add, drop or rewrite a file. The tree is then read back from git and
   hashed again: it must still be the approved SHA-256.
2. ``publish`` (network, outside the lock). ``ls-remote`` + ``fetch`` read the tip of the release branch (no remote
   or tracking ref is ever configured), then exactly one of:
   - the tip already is this release (same ``Diff-SHA256`` and ``Thread`` trailers, same tree): a retried deploy,
     nothing is committed or pushed again (``pushed=False``). An equal tree alone never counts;
   - the tip does not hold the prod tree (someone else pushed, or the workspace was reset): ``remote_diverged``.
     Nothing is pushed and no "sync" commit is ever written on the human's behalf;
   - the branch does not exist: it is initialised with a baseline commit of the prod tree;
   - the tip holds the prod tree: the release commit goes on top of it.
3. The release commit is pushed as ``<commit>:refs/heads/<branch>``: no ``+``, no ``--force``. A remote that moved
   meanwhile rejects it (``conflict``) and nothing is overwritten. Its trailers record the gate evidence
   (``Diff-SHA256``, ``Quiz-Attempts``, ``Approved-By: human``, ``Rules``, ``Thread``).

One whole ``publish`` is bounded by ``RELEASE_DEADLINE_S``; a git command that outlives its budget is killed with
its whole process group.

``initialize_branch`` (``no-quiz-no-prod init-github``) pushes the seed as the first commit of a branch that does not
exist yet, and ``inspect_remote`` (``init-github --verify``) only reads the advertised tip.

Secrets: the GitHub token is read from ``settings.github_token`` only; this module never reads or writes
``os.environ`` for it. It only ever reaches git as an ``http.<remote>.extraHeader`` set through ``GIT_CONFIG_*``
environment variables of the network commands, and only for an ``https`` remote whose host is in
``settings.git_allowed_hosts``; redirects are not followed. It is never in argv, a remote URL, a git config file, a
result, a log or an error (git's output is redacted anyway). Known limit: while ``ls-remote``, ``fetch`` or ``push``
runs, the header sits in the launch environment of that git process, which another process of the same OS user can
read on some systems; the gate therefore never runs staged tests during a deploy. The user's global and system git
config and home directory are ignored, so no credential helper or ``.netrc`` is used and no hook, signing key or
rewrite rule interferes.
"""

import base64
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlsplit

from no_quiz_no_prod.backend.code_tree import copy_tree, tree_files
from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import (
    GIT_LOCAL_TIMEOUT_S,
    GIT_NETWORK_TIMEOUT_S,
    RELEASE_DEADLINE_S,
    ReleaseCommit,
    ReleaseRemoteKind,
    RuntimeSettings,
    compute_tree_sha256,
    diff_sha256_from_trees,
)

GIT_EXECUTABLE: Final = "git"
GIT_MIN_VERSION: Final = (2, 31)  # GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n
GIT_OUTPUT_TAIL_CHARS: Final = 600
GATE_AUTHOR_NAME: Final = "No-Quiz-No-Prod Gate"
GATE_AUTHOR_EMAIL: Final = "gate@no-quiz-no-prod.invalid"
GITHUB_TOKEN_USER: Final = "x-access-token"  # GitHub's HTTPS basic-auth user for a token
SNAPSHOT_DIR_PREFIX: Final = "snapshot-"  # private copies of a tree, inside the commit cache
REGULAR_FILE_MODE: Final = "100644"  # every released file: the tree hash covers paths and bytes, not modes
EXPORTABLE_FILE_MODES: Final = frozenset({REGULAR_FILE_MODE, "100755"})
HTTPS_DEFAULT_PORT: Final = 443
REDACTED: Final = "***"
TRAILER_DIFF: Final = "Diff-SHA256"
TRAILER_THREAD: Final = "Thread"
TRAILER_VALUE_MAX_CHARS: Final = 200
BASELINE_COMMIT_MESSAGE: Final = (
    "chore(release): initialize the release branch with production\n\n"
    "Baseline of the production tree before the first gated deploy.\n"
)
_BRANCH_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_REJECTED_PUSH: Final = ("[rejected]", "non-fast-forward", "fetch first", "stale info", "! [remote rejected]")
_GIT_VERSION: Final = re.compile(r"(\d+)\.(\d+)")
GITHUB_HOST: Final = "github.com"
_GITHUB_NAME: Final = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
# The only inherited variables git gets (least privilege: no API key of the parent process reaches it). HOME is not
# one of them: git gets an unusable one, so no ~/.netrc, ~/.config/git/ignore or ~/.config/git/attributes applies.
GIT_ENV_ALLOWLIST: Final = frozenset(
    {
        "PATH",
        "TMPDIR",
        "SYSTEMROOT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
    }
)
GIT_FIXED_ENV: Final[Mapping[str, str]] = {
    "HOME": os.devnull,
    "XDG_CONFIG_HOME": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_TRACE_REDACT": "1",
    "GIT_AUTHOR_NAME": GATE_AUTHOR_NAME,
    "GIT_AUTHOR_EMAIL": GATE_AUTHOR_EMAIL,
    "GIT_COMMITTER_NAME": GATE_AUTHOR_NAME,
    "GIT_COMMITTER_EMAIL": GATE_AUTHOR_EMAIL,
    "LC_ALL": "C",
    "LANG": "C",
}

# Process-wide git configuration of every gate command (passed through the environment, never written to a file).
_BASE_GIT_CONFIG: Final[tuple[tuple[str, str], ...]] = (
    ("credential.helper", ""),  # no helper may answer with, or store, a credential
    ("core.hooksPath", os.devnull),  # no repository or template hook runs
    ("commit.gpgSign", "false"),
    ("core.autocrlf", "false"),  # committed bytes == hashed bytes
    ("http.followRedirects", "false"),  # the token header never follows a redirect to another host
    ("protocol.allow", "never"),
    ("protocol.file.allow", "always"),
    ("protocol.https.allow", "always"),
)


@dataclass(frozen=True)
class ReleaseRemote:
    """Where releases are pushed: the git argument, its credential-free display form, and its kind."""

    target: str
    display: str
    kind: ReleaseRemoteKind
    create: bool = False  # the default workspace remote: initialized as a bare repository on first use


@dataclass(frozen=True)
class RemoteHead:
    """What ``inspect_remote`` / ``initialize_branch`` saw: the remote, the branch and its tip (None: no branch)."""

    remote: ReleaseRemote
    branch: str
    tip: str | None
    git_version: str
    created: bool = False  # initialize_branch pushed the first commit of the branch
    remote_exists: bool = True  # False for the default local remote before its first use


@dataclass(frozen=True)
class ReleaseConfig:
    """A release configuration that passed ``validate_release_config``."""

    remote: ReleaseRemote
    branch: str
    git_version: str


@dataclass(frozen=True)
class PreparedRelease:
    """A release built locally by ``prepare`` and not published yet: verified git trees and the commit message."""

    diff_sha256: str
    approved_tree_sha256: str  # the tree hash prod has once promoted (the staging tree that passed the 4 locks)
    target_tree: str  # git tree of the approved staging snapshot
    baseline_tree: str  # git tree of the prod snapshot the change was approved against
    message: str
    thread_id: str | None


# ---- remote description (pure) -------------------------------------------------------------------------------------
def github_repo(url: str | None) -> str | None:
    """``<owner>/<repo>`` of a credential-free ``https://github.com/<owner>/<repo>[.git]`` URL, else None."""
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme != "https" or "@" in parts.netloc or (parts.hostname or "").lower() != GITHUB_HOST:
        return None
    if parts.port not in (None, HTTPS_DEFAULT_PORT) or parts.query or parts.fragment:
        return None
    segments = parts.path.strip("/").split("/")
    if len(segments) != 2:
        return None
    owner, repo = segments[0], segments[1].removesuffix(".git")
    if not (_GITHUB_NAME.match(owner) and _GITHUB_NAME.match(repo)):
        return None
    return f"{owner}/{repo}"


def github_commit_url(remote_url: str | None, commit_sha: str) -> str | None:
    """``https://github.com/<owner>/<repo>/commit/<sha>`` for a github.com remote, else None."""
    repo = github_repo(remote_url)
    return f"https://{GITHUB_HOST}/{repo}/commit/{commit_sha}" if repo else None


def describe_release_remote(settings: RuntimeSettings) -> dict[str, Any]:
    """The Studio's secret-free view of the release remote: kind, GitHub repo, branch and whether a token is set.

    Pure (no git, no network): a local path or a misconfigured URL is never echoed.
    """
    url = settings.git_remote_url
    kind: ReleaseRemoteKind = "https" if url and urlsplit(url).scheme == "https" else "local"
    return {
        "remote_kind": kind,
        "repo": github_repo(url),
        "branch": settings.git_branch,
        "token_set": settings.github_token is not None,
    }


def _token_may_go_to(url: str, settings: RuntimeSettings) -> bool:
    """True when ``url`` is an ``https`` URL on the default port of a host in ``settings.git_allowed_hosts``."""
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:  # a port that is not a number
        return False
    allowed = {host.lower() for host in settings.git_allowed_hosts}
    return parts.scheme == "https" and (parts.hostname or "").lower() in allowed and port in (None, HTTPS_DEFAULT_PORT)


def _trailer_value(text: object) -> str:
    """One-line trailer value: whitespace (newlines included) collapsed, bounded."""
    return " ".join(str(text).split())[:TRAILER_VALUE_MAX_CHARS] or "none"


def _trailers(message: str) -> dict[str, str]:
    """``{key: value}`` of the last paragraph of a commit message: the trailer block ``prepare`` always writes last.

    Parsed here rather than by git, so text in the release note above it can never pass for a trailer.
    """
    trimmed = message.rstrip("\n")
    if "\n\n" not in trimmed:
        return {}
    block = trimmed.rsplit("\n\n", 1)[-1]
    return dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)


def _run_process(
    argv: Sequence[str], *, stdin: bytes | None, timeout: float, env: Mapping[str, str], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    """Run one command in its own process group; on timeout the whole group is killed before re-raising."""
    with subprocess.Popen(  # fixed executable, no shell, token-free argv
        list(argv),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)  # git's helpers (git-remote-https) included
            except ProcessLookupError:
                pass  # the group is already gone
            process.communicate()
            raise
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


class GitReleasePublisher:
    """Builds and pushes the release commit of one deploy (see the module docstring)."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.git_dir = settings.release_git_dir

    # ---- public API ----------------------------------------------------------------------------------------------
    def validate(self) -> ReleaseConfig:
        """See ``validate_release_config``."""
        remote = self.resolve_remote()  # pure validation first: a bad URL never reaches git
        if remote.kind == "https" and self.settings.github_token is None:
            raise WorkspaceError(
                "invalid_argument", "NO_QUIZ_GITHUB_TOKEN is not set: a push to an https:// release remote needs it."
            )
        branch = self.settings.git_branch
        if not _BRANCH_NAME.match(branch) or ".." in branch or branch.endswith((".lock", "/", ".")) or "//" in branch:
            raise WorkspaceError("invalid_argument", f"NO_QUIZ_GIT_BRANCH {branch!r} is not a valid branch name.")
        version = self._check_git_version()
        self._git("check-ref-format", f"refs/heads/{branch}", in_repo=False)
        return ReleaseConfig(remote=remote, branch=branch, git_version=version)

    def prepare(
        self,
        *,
        source_dir: Path,
        baseline_dir: Path,
        diff_sha256: str,
        approved_tree_sha256: str,
        baseline_tree_sha256: str,
        release_note: str,
        quiz_attempts: int = 0,
        applied_rules: Sequence[str] = (),
        thread_id: str | None = None,
    ) -> PreparedRelease:
        """Build, locally, the release of ``source_dir`` (staging) on top of ``baseline_dir`` (prod).

        No network: call it under the workspace lock. ``approved_tree_sha256`` and ``baseline_tree_sha256`` are the
        ``compute_tree_sha256`` of staging and prod that the gate approved as ``diff_sha256``; a directory that no
        longer hashes to its value is refused (``conflict``), and so is a git tree that would not hold exactly the
        hashed files (``internal_error``). ``quiz_attempts``, ``applied_rules`` and ``thread_id`` (when known)
        become commit trailers next to ``Diff-SHA256``.
        """
        if diff_sha256 != diff_sha256_from_trees(baseline_tree_sha256, approved_tree_sha256):
            raise WorkspaceError("invalid_argument", "diff_sha256 is not the diff of the baseline and approved trees")
        self.validate()
        self._ensure_git_dir()
        trailers = {
            TRAILER_DIFF: diff_sha256,
            "Prod-Tree-SHA256": approved_tree_sha256,
            "Gate": "tests_passed, sha256_matches, quiz_passed, human_approved",
            "Quiz-Attempts": str(quiz_attempts),
            "Approved-By": "human",
            "Rules": ", ".join(applied_rules) or "none",
            **({TRAILER_THREAD: thread_id} if thread_id else {}),
        }
        message = f"deploy: {self._subject(release_note)}\n\n{release_note.strip()}\n\n" + "".join(
            f"{key}: {_trailer_value(value)}\n" for key, value in trailers.items()
        )
        return PreparedRelease(
            diff_sha256=diff_sha256,
            approved_tree_sha256=approved_tree_sha256,
            target_tree=self._verified_tree(source_dir, approved_tree_sha256, "staging"),
            baseline_tree=self._verified_tree(baseline_dir, baseline_tree_sha256, "production"),
            message=message,
            thread_id=thread_id,
        )

    def publish(
        self,
        prepared: PreparedRelease,
        *,
        before_push: Callable[[ReleaseCommit], None] | None = None,
        reject_push: Callable[[], bool] | None = None,
    ) -> ReleaseCommit:
        """Push ``prepared`` to the release branch (network: call it WITHOUT the workspace lock).

        Returns the release; ``pushed`` is False when the branch tip already is this release (a retried deploy).
        Raises ``WorkspaceError``: ``remote_diverged`` when the tip does not hold the prod tree (nothing is pushed,
        a human must reconcile), ``conflict`` when the remote moved during the deploy, ``upstream_error`` for a
        refused or failed push, ``timeout`` with ``outcome_unknown`` when git did not answer in time,
        ``invalid_argument`` for a bad remote or branch.

        ``before_push`` receives the commit about to be pushed (the gate journals its intent there).
        ``reject_push`` is the eval's ``push_rejected`` fault hook: asked once the release commit is built, a True
        answer refuses that push as a retryable ``upstream_error`` before anything reaches the remote.
        """
        deadline = time.monotonic() + RELEASE_DEADLINE_S
        config = self.validate()
        remote, branch = config.remote, config.branch
        self._ensure_git_dir()
        if remote.create:
            self._ensure_bare_remote(Path(remote.target))
        tip = self._remote_tip(remote, branch, deadline)
        if tip is not None and self._is_same_release(tip, prepared):
            return self._release(tip, self._first_parent(tip), config, prepared, pushed=False)
        if tip is None:
            parent = self._commit_tree(prepared.baseline_tree, None, BASELINE_COMMIT_MESSAGE)
        elif self._tree_of(tip) == prepared.baseline_tree:
            parent = tip
        else:
            raise WorkspaceError(
                "remote_diverged",
                f"The release branch {branch!r} does not hold the current production tree: someone else pushed to "
                "it, or the workspace was reset. Nothing was pushed and production was not touched. A human must "
                "reconcile them: bring the branch back to the production tree, or release to a new branch "
                "(NO_QUIZ_GIT_BRANCH), which is initialised from production. No sync commit is ever pushed for you.",
                details={"branch": branch, "remote": remote.display, "remote_tip": tip},
            )
        commit = self._commit_tree(prepared.target_tree, parent, prepared.message)
        release = self._release(commit, parent, config, prepared, pushed=True)
        if before_push is not None:
            before_push(release)
        if reject_push is not None and reject_push():
            raise WorkspaceError(
                "upstream_error",
                f"The release remote rejected the push to {branch!r} (injected push_rejected fault). Nothing was "
                "pushed and production was not touched: repeating the same deploy is safe.",
                retryable=True,
                details={"branch": branch, "remote": remote.display, "fault": "push_rejected"},
            )
        self._push(remote, commit, branch, deadline)
        self._git("update-ref", f"refs/heads/{branch}", commit)  # local record of the last release
        return release

    def export_tree(self, commit_sha: str, destination: Path) -> None:
        """Write the files of a commit (or tree) of the commit cache into the empty directory ``destination``.

        Raw blobs, no checkout: no filter, attribute or end-of-line conversion applies. The gate promotes prod from
        the release commit with it, in the normal path and when it completes a journaled release after a restart.
        """
        listing = self._git("ls-tree", "-r", "-z", "--full-tree", commit_sha)
        for record in filter(None, listing.split("\0")):
            meta, relative = record.split("\t", 1)
            mode, kind, blob = meta.split()
            path = PurePosixPath(relative)
            if kind != "blob" or mode not in EXPORTABLE_FILE_MODES or path.is_absolute() or ".." in path.parts:
                raise WorkspaceError(
                    "internal_error", f"commit {commit_sha} holds an entry the gate never writes ({mode} {kind})"
                )
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._git_bytes("cat-file", "blob", blob))

    def remote_has_commit(self, commit_sha: str) -> bool:
        """True when the release branch of the remote contains ``commit_sha`` (network: ``ls-remote`` + ``fetch``).

        For reconciliation after a restart: did the push of a journaled release land? A network failure raises.
        """
        deadline = time.monotonic() + RELEASE_DEADLINE_S
        config = self.validate()
        if config.remote.create and not (Path(config.remote.target) / "HEAD").is_file():
            return False
        self._ensure_git_dir()
        tip = self._remote_tip(config.remote, config.branch, deadline)
        return tip is not None and (tip == commit_sha or self._is_ancestor(commit_sha, tip))

    def discard_default_remote(self) -> None:
        """Delete the workspace's own bare remote (a workspace reset); a configured remote is never touched.

        Without it every deploy after a reset would be ``remote_diverged``: the old releases would no longer match
        the reseeded prod. The next deploy initialises the branch again.
        """
        if self.settings.git_remote_url is None:
            shutil.rmtree(self.settings.default_release_remote_dir, ignore_errors=True)

    def is_released(self, release: ReleaseCommit, prod_dir: Path) -> bool:
        """True when ``release`` is on its remote branch and holds the current tree of ``prod_dir``.

        An ``https`` remote is judged on what the deploy recorded (the push succeeded, and prod still has the tree it
        pushed): no network call. A local remote is read for real: the commit must be reachable from the branch tip.
        """
        if not prod_dir.is_dir() or release.prod_tree_sha256 != compute_tree_sha256(prod_dir):
            return False
        if release.remote_kind == "https":
            return True
        remote = ReleaseRemote(target=release.remote, display=release.remote, kind="local")
        try:
            if not Path(remote.target).exists():
                return False
            self._ensure_git_dir()
            tip = self._remote_tip(remote, release.branch, time.monotonic() + RELEASE_DEADLINE_S)
            return tip is not None and self._is_ancestor(release.commit_sha, tip)
        except WorkspaceError:
            return False

    def git_version(self) -> str:
        """The installed git version; ``internal_error`` when git is missing or older than ``GIT_MIN_VERSION``."""
        return self._check_git_version()

    def inspect_remote(self) -> RemoteHead:
        """Read-only check of the configuration and of the remote (``init-github --verify``): git version, remote
        URL, branch name, then ``ls-remote`` (access + tip). Nothing is fetched, committed or pushed."""
        deadline = time.monotonic() + RELEASE_DEADLINE_S
        config = self.validate()
        remote, branch, version = config.remote, config.branch, config.git_version
        if remote.create and not (Path(remote.target) / "HEAD").is_file():
            return RemoteHead(remote=remote, branch=branch, tip=None, git_version=version, remote_exists=False)
        return RemoteHead(
            remote=remote, branch=branch, tip=self._advertised_tip(remote, branch, deadline), git_version=version
        )

    def initialize_branch(self, source_dir: Path, *, message: str) -> RemoteHead:
        """Push ``source_dir`` as the root commit of the release branch if the remote does not have it yet.

        An existing branch is left exactly as it is (``created=False``); a new one is created by a plain
        fast-forward push (``conflict`` if someone created it meanwhile, never forced).
        """
        deadline = time.monotonic() + RELEASE_DEADLINE_S
        config = self.validate()
        remote, branch, version = config.remote, config.branch, config.git_version
        self._ensure_git_dir()
        if remote.create:
            self._ensure_bare_remote(Path(remote.target))
        tip = self._advertised_tip(remote, branch, deadline)
        if tip is not None:
            return RemoteHead(remote=remote, branch=branch, tip=tip, git_version=version)
        commit = self._commit_tree(self._write_tree(source_dir), None, message)
        self._push(remote, commit, branch, deadline)
        self._git("update-ref", f"refs/heads/{branch}", commit)
        return RemoteHead(remote=remote, branch=branch, tip=commit, git_version=version, created=True)

    def resolve_remote(self) -> ReleaseRemote:
        """The configured remote, validated: an ``https://`` URL without credentials on an allowed host, or a local
        repository path. Pure: no git, no network.

        Never echoes the configured URL in an error, since a misconfigured one may embed a secret.
        """
        url = self.settings.git_remote_url
        if url is None:
            path = self.settings.default_release_remote_dir
            return ReleaseRemote(target=str(path), display=str(path), kind="local", create=True)
        if any(char.isspace() or ord(char) < 32 for char in url):
            raise WorkspaceError("invalid_argument", "NO_QUIZ_GIT_REMOTE_URL must not contain spaces or control bytes.")
        parts = urlsplit(url)
        if parts.scheme == "https":
            if "@" in parts.netloc or parts.username is not None or parts.password is not None:
                raise WorkspaceError(
                    "invalid_argument",
                    "NO_QUIZ_GIT_REMOTE_URL must not embed credentials: set the token in NO_QUIZ_GITHUB_TOKEN.",
                )
            if not parts.hostname or parts.query or parts.fragment:
                raise WorkspaceError("invalid_argument", "NO_QUIZ_GIT_REMOTE_URL is not a valid https:// git URL.")
            if not _token_may_go_to(url, self.settings):
                allowed = ", ".join(self.settings.git_allowed_hosts)
                raise WorkspaceError(
                    "invalid_argument",
                    "NO_QUIZ_GIT_REMOTE_URL points at a host the release token may not be sent to. Allowed hosts "
                    f"(default port only): {allowed}. Add a host you trust with NO_QUIZ_GIT_ALLOWED_HOSTS.",
                    details={"allowed_hosts": list(self.settings.git_allowed_hosts)},
                )
            return ReleaseRemote(target=url, display=url, kind="https")
        local = Path(parts.path) if parts.scheme == "file" and not parts.netloc else Path(url)
        if parts.scheme not in ("", "file") or parts.query or parts.fragment or not local.is_absolute():
            raise WorkspaceError(
                "invalid_argument",
                "NO_QUIZ_GIT_REMOTE_URL must be an https:// URL or an absolute path to a local git repository.",
            )
        return ReleaseRemote(target=str(local), display=str(local), kind="local")

    # ---- git steps -----------------------------------------------------------------------------------------------
    def _check_git_version(self) -> str:
        """The installed git version (e.g. "2.50.1"); ``internal_error`` below ``GIT_MIN_VERSION``."""
        output = self._git("version", in_repo=False).strip()
        found = _GIT_VERSION.search(output)
        if found is None or (int(found[1]), int(found[2])) < GIT_MIN_VERSION:
            required = ".".join(map(str, GIT_MIN_VERSION))
            raise WorkspaceError("internal_error", f"the release gate needs git >= {required} (found {output!r})")
        return output.removeprefix("git version ").strip()

    def _ensure_git_dir(self) -> None:
        if not (self.git_dir / "HEAD").is_file():
            shutil.rmtree(self.git_dir, ignore_errors=True)
            self._git("init", "--quiet", "--bare", str(self.git_dir), in_repo=False)

    def _ensure_bare_remote(self, path: Path) -> None:
        if not (path / "HEAD").is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._git("init", "--quiet", "--bare", str(path), in_repo=False)

    def _advertised_tip(self, remote: ReleaseRemote, branch: str, deadline: float | None = None) -> str | None:
        """The commit id the remote advertises for the branch (``ls-remote``), or None if the branch is new."""
        ref = f"refs/heads/{branch}"
        listing = self._git("ls-remote", "--heads", remote.target, ref, remote=remote, deadline=deadline)
        tips = [line.split("\t", 1)[0] for line in listing.splitlines() if line.endswith(f"\t{ref}")]
        return tips[0] if tips else None

    def _remote_tip(self, remote: ReleaseRemote, branch: str, deadline: float) -> str | None:
        """The commit the remote branch points at, with its objects fetched, or None if the branch is new."""
        advertised = self._advertised_tip(remote, branch, deadline)
        if advertised is None:
            return None
        ref = f"refs/heads/{branch}"
        fetch = ("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", remote.target, ref)
        self._git(*fetch, remote=remote, deadline=deadline)
        # no FETCH_HEAD and no ref is written: the objects are now local, so the advertised tip resolves directly
        return self._git("rev-parse", "--verify", f"{advertised}^{{commit}}").strip()

    def _is_ancestor(self, commit: str, tip: str) -> bool:
        """True when ``commit`` is ``tip`` or one of its ancestors; False too when the cache does not know it."""
        try:
            self._git("cat-file", "-e", f"{commit}^{{commit}}")
            self._git("merge-base", "--is-ancestor", commit, tip)
        except WorkspaceError as error:
            if error.payload.details.get("exit_code") in (1, 128):  # "not an ancestor" / "no such commit"
                return False
            raise
        return True

    def _is_same_release(self, tip: str, prepared: PreparedRelease) -> bool:
        """True when the branch tip is ``prepared`` itself, already pushed: identity, never an equal tree alone."""
        _, _, message = self._git("cat-file", "commit", tip).partition("\n\n")
        trailers = _trailers(message)
        thread = _trailer_value(prepared.thread_id) if prepared.thread_id else None
        return (
            trailers.get(TRAILER_DIFF) == prepared.diff_sha256
            and trailers.get(TRAILER_THREAD) == thread
            and self._tree_of(tip) == prepared.target_tree
        )

    @contextmanager
    def _scratch_dir(self) -> Iterator[Path]:
        """A private directory inside the commit cache, removed on exit."""
        scratch = Path(tempfile.mkdtemp(prefix=SNAPSHOT_DIR_PREFIX, dir=self.git_dir))
        try:
            yield scratch
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _verified_tree(self, directory: Path, expected_sha256: str, what: str) -> str:
        """The git tree of exactly the files that make ``directory`` hash to ``expected_sha256``.

        Built from a private copy, so a write that lands in ``directory`` after the check cannot change the tree;
        then read back from git and hashed again, so nothing but the approved bytes can be pushed.
        """
        if any(entry.is_symlink() for entry in (directory, *directory.rglob("*"))):
            raise WorkspaceError("forbidden", f"the {what} tree contains a symbolic link: nothing was committed")
        with self._scratch_dir() as scratch:
            snapshot, readback = scratch / "snapshot", scratch / "readback"
            snapshot.mkdir()
            readback.mkdir()
            copy_tree(directory, snapshot)
            if compute_tree_sha256(snapshot) != expected_sha256:
                raise WorkspaceError(
                    "conflict",
                    f"The {what} tree changed since it was approved: nothing was committed or pushed. Review the "
                    "current diff and earn the locks again.",
                    retryable=False,
                    details={"tree": what, "expected_tree_sha256": expected_sha256},
                )
            tree = self._write_tree(snapshot)
            self.export_tree(tree, readback)
            if compute_tree_sha256(readback) != expected_sha256:
                raise WorkspaceError(
                    "internal_error", f"the git tree built for {what} does not hold exactly the approved files"
                )
            return tree

    def _write_tree(self, directory: Path) -> str:
        """The git tree of the files ``compute_tree_sha256(directory)`` hashes, as plain regular files.

        Plumbing only: each blob is stored unfiltered and listed in a throwaway index by name, so no ``.gitignore``,
        ``.gitattributes``, global ignore file or filter is consulted.
        """
        files = tree_files(directory)
        with self._scratch_dir() as scratch:
            env = {"GIT_INDEX_FILE": str(scratch / "index")}
            if files:
                paths = [str(path) for path in files.values()]
                blobs = self._git("hash-object", "-w", "--no-filters", "--", *paths).split()
                entries = "".join(
                    f"{REGULAR_FILE_MODE} {blob}\t{relative}\0" for relative, blob in zip(files, blobs, strict=True)
                )
                self._git("update-index", "-z", "--index-info", env=env, stdin=entries)
            return self._git("write-tree", env=env).strip()

    def _tree_of(self, commit: str) -> str:
        return self._git("rev-parse", "--verify", f"{commit}^{{tree}}").strip()

    def _first_parent(self, commit: str) -> str | None:
        parents = self._git("rev-list", "--parents", "-n", "1", commit).split()
        return parents[1] if len(parents) > 1 else None

    def _commit_tree(self, tree: str, parent: str | None, message: str) -> str:
        args = ["commit-tree", tree, *(["-p", parent] if parent else []), "-F", "-"]
        return self._git(*args, stdin=message).strip()

    def _push(self, remote: ReleaseRemote, commit: str, branch: str, deadline: float | None = None) -> None:
        """Fast-forward push of ``commit`` to ``branch``: the refspec has no ``+`` and no force flag exists here."""
        refspec = f"{commit}:refs/heads/{branch}"
        push = ("push", "--quiet", "--porcelain", "--no-verify", remote.target, refspec)
        try:
            self._git(*push, remote=remote, deadline=deadline)
        except WorkspaceError as error:
            output = str(error.payload.details.get("git_output", ""))
            if error.code == "upstream_error" and any(marker in output for marker in _REJECTED_PUSH):
                raise WorkspaceError(
                    "conflict",
                    f"The release branch {branch!r} moved during the deploy, so the push was refused (it is never "
                    "forced). Production was not touched: deploying again builds on the new tip.",
                    retryable=True,
                    details={"branch": branch, "remote": remote.display, "git_output": output},
                ) from None
            raise

    def _release(
        self, commit: str, parent: str | None, config: ReleaseConfig, prepared: PreparedRelease, *, pushed: bool
    ) -> ReleaseCommit:
        remote = config.remote
        return ReleaseCommit(
            commit_sha=commit,
            parent_sha=parent,
            branch=config.branch,
            remote=remote.display,
            remote_kind=remote.kind,
            diff_sha256=prepared.diff_sha256,
            prod_tree_sha256=prepared.approved_tree_sha256,
            pushed=pushed,
            commit_url=github_commit_url(remote.target if remote.kind == "https" else None, commit),
        )

    @staticmethod
    def _subject(release_note: str) -> str:
        first_line = next((line.strip() for line in release_note.splitlines() if line.strip()), "gated release")
        return first_line[:72]

    # ---- process plumbing ----------------------------------------------------------------------------------------
    def _secrets(self) -> list[str]:
        """Every form a configured token could take in git's output (raw and as its basic-auth header value)."""
        secret = self.settings.github_token
        token = secret.get_secret_value() if secret else ""
        return [secret for secret in (token, self._basic_auth(token) if token else "") if secret]

    @staticmethod
    def _basic_auth(token: str) -> str:
        return base64.b64encode(f"{GITHUB_TOKEN_USER}:{token}".encode()).decode("ascii")

    def _redact(self, text: str) -> str:
        for secret in self._secrets():
            text = text.replace(secret, REDACTED)
        return text[-GIT_OUTPUT_TAIL_CHARS:]

    def _environment(self, remote: ReleaseRemote | None, extra: Mapping[str, str] | None) -> dict[str, str]:
        """A clean git environment: only ``GIT_ENV_ALLOWLIST`` inherited, user/system config ignored, gate identity.

        The token header is added for network commands only, to an ``https`` remote on an allowed host only, and
        scoped to that remote's URL.
        """
        env = {name: value for name, value in os.environ.items() if name in GIT_ENV_ALLOWLIST} | dict(GIT_FIXED_ENV)
        config = list(_BASE_GIT_CONFIG)
        token = self.settings.github_token
        if remote is not None and remote.kind == "https" and token is not None:
            if not _token_may_go_to(remote.target, self.settings):
                raise WorkspaceError("invalid_argument", "the release token may not be sent to this remote's host")
            header = f"Authorization: Basic {self._basic_auth(token.get_secret_value())}"
            config.append((f"http.{remote.target}.extraHeader", header))
        env["GIT_CONFIG_COUNT"] = str(len(config))
        for index, (key, value) in enumerate(config):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env | dict(extra or {})

    def _git(self, *args: str, **options: Any) -> str:
        """``_git_bytes`` as text (file-system encoding: any path git prints survives the round trip)."""
        return os.fsdecode(self._git_bytes(*args, **options))

    def _git_bytes(
        self,
        *args: str,
        in_repo: bool = True,
        remote: ReleaseRemote | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        deadline: float | None = None,
    ) -> bytes:
        """Run one git command and return its stdout; any failure is a ``WorkspaceError`` with redacted output.

        ``in_repo`` runs it against the gate's commit cache; ``remote`` marks a network command (longer timeout, the
        token header when applicable); ``deadline`` (``time.monotonic()``) caps the timeout of this command.
        """
        prefix: Sequence[str] = [f"--git-dir={self.git_dir}"] if in_repo else []
        argv = [GIT_EXECUTABLE, *prefix, *args]
        timeout = GIT_NETWORK_TIMEOUT_S if remote is not None else GIT_LOCAL_TIMEOUT_S
        if deadline is not None:
            timeout = max(min(timeout, deadline - time.monotonic()), 0.0)
        workdir = self.settings.workspace_dir
        workdir.mkdir(parents=True, exist_ok=True)  # a missing cwd must not read as "git is not installed"
        try:
            completed = _run_process(
                argv,
                stdin=None if stdin is None else os.fsencode(stdin),
                timeout=timeout,
                env=self._environment(remote, env),
                cwd=workdir,
            )
        except FileNotFoundError:
            if shutil.which(GIT_EXECUTABLE) is not None:
                raise
            raise WorkspaceError("internal_error", "git is not installed: the release gate cannot publish") from None
        except subprocess.TimeoutExpired:
            raise WorkspaceError(
                "timeout",
                f"git {args[0]} did not answer within {timeout:.0f}s. The release MAY have been pushed: deploying "
                "again with the same idempotency key is safe (an already-pushed release is never pushed twice).",
                retryable=True,
                outcome_unknown=remote is not None,
            ) from None
        if completed.returncode != 0:
            raw = b"\n".join((completed.stderr, completed.stdout)).decode("utf-8", errors="replace")
            where = f" ({remote.display})" if remote is not None else ""
            raise WorkspaceError(
                "upstream_error" if remote is not None else "internal_error",
                f"git {args[0]}{where} failed with exit code {completed.returncode}.",
                details={
                    "git_command": args[0],
                    "exit_code": completed.returncode,
                    "git_output": self._redact(raw.strip()),
                },
            )
        return completed.stdout


def validate_release_config(settings: RuntimeSettings) -> ReleaseConfig:
    """The one check of the release configuration: harness, CLI and UI call it when a run starts.

    Verifies the remote URL (an ``https://`` URL without credentials on an allowed host, or an absolute local path),
    that an ``https`` remote has a token, the branch name, and git >= ``GIT_MIN_VERSION``. No network and no write
    beyond creating the workspace directory. Raises ``WorkspaceError``: ``invalid_argument`` for the configuration,
    ``internal_error`` for git. No message echoes the URL or the token.
    """
    return GitReleasePublisher(settings).validate()


__all__ = [
    "GitReleasePublisher",
    "PreparedRelease",
    "ReleaseConfig",
    "ReleaseRemote",
    "RemoteHead",
    "describe_release_remote",
    "github_commit_url",
    "github_repo",
    "validate_release_config",
]
