"""GitReleasePublisher: a production deploy is a real git commit, pushed fast-forward to the release remote.

Only ``SandboxWorkspace.deploy_to_production`` calls it, after the four locks hold and under the workspace lock, so
no agent can ever reach it (no agent toolset holds ``deploy_to_production``, and there is no git tool at all).

How one release is built, with the git CLI (no Python dependency):

1. ``ls-remote`` + ``fetch`` read the tip of the release branch (``FETCH_HEAD``: no remote or tracking ref is ever
   configured). A retried deploy whose tree the tip already holds stops here (``pushed=False``).
2. The pre-deploy prod tree and the staging tree are written as git trees (a throwaway index, the
   ``TREE_HASH_IGNORED_*`` patterns excluded), so each git tree holds exactly the files ``compute_tree_sha256`` hashes.
3. When the tip does not hold the prod tree (first release, a reset, a commit pushed by someone else), a baseline
   commit of the prod tree goes first, so the release commit's own diff is exactly the change the human approved.
4. The release commit is pushed as ``<commit>:refs/heads/<branch>``: no ``+``, no ``--force``. A remote that moved
   meanwhile rejects it (``conflict``) and nothing is overwritten. Its trailers record the gate evidence
   (``Diff-SHA256``, ``Quiz-Attempts``, ``Approved-By: human``, ``Rules``, ``Thread``).

``initialize_branch`` (``no-quiz-no-prod init-github``) pushes the seed as the first commit of a branch that does not
exist yet, and ``inspect_remote`` (``init-github --verify``) only reads the advertised tip.

Secrets: the GitHub token only ever reaches git as an ``http.<remote>.extraHeader`` set through ``GIT_CONFIG_*``
environment variables of the network commands. It is never in argv, a remote URL, a git config file, a result, a log
or an error (git's output is redacted anyway). The user's global and system git config are ignored, so no credential
helper can store the token and no hook, signing key or rewrite rule interferes.
"""

import base64
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import (
    TREE_HASH_IGNORED_DIRS,
    TREE_HASH_IGNORED_FILES,
    TREE_HASH_IGNORED_SUFFIXES,
    ReleaseCommit,
    ReleaseRemoteKind,
    RuntimeSettings,
    compute_tree_sha256,
)

GIT_EXECUTABLE: Final = "git"
GIT_MIN_VERSION: Final = (2, 31)  # GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n
GIT_LOCAL_TIMEOUT_S: Final = 30.0
GIT_NETWORK_TIMEOUT_S: Final = 120.0
GIT_OUTPUT_TAIL_CHARS: Final = 600
GATE_AUTHOR_NAME: Final = "No-Quiz-No-Prod Gate"
GATE_AUTHOR_EMAIL: Final = "gate@no-quiz-no-prod.invalid"
GITHUB_TOKEN_USER: Final = "x-access-token"  # GitHub's HTTPS basic-auth user for a token
PUBLISH_INDEX_FILENAME: Final = "publish.index"
REDACTED: Final = "***"
_BRANCH_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_REJECTED_PUSH: Final = ("[rejected]", "non-fast-forward", "fetch first", "stale info", "! [remote rejected]")
_GIT_VERSION: Final = re.compile(r"(\d+)\.(\d+)")
GITHUB_HOST: Final = "github.com"
_GITHUB_NAME: Final = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
TRAILER_VALUE_MAX_CHARS: Final = 200
# The only inherited variables git gets (least privilege: no API key of the parent process reaches it).
GIT_ENV_ALLOWLIST: Final = frozenset(
    {
        "PATH",
        "HOME",
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

# Process-wide git configuration of every gate command (passed through the environment, never written to a file).
_BASE_GIT_CONFIG: Final[tuple[tuple[str, str], ...]] = (
    ("credential.helper", ""),  # no helper may answer with, or store, a credential
    ("core.hooksPath", os.devnull),  # no repository or template hook runs
    ("commit.gpgSign", "false"),
    ("core.autocrlf", "false"),  # committed bytes == hashed bytes
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


def github_repo(url: str | None) -> str | None:
    """``<owner>/<repo>`` of a credential-free ``https://github.com/<owner>/<repo>[.git]`` URL, else None."""
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme != "https" or "@" in parts.netloc or (parts.hostname or "").lower() != GITHUB_HOST:
        return None
    if parts.port not in (None, 443) or parts.query or parts.fragment:
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


def _trailer_value(text: object) -> str:
    """One-line trailer value: whitespace (newlines included) collapsed, bounded."""
    return " ".join(str(text).split())[:TRAILER_VALUE_MAX_CHARS] or "none"


def _exclude_patterns() -> list[str]:
    """``info/exclude`` lines mirroring the tree-hash ignore rules (directories, file names, suffixes)."""
    return [
        *(f"{name}/" for name in sorted(TREE_HASH_IGNORED_DIRS)),
        *sorted(TREE_HASH_IGNORED_FILES),
        *(f"*{suffix}" for suffix in sorted(TREE_HASH_IGNORED_SUFFIXES)),
    ]


class GitReleasePublisher:
    """Builds and pushes the release commit of one deploy (see the module docstring)."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.git_dir = settings.release_git_dir

    # ---- public API ----------------------------------------------------------------------------------------------
    def publish(
        self,
        *,
        source_dir: Path,
        baseline_dir: Path,
        diff_sha256: str,
        prod_tree_sha256: str,
        release_note: str,
        quiz_attempts: int = 0,
        applied_rules: Sequence[str] = (),
        thread_id: str | None = None,
    ) -> ReleaseCommit:
        """Commit ``source_dir`` (staging) on top of ``baseline_dir`` (prod) and push it to the release branch.

        ``prod_tree_sha256`` is the tree hash prod will have once promoted; ``quiz_attempts``, ``applied_rules`` and
        ``thread_id`` (when known) become commit trailers next to ``Diff-SHA256``. Raises ``WorkspaceError`` (``conflict``
        when the remote moved during the deploy, ``upstream_error`` for a refused or failed push, ``timeout`` with
        ``outcome_unknown`` when git did not answer in time, ``invalid_argument`` for a bad remote or branch).
        """
        remote = self.resolve_remote()  # pure validation first: a bad URL never reaches git
        branch = self._branch()
        self._check_git_version()
        self._ensure_git_dir()
        if remote.create:
            self._ensure_bare_remote(Path(remote.target))
        tip = self._remote_tip(remote, branch)
        target_tree = self._write_tree(source_dir)
        if tip is not None and self._tree_of(tip) == target_tree:  # a retried deploy: the remote already holds it
            return self._release(tip, self._first_parent(tip), branch, remote, diff_sha256, prod_tree_sha256, False)
        parent = tip
        baseline_tree = self._write_tree(baseline_dir)
        if parent is None or self._tree_of(parent) != baseline_tree:
            message = (
                "chore(release): sync the release branch with production\n\n"
                "Baseline of the production tree before the next gated deploy.\n"
            )
            parent = self._commit_tree(baseline_tree, parent, message)
        trailers = {
            "Diff-SHA256": diff_sha256,
            "Prod-Tree-SHA256": prod_tree_sha256,
            "Gate": "tests_passed, sha256_matches, quiz_passed, human_approved",
            "Quiz-Attempts": str(quiz_attempts),
            "Approved-By": "human",
            "Rules": ", ".join(applied_rules) or "none",
            **({"Thread": thread_id} if thread_id else {}),
        }
        message = f"deploy: {self._subject(release_note)}\n\n{release_note.strip()}\n\n" + "".join(
            f"{key}: {_trailer_value(value)}\n" for key, value in trailers.items()
        )
        commit = self._commit_tree(target_tree, parent, message)
        self._push(remote, commit, branch)
        self._git("update-ref", f"refs/heads/{branch}", commit)  # local record of the last release
        return self._release(commit, parent, branch, remote, diff_sha256, prod_tree_sha256, True)

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
            tip = self._remote_tip(remote, release.branch)
            if tip is None or not self._is_ancestor(release.commit_sha, tip):
                return False
            return self._tree_of(release.commit_sha) == self._write_tree(prod_dir)
        except WorkspaceError:
            return False

    def git_version(self) -> str:
        """The installed git version; ``internal_error`` when git is missing or older than ``GIT_MIN_VERSION``."""
        return self._check_git_version()

    def inspect_remote(self) -> RemoteHead:
        """Read-only check of the configuration and of the remote (``init-github --verify``): git version, remote
        URL, branch name, then ``ls-remote`` (access + tip). Nothing is fetched, committed or pushed."""
        version = self._check_git_version()
        remote = self.resolve_remote()
        branch = self._branch()
        if remote.create and not (Path(remote.target) / "HEAD").is_file():
            return RemoteHead(remote=remote, branch=branch, tip=None, git_version=version, remote_exists=False)
        return RemoteHead(remote=remote, branch=branch, tip=self._advertised_tip(remote, branch), git_version=version)

    def initialize_branch(self, source_dir: Path, *, message: str) -> RemoteHead:
        """Push ``source_dir`` as the root commit of the release branch if the remote does not have it yet.

        An existing branch is left exactly as it is (``created=False``); a new one is created by a plain
        fast-forward push (``conflict`` if someone created it meanwhile, never forced).
        """
        remote = self.resolve_remote()
        branch = self._branch()
        version = self._check_git_version()
        self._ensure_git_dir()
        if remote.create:
            self._ensure_bare_remote(Path(remote.target))
        tip = self._advertised_tip(remote, branch)
        if tip is not None:
            return RemoteHead(remote=remote, branch=branch, tip=tip, git_version=version)
        commit = self._commit_tree(self._write_tree(source_dir), None, message)
        self._push(remote, commit, branch)
        self._git("update-ref", f"refs/heads/{branch}", commit)
        return RemoteHead(remote=remote, branch=branch, tip=commit, git_version=version, created=True)

    def resolve_remote(self) -> ReleaseRemote:
        """The configured remote, validated: an ``https://`` URL without credentials, or a local repository path.

        Never echoes the configured URL in an error, since a misconfigured one may embed a secret.
        """
        url = self.settings.git_remote_url
        if url is None:
            path = self.settings.default_release_remote_dir
            return ReleaseRemote(target=str(path), display=str(path), kind="local", create=True)
        parts = urlsplit(url)
        if parts.scheme == "https":
            if "@" in parts.netloc or parts.username is not None or parts.password is not None:
                raise WorkspaceError(
                    "invalid_argument",
                    "NO_QUIZ_GIT_REMOTE_URL must not embed credentials: set the token in NO_QUIZ_GITHUB_TOKEN.",
                )
            if not parts.hostname or parts.query or parts.fragment:
                raise WorkspaceError("invalid_argument", "NO_QUIZ_GIT_REMOTE_URL is not a valid https:// git URL.")
            return ReleaseRemote(target=url, display=url, kind="https")
        local = Path(parts.path) if parts.scheme == "file" and not parts.netloc else Path(url)
        if parts.scheme not in ("", "file") or not local.is_absolute():
            raise WorkspaceError(
                "invalid_argument",
                "NO_QUIZ_GIT_REMOTE_URL must be an https:// URL or an absolute path to a local git repository.",
            )
        return ReleaseRemote(target=str(local), display=str(local), kind="local")

    # ---- git steps -----------------------------------------------------------------------------------------------
    def _branch(self) -> str:
        branch = self.settings.git_branch
        if not _BRANCH_NAME.match(branch) or ".." in branch or branch.endswith((".lock", "/", ".")) or "//" in branch:
            raise WorkspaceError("invalid_argument", f"NO_QUIZ_GIT_BRANCH {branch!r} is not a valid branch name.")
        self._git("check-ref-format", f"refs/heads/{branch}", in_repo=False)
        return branch

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
        exclude = self.git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("\n".join(_exclude_patterns()) + "\n", encoding="utf-8")

    def _ensure_bare_remote(self, path: Path) -> None:
        if not (path / "HEAD").is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._git("init", "--quiet", "--bare", str(path), in_repo=False)

    def _advertised_tip(self, remote: ReleaseRemote, branch: str) -> str | None:
        """The commit id the remote advertises for the branch (``ls-remote``), or None if the branch is new."""
        ref = f"refs/heads/{branch}"
        listing = self._git("ls-remote", "--heads", remote.target, ref, remote=remote)
        tips = [line.split("\t", 1)[0] for line in listing.splitlines() if line.endswith(f"\t{ref}")]
        return tips[0] if tips else None

    def _remote_tip(self, remote: ReleaseRemote, branch: str) -> str | None:
        """The commit the remote branch points at, with its objects fetched, or None if the branch is new."""
        advertised = self._advertised_tip(remote, branch)
        if advertised is None:
            return None
        ref = f"refs/heads/{branch}"
        self._git("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", remote.target, ref, remote=remote)
        # no FETCH_HEAD and no ref is written: the objects are now local, so the advertised tip resolves directly
        return self._git("rev-parse", "--verify", f"{advertised}^{{commit}}").strip()

    def _is_ancestor(self, commit: str, tip: str) -> bool:
        try:
            self._git("merge-base", "--is-ancestor", commit, tip)
        except WorkspaceError as error:
            if error.payload.details.get("exit_code") == 1:  # git's "not an ancestor" answer
                return False
            raise
        return True

    def _write_tree(self, directory: Path) -> str:
        """The git tree of ``directory`` (tree-hash ignore rules applied), built in a throwaway index."""
        index = self.git_dir / PUBLISH_INDEX_FILENAME
        index.unlink(missing_ok=True)
        env = {"GIT_INDEX_FILE": str(index)}
        try:
            self._git(f"--work-tree={directory}", "add", "--all", "--", ".", env=env, cwd=directory)
            return self._git("write-tree", env=env).strip()
        finally:
            index.unlink(missing_ok=True)

    def _tree_of(self, commit: str) -> str:
        return self._git("rev-parse", "--verify", f"{commit}^{{tree}}").strip()

    def _first_parent(self, commit: str) -> str | None:
        parents = self._git("rev-list", "--parents", "-n", "1", commit).split()
        return parents[1] if len(parents) > 1 else None

    def _commit_tree(self, tree: str, parent: str | None, message: str) -> str:
        args = ["commit-tree", tree, *(["-p", parent] if parent else []), "-F", "-"]
        return self._git(*args, stdin=message).strip()

    def _push(self, remote: ReleaseRemote, commit: str, branch: str) -> None:
        """Fast-forward push of ``commit`` to ``branch``: the refspec has no ``+`` and no force flag exists here."""
        refspec = f"{commit}:refs/heads/{branch}"
        try:
            self._git("push", "--quiet", "--porcelain", "--no-verify", remote.target, refspec, remote=remote)
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
        self,
        commit: str,
        parent: str | None,
        branch: str,
        remote: ReleaseRemote,
        diff_sha256: str,
        prod_tree_sha256: str,
        pushed: bool,
    ) -> ReleaseCommit:
        return ReleaseCommit(
            commit_sha=commit,
            parent_sha=parent,
            branch=branch,
            remote=remote.display,
            remote_kind=remote.kind,
            diff_sha256=diff_sha256,
            prod_tree_sha256=prod_tree_sha256,
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
        token = self.settings.github_token.get_secret_value() if self.settings.github_token else ""
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

        The token header is added for network commands to an ``https`` remote only, scoped to that remote's URL.
        """
        env = {name: value for name, value in os.environ.items() if name in GIT_ENV_ALLOWLIST}
        env |= {
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
        config = list(_BASE_GIT_CONFIG)
        token = self.settings.github_token
        if remote is not None and remote.kind == "https" and token is not None:
            header = f"Authorization: Basic {self._basic_auth(token.get_secret_value())}"
            config.append((f"http.{remote.target}.extraHeader", header))
        env["GIT_CONFIG_COUNT"] = str(len(config))
        for index, (key, value) in enumerate(config):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env | dict(extra or {})

    def _git(
        self,
        *args: str,
        in_repo: bool = True,
        remote: ReleaseRemote | None = None,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
        cwd: Path | None = None,
    ) -> str:
        """Run one git command and return its stdout; any failure is a ``WorkspaceError`` with redacted output.

        ``in_repo`` runs it against the gate's commit cache; ``remote`` marks a network command (longer timeout, the
        token header when applicable).
        """
        prefix: Sequence[str] = [f"--git-dir={self.git_dir}"] if in_repo else []
        argv = [GIT_EXECUTABLE, *prefix, *args]
        timeout = GIT_NETWORK_TIMEOUT_S if remote is not None else GIT_LOCAL_TIMEOUT_S
        workdir = cwd or self.settings.workspace_dir
        workdir.mkdir(parents=True, exist_ok=True)  # a missing cwd must not read as "git is not installed"
        try:
            completed = subprocess.run(  # fixed executable, no shell, token-free argv
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._environment(remote, env),
                cwd=workdir,
                check=False,
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
            output = self._redact(f"{completed.stderr}\n{completed.stdout}".strip())
            where = f" ({remote.display})" if remote is not None else ""
            raise WorkspaceError(
                "upstream_error" if remote is not None else "internal_error",
                f"git {args[0]}{where} failed with exit code {completed.returncode}.",
                details={"git_command": args[0], "exit_code": completed.returncode, "git_output": output},
            )
        return completed.stdout


__all__ = [
    "GitReleasePublisher",
    "ReleaseRemote",
    "RemoteHead",
    "describe_release_remote",
    "github_commit_url",
    "github_repo",
]
