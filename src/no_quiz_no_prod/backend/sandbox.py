"""OS isolation for untrusted code: the agent-written staged tests never run next to the gate's state.

``run_isolated`` runs this interpreter on one private directory (the *sandbox root*) and nothing else:

- **Filesystem**: read and write inside the root; read-only access to the interpreter (``sys.prefix`` and
  ``sys.base_prefix``) and to the system libraries; everything else is denied, in particular the workspace (ledger,
  ``prod/``, ``state.sqlite``, checkpoints, release remote), the repository (``.env``, ``src/``) and the user's home.
- **Network**: none, loopback included.
- **Environment**: built here from scratch (``HOME`` and ``TMPDIR`` inside the root); nothing is inherited, so no API
  key, token or ``NO_QUIZ_*`` variable of the parent reaches the child.
- **Lifetime and resources**: its own process group, killed as a whole on timeout and again when the run ends;
  output read into a bounded buffer; CPU time, file size, open files and core dumps capped with ``setrlimit``.

Isolation backends (``IsolationLevel``):

- ``sandbox-exec`` (macOS): a deny-default Seatbelt profile. Enforced: filesystem, network (loopback included) and
  Mach/IPC lookups. Not enforced: ``sysctl(KERN_PROCARGS2)`` still exposes the initial launch environment and argv of
  same-user processes (secrets must therefore never be placed in a process's launch environment), a memory cap (macOS
  ignores ``RLIMIT_AS``), and a process that leaves the process group with ``setsid`` survives the ``SIGKILL`` (it
  stays inside the sandbox and under ``RLIMIT_CPU``).
- ``bubblewrap`` (Linux): new user, mount, PID, network, IPC and UTS namespaces; only the allow-listed paths are
  mounted. The PID namespace dies with the run, so nothing survives it. ``RLIMIT_AS`` caps memory.
- ``none``: no isolation exists on this platform. ``require_isolation`` then refuses (``sandbox_unavailable``) unless
  the caller passes the explicit unsafe opt-out, and the level is reported so the gate can show it.

A backend is trusted only after a probe really ran this interpreter under it (``detect_isolation``).
"""

import functools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, NamedTuple

from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import EnvVar, IsolationLevel

logger = logging.getLogger(__name__)

SANDBOX_EXEC: Final = "/usr/bin/sandbox-exec"
BUBBLEWRAP: Final = "bwrap"
SANDBOX_ROOT_PREFIX: Final = "nqnp_sandbox_"
HOME_DIRNAME: Final = "home"
TMP_DIRNAME: Final = "tmp"
TIMEOUT_EXIT_CODE: Final = 124  # the `timeout(1)` convention: the run was killed, it did not pass
PROBE_TIMEOUT_S: Final = 20.0
PROBE_MODULE: Final = "pytest"  # the probe imports what the sandboxed runs need
READ_CHUNK_BYTES: Final = 65_536
READER_JOIN_TIMEOUT_S: Final = 2.0
MAX_SYMLINK_DEPTH: Final = 16
SANDBOX_PATH: Final = "/usr/bin:/bin"
# Flags of every sandboxed interpreter: no user site-packages, no bytecode files, UTF-8 I/O.
PYTHON_FLAGS: Final = ("-s", "-B", "-X", "utf8")
BASE_ENVIRONMENT: Final[Mapping[str, str]] = {
    "PATH": SANDBOX_PATH,
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}

# Read-only system locations a sandboxed interpreter needs (macOS). No user data lives under any of them.
MACOS_SYSTEM_READ_PATHS: Final = (
    "/usr/lib",
    "/usr/bin",
    "/usr/share",
    "/usr/libexec",
    "/bin",
    "/System",
    "/private/var/db/dyld",
    "/private/var/db/timezone",
    "/private/etc/localtime",
)
# Single files: the devices, and the root directory entry itself (the dynamic loader reads it; its listing only
# names the top-level system directories).
MACOS_LITERAL_READ_PATHS: Final = ("/", "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
MACOS_DEVICE_WRITE_PATHS: Final = ("/dev/null",)
# Read-only system locations mounted in the bubblewrap namespace (Linux); the missing ones are skipped by bwrap.
LINUX_SYSTEM_READ_PATHS: Final = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc/ld.so.cache")

# Runs inside the sandbox before the real command: applies the limits it is given, then replaces itself.
_BOOTSTRAP: Final = """
import json, os, resource, sys
for name, value in json.loads(sys.argv[1]).items():
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
    except (ValueError, OSError):
        print(f"sandbox: {name} could not be applied", file=sys.stderr)
os.execv(sys.executable, [sys.executable, *sys.argv[2:]])
"""


@dataclass(frozen=True)
class SandboxLimits:
    """Bounds of one sandboxed run. ``address_space_bytes`` is only enforced where the kernel honours it (Linux)."""

    timeout_s: float
    output_tail_bytes: int
    file_size_bytes: int = 16 * 1024 * 1024
    open_files: int = 256
    address_space_bytes: int = 2 * 1024 * 1024 * 1024
    cpu_grace_s: int = 5  # CPU-time cap = the wall-clock timeout plus this margin

    def rlimits(self) -> dict[str, int]:
        """``{RLIMIT name: value}`` applied inside the sandbox (soft and hard)."""
        limits = {
            "RLIMIT_CPU": int(self.timeout_s) + self.cpu_grace_s,
            "RLIMIT_FSIZE": self.file_size_bytes,
            "RLIMIT_NOFILE": self.open_files,
            "RLIMIT_CORE": 0,
        }
        if sys.platform.startswith("linux"):
            limits["RLIMIT_AS"] = self.address_space_bytes
        return limits


class SandboxResult(NamedTuple):
    exit_code: int  # TIMEOUT_EXIT_CODE when the run was killed for exceeding its timeout
    output: str  # bounded tail of stdout + stderr
    timed_out: bool
    duration_s: float
    isolation: IsolationLevel


class IsolationProbe(NamedTuple):
    level: IsolationLevel
    detail: str  # why no isolation is available (empty when one is)


# ---- which paths the child may see -----------------------------------------------------------------------------------
def _interpreter_roots() -> list[Path]:
    """The directories holding this interpreter and its packages (virtual environment and base installation)."""
    roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(sys.executable).resolve().parent.parent}
    return sorted(roots)


def _resolution_trail(path: Path, depth: int = 0) -> set[Path]:
    """Every path the kernel reads to resolve ``path``: each symlink on the way, its target, and the final file."""
    trail: set[Path] = set()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if depth < MAX_SYMLINK_DEPTH and current.is_symlink():
            trail.add(current)
            target = Path(os.path.normpath(current.parent / os.readlink(current)))
            trail |= _resolution_trail(target, depth + 1)
            current = target.resolve()
    trail.add(current)
    return trail


_MACHO_MAGIC_64: Final = 0xFEEDFACF
_MACHO_FAT_MAGIC: Final = 0xCAFEBABE
# LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LAZY_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB
_MACHO_DYLIB_COMMANDS: Final = frozenset({0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023})


def _macho_dylibs(path: Path) -> set[str]:
    """The install names a Mach-O file (thin 64-bit or fat) loads; empty when it is not one or cannot be read."""
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    if len(data) < 8:
        return set()
    slices: list[int] = []
    if int.from_bytes(data[:4], "big") == _MACHO_FAT_MAGIC:
        count = int.from_bytes(data[4:8], "big")
        slices = [int.from_bytes(data[8 + 20 * i + 8 : 8 + 20 * i + 12], "big") for i in range(min(count, 8))]
    else:
        slices = [0]
    names: set[str] = set()
    for base in slices:
        if int.from_bytes(data[base : base + 4], "little") != _MACHO_MAGIC_64:
            continue
        ncmds = int.from_bytes(data[base + 16 : base + 20], "little")
        offset = base + 32
        for _ in range(ncmds):
            cmd = int.from_bytes(data[offset : offset + 4], "little")
            size = int.from_bytes(data[offset + 4 : offset + 8], "little")
            if size < 8:
                break
            if cmd in _MACHO_DYLIB_COMMANDS:
                start = offset + int.from_bytes(data[offset + 8 : offset + 12], "little")
                names.add(data[start : offset + size].split(b"\0", 1)[0].decode("utf-8", "replace"))
            offset += size
    return names


@functools.cache
def _linked_libraries() -> frozenset[Path]:
    """Absolute non-system libraries the interpreter and its extension modules load, followed transitively.

    A Homebrew Python links ``_sqlite3``, ``_ssl``, ``_lzma``... against other kegs (``/opt/homebrew/opt/<pkg>/lib``)
    outside ``sys.base_prefix``; without them ``import sqlite3`` fails under the deny-default profile.
    """
    system = tuple(MACOS_SYSTEM_READ_PATHS)
    roots = _interpreter_roots()
    queue = [Path(sys.executable).resolve()]
    for root in roots:
        queue += [path for path in root.glob("lib/python*/lib-dynload/*.so")]
        queue += [path for path in root.glob("Python") if path.is_file()]
    seen: set[Path] = set()
    found: set[Path] = set()
    while queue:
        binary = queue.pop()
        if binary in seen:
            continue
        seen.add(binary)
        for name in _macho_dylibs(binary):
            if not name.startswith("/") or name.startswith(system):
                continue  # @rpath/@loader_path stay inside readable roots; system libraries are already allowed
            library = Path(name)
            if not library.exists():
                continue
            resolved = library.resolve()
            if not any(resolved.is_relative_to(root) for root in roots):
                found.add(library)
            queue.append(resolved)
    return frozenset(found)


def _with_ancestors(paths: Iterable[Path]) -> list[Path]:
    return sorted({ancestor for path in paths for ancestor in (path, *path.parents)})


def _quoted(path: Path | str) -> str:
    """A path as a Seatbelt string literal."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _filters(kind: str, paths: Iterable[Path | str]) -> str:
    return " ".join(f"({kind} {_quoted(path)})" for path in paths)


def seatbelt_profile(root: Path) -> str:
    """The deny-default macOS profile of a run confined to ``root`` (an absolute, resolved path).

    Metadata (``stat``) is allowed on the ancestors of the readable paths only: resolving a path needs it, and it
    reveals no file content.
    """
    readable = [*(Path(path) for path in MACOS_SYSTEM_READ_PATHS), *_interpreter_roots(), root]
    libraries: set[Path] = set()
    for library in sorted(_linked_libraries()):
        libraries |= _resolution_trail(library)
    trail = _resolution_trail(Path(sys.executable)) | set(readable) | libraries
    literal_reads: list[Path | str] = [*MACOS_LITERAL_READ_PATHS, *sorted(libraries)]
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            "(allow process-fork)",
            "(allow process-exec)",
            "(allow signal (target same-sandbox))",
            "(allow sysctl-read)",
            f"(allow file-read* {_filters('subpath', readable)} {_filters('literal', literal_reads)})",
            f"(allow file-read-metadata {_filters('literal', _with_ancestors(trail))})",
            f"(allow file-write* (subpath {_quoted(root)}))",
            f"(allow file-write-data {_filters('literal', MACOS_DEVICE_WRITE_PATHS)})",
        ]
    )


def bubblewrap_argv(root: Path, cwd: Path) -> list[str]:
    """The ``bwrap`` prefix of a run confined to ``root``: only allow-listed paths exist in its mount namespace."""
    argv = [BUBBLEWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for path in LINUX_SYSTEM_READ_PATHS:
        argv += ["--ro-bind-try", path, path]
    for interpreter_root in _interpreter_roots():
        argv += ["--ro-bind", str(interpreter_root), str(interpreter_root)]
    for link in sorted(_resolution_trail(Path(sys.executable))):
        if link.is_symlink():  # the venv's `python` link, recreated so the command resolves as it does outside
            argv += ["--symlink", os.readlink(link), str(link)]
    argv += ["--proc", "/proc", "--dev", "/dev", "--bind", str(root), str(root), "--chdir", str(cwd), "--"]
    return argv


def _isolation_prefix(level: IsolationLevel, root: Path, cwd: Path) -> list[str]:
    if level == "sandbox-exec":
        return [SANDBOX_EXEC, "-p", seatbelt_profile(root)]
    if level == "bubblewrap":
        return bubblewrap_argv(root, cwd)
    return []


def sandbox_environment(root: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The whole environment of a sandboxed run: nothing inherited, ``HOME`` and ``TMPDIR`` inside ``root``."""
    env = dict(BASE_ENVIRONMENT) | {"HOME": str(root / HOME_DIRNAME), "TMPDIR": str(root / TMP_DIRNAME)}
    env |= dict(extra or {})
    leaked = sorted(name for name in env if name in {var.value for var in EnvVar})
    if leaked:
        raise ValueError(f"project variables must never reach a sandboxed run: {leaked}")
    return env


# ---- running -------------------------------------------------------------------------------------------------------
class _TailReader(threading.Thread):
    """Drains a pipe into a buffer that never holds more than ``limit`` bytes (the most recent ones)."""

    def __init__(self, stream: IO[bytes], limit: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._tail = bytearray()

    def run(self) -> None:
        while chunk := os.read(self._stream.fileno(), READ_CHUNK_BYTES):
            self._tail += chunk
            if len(self._tail) > self._limit:
                del self._tail[: len(self._tail) - self._limit]

    def text(self) -> str:
        return bytes(self._tail).decode("utf-8", errors="replace")


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """Kill the whole process group of the run (it is its own session leader), children included."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # the group is already gone


def run_isolated(
    python_args: Sequence[str],
    *,
    root: Path,
    cwd: Path,
    limits: SandboxLimits,
    isolation: IsolationLevel,
    env: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Run ``python <python_args>`` confined to ``root`` under ``isolation`` (see the module docstring).

    ``root`` must be a resolved directory outside anything the child must not see; ``cwd`` lies inside it. ``env``
    adds variables to the minimal environment (never a project variable). Never raises for a failing child: the
    exit code says so.
    """
    for directory in (root / HOME_DIRNAME, root / TMP_DIRNAME):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        *_isolation_prefix(isolation, root, cwd),
        sys.executable,
        *PYTHON_FLAGS,
        "-c",
        _BOOTSTRAP,
        json.dumps(limits.rlimits()),
        *PYTHON_FLAGS,
        *python_args,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(  # noqa: S603 - fixed executable, list argv, no shell
        command,
        cwd=cwd,
        env=sandbox_environment(root, env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # its own process group: one signal reaches every descendant
    )
    assert process.stdout is not None  # stdout=PIPE above
    reader = _TailReader(process.stdout, limits.output_tail_bytes)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        _kill_group(process)  # also on success: a child left behind must not outlive the run or hold the pipe open
        process.wait()
        reader.join(READER_JOIN_TIMEOUT_S)
        process.stdout.close()
    return SandboxResult(
        exit_code=TIMEOUT_EXIT_CODE if timed_out else process.returncode,
        output=reader.text(),
        timed_out=timed_out,
        duration_s=time.perf_counter() - started,
        isolation=isolation,
    )


# ---- detection -----------------------------------------------------------------------------------------------------
def _candidate() -> IsolationProbe:
    if sys.platform == "darwin":
        if Path(SANDBOX_EXEC).is_file():
            return IsolationProbe("sandbox-exec", "")
        return IsolationProbe("none", f"{SANDBOX_EXEC} is missing")
    if sys.platform.startswith("linux"):
        if shutil.which(BUBBLEWRAP) is not None:
            return IsolationProbe("bubblewrap", "")
        return IsolationProbe("none", "bubblewrap (bwrap) is not installed")
    return IsolationProbe("none", f"no sandbox backend for platform {sys.platform!r}")


@functools.lru_cache(maxsize=1)
def detect_isolation() -> IsolationProbe:
    """The isolation this process can really use, probed once: the backend must run this interpreter and import
    what the sandboxed runs need. A backend that is installed but does not work counts as none."""
    candidate = _candidate()
    if candidate.level == "none":
        return candidate
    with tempfile.TemporaryDirectory(prefix=SANDBOX_ROOT_PREFIX) as tmp:
        root = Path(tmp).resolve()
        limits = SandboxLimits(timeout_s=PROBE_TIMEOUT_S, output_tail_bytes=READ_CHUNK_BYTES)
        try:
            result = run_isolated(
                ["-c", f"import {PROBE_MODULE}"], root=root, cwd=root, limits=limits, isolation=candidate.level
            )
        except OSError as error:
            return IsolationProbe("none", f"{candidate.level} could not be started: {error}")
    if result.exit_code != 0:
        tail = " ".join(result.output.split())[-300:]
        return IsolationProbe("none", f"{candidate.level} probe failed with exit code {result.exit_code}: {tail}")
    return candidate


def require_isolation(*, allow_unsafe: bool) -> IsolationLevel:
    """The isolation level to run untrusted code under. Fails closed: without a working backend it raises
    ``sandbox_unavailable`` unless ``allow_unsafe`` (``NO_QUIZ_UNSAFE_NO_SANDBOX=1``), which is logged every time."""
    probe = detect_isolation()
    if probe.level != "none":
        return probe.level
    if not allow_unsafe:
        raise WorkspaceError(
            "sandbox_unavailable",
            "Staged tests cannot be isolated on this machine, so they were not run. Install a sandbox backend "
            f"(macOS: sandbox-exec, Linux: bubblewrap) or set {EnvVar.NO_QUIZ_UNSAFE_NO_SANDBOX.value}=1 to run "
            "them WITHOUT isolation.",
            details={"isolation": "none", "reason": probe.detail},
        )
    logger.warning(
        "%s is set: running agent-written tests WITHOUT isolation (%s)",
        EnvVar.NO_QUIZ_UNSAFE_NO_SANDBOX.value,
        probe.detail,
    )
    return "none"


__all__ = [
    "SANDBOX_ROOT_PREFIX",
    "TIMEOUT_EXIT_CODE",
    "IsolationProbe",
    "SandboxLimits",
    "SandboxResult",
    "bubblewrap_argv",
    "detect_isolation",
    "require_isolation",
    "run_isolated",
    "sandbox_environment",
    "seatbelt_profile",
]
