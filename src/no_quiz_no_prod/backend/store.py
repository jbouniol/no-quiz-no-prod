"""Workspace persistence: the SQLite state database (WAL) and the cross-process workspace lock.

Both MCP servers and the harness are separate processes sharing one ``<workspace_dir>``. Every multi-step operation
(reset, staging write, pytest run, ledger update, deploy, grading) runs under ``workspace_lock``: a reentrant
in-process lock plus an exclusive ``flock`` on ``<workspace_dir>/.workspace.lock``, so no reader ever sees half of a
change and no ledger update is lost. SQLite runs in WAL mode with a busy timeout, and every write transaction is
``BEGIN IMMEDIATE``.
"""

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

try:  # POSIX only: elsewhere the in-process lock still serializes threads, processes are not coordinated.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

STATE_DB_FILENAME: Final = "state.sqlite"
LOCK_FILENAME: Final = ".workspace.lock"
SQLITE_BUSY_TIMEOUT_S: Final = 30.0

# Run tables are wiped by every reset; skill_profile is Layer 4 memory and survives `preserve_skill_memory=True`.
# Constant statements only: no SQL text is ever built from a value.
WIPE_RUN_TABLES: Final = (
    "DELETE FROM meta",
    "DELETE FROM notion_rules",
    "DELETE FROM ship_log",
    "DELETE FROM quizzes",
    "DELETE FROM pytest_runs",
    "DELETE FROM idempotency_store",
    "DELETE FROM events",
)
WIPE_MEMORY_TABLES: Final = ("DELETE FROM skill_profile",)

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notion_rules (rule_code TEXT PRIMARY KEY, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ship_log (
    diff_sha256 TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    record TEXT NOT NULL,
    synced_doc_sha256 TEXT,
    updated_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS quizzes (
    question_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    diff_sha256 TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    state TEXT NOT NULL,
    record TEXT NOT NULL,
    grade TEXT
);
CREATE TABLE IF NOT EXISTS pytest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diff_sha256 TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skill_profile (
    user_id TEXT NOT NULL,
    skill_tag TEXT NOT NULL,
    record TEXT NOT NULL,
    PRIMARY KEY (user_id, skill_tag)
);
CREATE TABLE IF NOT EXISTS idempotency_store (
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (server, tool, idempotency_key)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> datetime:
    """Timezone-aware current time (every stored timestamp is UTC)."""
    return datetime.now(UTC)


# ---- cross-process workspace lock ---------------------------------------------------------------------------------
class _ProcessLock:
    """One per lock file and process: an RLock for threads, plus a ``flock`` held while the outermost owner holds it.

    A single descriptor per path matters: BSD/macOS ``flock`` locks belong to the open file description, so two
    descriptors of the same file in one process would block each other.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rlock = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._rlock:
            if self._depth == 0:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_EX)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._fd is not None:
                    if fcntl is not None:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    os.close(self._fd)
                    self._fd = None


_LOCKS: dict[Path, _ProcessLock] = {}
_LOCKS_GUARD = threading.Lock()


def workspace_lock(workspace_dir: Path) -> AbstractContextManager[None]:
    """Reentrant, cross-process exclusive lock of one workspace (use as ``with workspace_lock(root): ...``)."""
    path = (workspace_dir / LOCK_FILENAME).resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(path, _ProcessLock(path))
    return lock.hold()


# ---- SQLite --------------------------------------------------------------------------------------------------------
@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Autocommit connection with the schema in place; use ``transaction`` for writes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_S, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT`` (rolled back on any exception): one atomic, serialized write."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def wipe(conn: sqlite3.Connection, *, keep_memory: bool) -> None:
    """Empty every run table, and the Layer 4 memory too unless ``keep_memory``."""
    for statement in WIPE_RUN_TABLES if keep_memory else (*WIPE_RUN_TABLES, *WIPE_MEMORY_TABLES):
        conn.execute(statement)


def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def idempotency_get(conn: sqlite3.Connection, server: str, tool: str, key: str) -> Any | None:
    """The payload stored for ``(server, tool, key)``, or None when this logical write was never answered."""
    row = conn.execute(
        "SELECT payload FROM idempotency_store WHERE server = ? AND tool = ? AND idempotency_key = ?",
        (server, tool, key),
    ).fetchone()
    return None if row is None else json.loads(row["payload"])


def idempotency_put(conn: sqlite3.Connection, server: str, tool: str, key: str, payload: Any) -> None:
    conn.execute(
        "INSERT INTO idempotency_store (server, tool, idempotency_key, payload, created_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(server, tool, idempotency_key) DO UPDATE SET payload = excluded.payload, "
        "created_at = excluded.created_at",
        (server, tool, key, json.dumps(payload, sort_keys=True), utc_now().isoformat()),
    )


def log_event(conn: sqlite3.Connection, kind: str, detail: dict[str, Any]) -> None:
    """Append one audit event (reset, deploy, decision, honeypot access...)."""
    conn.execute(
        "INSERT INTO events (kind, detail, created_at) VALUES (?, ?, ?)",
        (kind, json.dumps(detail, sort_keys=True, default=str), utc_now().isoformat()),
    )
