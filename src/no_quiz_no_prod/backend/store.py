"""Workspace persistence: the SQLite state database (WAL), the gate ledger, the release journal and the locks.

Both MCP servers and the harness are separate processes sharing one ``<workspace_dir>``. Every multi-step operation
(reset, staging write, ledger update, promotion, grading) runs under ``workspace_lock``: a reentrant in-process lock
plus an exclusive ``flock`` on ``<workspace_dir>/.workspace.lock``, so no reader ever sees half of a change. It is
never held across a network call or a test run: a deploy reserves ``deploy_slot`` instead (non-blocking: a second
deploy, or a reset during one, is refused as ``run_active``).

The gate ledger lives in ``state.sqlite`` (table ``gate_ledger``), so a lock is written in the same transaction as
the fact that earns it (a graded quiz row, a pytest run). SQLite runs in WAL mode with a busy timeout, and every
write transaction is ``BEGIN IMMEDIATE``.
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

from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import GateLedgerEntry, ReleaseJournalState

try:  # POSIX only: elsewhere the in-process lock still serializes threads, processes are not coordinated.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

STATE_DB_FILENAME: Final = "state.sqlite"
LOCK_FILENAME: Final = ".workspace.lock"
DEPLOY_LOCK_FILENAME: Final = ".deploy.lock"
SCHEMA_VERSION: Final = 2  # 2: gate_ledger and release_journal tables, quizzes.epoch
# Journal entries still to reconcile: a publish that began ("push_started") or landed ("pushed") but is not completed.
_OPEN_JOURNAL_ENTRIES: Final = (
    "SELECT id, record FROM release_journal WHERE state IN ('push_started', 'pushed') ORDER BY id"
)
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
    "DELETE FROM gate_ledger",
    "DELETE FROM release_journal",
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
    grade TEXT,
    epoch INTEGER NOT NULL DEFAULT 1
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
CREATE TABLE IF NOT EXISTS gate_ledger (
    entry TEXT NOT NULL,
    diff_sha256 TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (entry, diff_sha256)
);
CREATE TABLE IF NOT EXISTS release_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diff_sha256 TEXT NOT NULL,
    state TEXT NOT NULL,
    record TEXT NOT NULL
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


_BUSY_DEPLOY_SLOTS: set[Path] = set()


@contextmanager
def deploy_slot(workspace_dir: Path) -> Iterator[None]:
    """Reserve the workspace's single slot for long operations: a whole deploy (network calls included), a staged
    test run, a reset.

    Non-blocking: when another thread or process holds it, raises a retryable ``run_active`` error instead of
    waiting. The workspace lock is NOT held by this slot, so other tools keep working during a slow push; and since
    tests and deploys exclude each other, no test process is alive while git holds the release token.
    """
    path = (workspace_dir / DEPLOY_LOCK_FILENAME).resolve()
    busy = WorkspaceError(
        "run_active",
        "A deploy or a staged test run is in progress on this workspace: wait until it is over, then repeat the call.",
        retryable=True,
    )
    with _LOCKS_GUARD:
        if path in _BUSY_DEPLOY_SLOTS:
            raise busy
        _BUSY_DEPLOY_SLOTS.add(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise busy from None
            yield
        finally:
            os.close(descriptor)  # closing the descriptor releases the flock
    finally:
        with _LOCKS_GUARD:
            _BUSY_DEPLOY_SLOTS.discard(path)


# ---- SQLite --------------------------------------------------------------------------------------------------------
@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Autocommit connection with the schema in place; use ``transaction`` for writes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_S, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(conn)
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


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the schema once per database file (``PRAGMA user_version``): no data is ever dropped."""
    if conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION:
        return
    conn.executescript(SCHEMA)
    with transaction(conn):
        if "epoch" not in {row["name"] for row in conn.execute("PRAGMA table_info(quizzes)")}:  # a version 1 file
            conn.execute("ALTER TABLE quizzes ADD COLUMN epoch INTEGER NOT NULL DEFAULT 1")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


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


# ---- gate ledger ---------------------------------------------------------------------------------------------------
def ledger_record(conn: sqlite3.Connection, entry: GateLedgerEntry, diff_sha256: str, epoch: int) -> None:
    """Grant ``entry`` to ``diff_sha256`` (no-op if it already holds). Call it inside the transaction that earns it."""
    conn.execute(
        "INSERT OR IGNORE INTO gate_ledger (entry, diff_sha256, epoch, created_at) VALUES (?, ?, ?, ?)",
        (entry, diff_sha256, epoch, utc_now().isoformat()),
    )


def ledger_revoke(conn: sqlite3.Connection, entry: GateLedgerEntry, diff_sha256: str) -> bool:
    """Withdraw ``entry`` from ``diff_sha256``; True when it was held."""
    cursor = conn.execute("DELETE FROM gate_ledger WHERE entry = ? AND diff_sha256 = ?", (entry, diff_sha256))
    return cursor.rowcount > 0


def ledger_entries(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """``{entry: [diff hashes, oldest first]}`` for every entry that has at least one hash."""
    entries: dict[str, list[str]] = {}
    for row in conn.execute("SELECT entry, diff_sha256 FROM gate_ledger ORDER BY rowid"):
        entries.setdefault(row["entry"], []).append(row["diff_sha256"])
    return entries


# ---- release journal -----------------------------------------------------------------------------------------------
def journal_append(conn: sqlite3.Connection, diff_sha256: str, state: ReleaseJournalState, record: str) -> int:
    """Open a journal entry (``record`` is the ``ReleaseJournalEntry`` JSON); returns its id."""
    cursor = conn.execute(
        "INSERT INTO release_journal (diff_sha256, state, record) VALUES (?, ?, ?)", (diff_sha256, state, record)
    )
    return int(cursor.lastrowid or 0)


def journal_update(conn: sqlite3.Connection, journal_id: int, state: ReleaseJournalState, record: str) -> None:
    conn.execute("UPDATE release_journal SET state = ?, record = ? WHERE id = ?", (state, record, journal_id))


def journal_open_entries(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """``[(id, record JSON)]`` of the entries still to reconcile (publish begun or landed, not completed)."""
    return [(int(row["id"]), str(row["record"])) for row in conn.execute(_OPEN_JOURNAL_ENTRIES)]
