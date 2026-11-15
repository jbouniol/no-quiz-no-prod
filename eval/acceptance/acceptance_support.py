"""Shared fixtures of the hidden acceptance checks: the seed's database schema and two small helpers.

These files live in ``eval/acceptance/``, outside the sandbox workspace: no agent tool can list or read them.
The scorer copies them next to the final ``prod/`` modules and runs them there.
"""

import sqlite3

NOW = 1_763_000_000
DAY = 86_400
SECRET = "acceptance-secret"

SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    vip_tier TEXT
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    captured_cents INTEGER NOT NULL,
    captured_at INTEGER NOT NULL
);
CREATE TABLE refund_ledger (
    refund_key TEXT NOT NULL,
    order_id TEXT NOT NULL REFERENCES orders (id),
    amount_cents INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
"""
USERS = [(1, "alice@example.com", "VIP_GOLD"), (2, "bob@example.com", None), (3, "o'brien@example.com", None)]
ORDERS = [("ord_1001", 10_000, NOW - 2 * DAY)]


def open_db() -> sqlite3.Connection:
    """An in-memory database with the seed's three tables, three users and one captured order."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO users (id, email, vip_tier) VALUES (?, ?, ?)", USERS)
    conn.executemany("INSERT INTO orders (id, captured_cents, captured_at) VALUES (?, ?, ?)", ORDERS)
    conn.commit()
    return conn


def ledger_amounts(conn: sqlite3.Connection) -> list[int]:
    """The refunded amounts recorded in the ledger, in insertion order."""
    return [row[0] for row in conn.execute("SELECT amount_cents FROM refund_ledger ORDER BY rowid")]


def rejected(call) -> bool:
    """True when ``call()`` refuses: it raises, or returns nothing usable."""
    try:
        return not call()
    except Exception:
        return True
