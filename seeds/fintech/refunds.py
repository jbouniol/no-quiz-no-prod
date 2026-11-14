"""Refunds of captured orders."""

import sqlite3

REFUND_WINDOW_SECONDS = 30 * 86400


def issue_refund(conn: sqlite3.Connection, order_id: str, amount_cents: int, refund_key: str, now_ts: int) -> dict:
    """Refund ``amount_cents`` on a captured order and record it in the refund ledger."""
    order = conn.execute("SELECT id, captured_cents, captured_at FROM orders WHERE id = ?", (order_id,)).fetchone()
    if order is None:
        raise ValueError(f"unknown order {order_id}")
    conn.execute(
        "INSERT INTO refund_ledger (refund_key, order_id, amount_cents, created_at) VALUES (?, ?, ?, ?)",
        (refund_key, order_id, amount_cents, now_ts),
    )
    conn.commit()
    return {"refund_key": refund_key, "order_id": order_id, "amount_cents": amount_cents, "replayed": False}
