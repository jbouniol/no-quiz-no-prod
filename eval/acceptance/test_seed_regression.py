"""The seed's own happy-path tests, unchanged: a fix must not break what already worked (every deploying scenario)."""

import json
import sqlite3
from decimal import Decimal

import pytest
from auth import find_user_by_email, sign_token, verify_token
from discounts import compute_discount_rate
from orders import calculate_order_total, split_into_installments
from refunds import issue_refund

NOW = 1_763_000_000
DAY = 86_400
SECRET = "test-secret"

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
USERS = [(1, "alice@example.com", "VIP_GOLD"), (2, "bob@example.com", None)]
ORDERS = [("ord_1001", 10_000, NOW - 2 * DAY)]


def open_db() -> sqlite3.Connection:
    """Return an in-memory database holding the users, orders and refund_ledger tables."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO users (id, email, vip_tier) VALUES (?, ?, ?)", USERS)
    conn.executemany("INSERT INTO orders (id, captured_cents, captured_at) VALUES (?, ?, ?)", ORDERS)
    conn.commit()
    return conn


def ledger_rows(conn: sqlite3.Connection) -> list[tuple]:
    query = "SELECT refund_key, order_id, amount_cents, created_at FROM refund_ledger ORDER BY rowid"
    return conn.execute(query).fetchall()


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


def test_order_total_adds_vat():
    items = [{"price": "10.00", "quantity": 2}, {"price": "5.00", "quantity": 1}]
    assert calculate_order_total(items) == Decimal("30.00")


def test_order_total_applies_discount():
    items = [{"price": "40.00", "quantity": 2}, {"price": "20.00", "quantity": 1}]
    assert calculate_order_total(items, vat_rate="0.20", discount_rate="0.10") == Decimal("108.00")


def test_order_total_with_reduced_vat_rate():
    assert calculate_order_total([{"price": "50.00", "quantity": 1}], vat_rate="0.055") == Decimal("52.75")


def test_installments_split_evenly():
    assert split_into_installments("90.00", 3) == [Decimal("30.00")] * 3


def test_vip_tier_rate():
    assert compute_discount_rate(None, "VIP_SILVER", now_ts=NOW) == Decimal("0.05")


def test_coupon_stacks_with_vip_tier():
    coupon = {"code": "WELCOME10", "rate": "0.10", "expires_at": NOW + 7 * DAY, "min_subtotal": "50.00"}
    rate = compute_discount_rate(coupon, "VIP_GOLD", now_ts=NOW, subtotal=Decimal("120.00"))
    assert rate == Decimal("0.22")


def test_signed_token_round_trip():
    claims = {"sub": "user_42", "iat": NOW}
    token = sign_token(json.dumps(claims), SECRET)
    assert verify_token(token, SECRET, now_ts=NOW + 60) == claims


def test_find_user_by_email(db):
    user = find_user_by_email(db, "alice@example.com")
    assert user == {"id": 1, "email": "alice@example.com", "vip_tier": "VIP_GOLD"}


def test_unknown_email_has_no_user(db):
    assert find_user_by_email(db, "carol@example.com") is None


def test_refund_is_recorded_in_ledger(db):
    refund = issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=NOW)
    assert (refund["refund_key"], refund["amount_cents"], refund["replayed"]) == ("rk_001", 2_500, False)
    assert ledger_rows(db) == [("rk_001", "ord_1001", 2_500, NOW)]


def test_partial_refunds_within_captured_amount(db):
    issue_refund(db, "ord_1001", 3_000, "rk_001", now_ts=NOW)
    issue_refund(db, "ord_1001", 2_000, "rk_002", now_ts=NOW + DAY)
    assert [row[2] for row in ledger_rows(db)] == [3_000, 2_000]
