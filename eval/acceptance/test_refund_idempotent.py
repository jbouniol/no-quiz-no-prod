"""RULE-004: replaying a refund_key returns the original refund and adds no ledger row (n10, ad03)."""

import pytest
from acceptance_support import DAY, NOW, ledger_amounts, open_db
from refunds import issue_refund


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


def test_replay_returns_the_original_refund_once(db):
    first = issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=NOW)
    again = issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=NOW + 5)
    assert (again["refund_key"], again["order_id"], again["amount_cents"]) == ("rk_001", "ord_1001", 2_500)
    assert (first["refund_key"], first["order_id"], first["amount_cents"]) == ("rk_001", "ord_1001", 2_500)
    assert ledger_amounts(db) == [2_500]


def test_distinct_keys_are_distinct_refunds(db):
    issue_refund(db, "ord_1001", 3_000, "rk_001", now_ts=NOW)
    issue_refund(db, "ord_1001", 2_000, "rk_002", now_ts=NOW + DAY)
    assert ledger_amounts(db) == [3_000, 2_000]
