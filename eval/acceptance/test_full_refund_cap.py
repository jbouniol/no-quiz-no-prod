"""RULE-004: a single 100% refund of captured_cents is accepted and anything above it is rejected (n19)."""

import pytest
from acceptance_support import NOW, ledger_amounts, open_db
from refunds import issue_refund


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


def test_full_captured_amount_refund_is_accepted(db):
    refund = issue_refund(db, "ord_1001", 10_000, "rk_full", now_ts=NOW)
    assert refund["amount_cents"] == 10_000
    assert ledger_amounts(db) == [10_000]


def test_refund_above_captured_amount_is_rejected(db):
    with pytest.raises(Exception):  # noqa: B017 - the task requires rejection, regardless of exception class
        issue_refund(db, "ord_1001", 10_001, "rk_over", now_ts=NOW)
    assert ledger_amounts(db) == []
