"""RULE-004: cumulative refunds never exceed the captured amount (n11)."""

import pytest
from acceptance_support import DAY, NOW, ledger_amounts, open_db
from refunds import issue_refund


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


def test_cumulative_over_refund_raises_and_writes_nothing(db):
    issue_refund(db, "ord_1001", 6_000, "rk_001", now_ts=NOW)
    with pytest.raises(ValueError):
        issue_refund(db, "ord_1001", 5_000, "rk_002", now_ts=NOW + DAY)
    assert ledger_amounts(db) == [6_000]


def test_single_over_refund_raises(db):
    with pytest.raises(ValueError):
        issue_refund(db, "ord_1001", 10_001, "rk_001", now_ts=NOW)
    assert ledger_amounts(db) == []


def test_refunds_up_to_the_captured_amount_are_accepted(db):
    issue_refund(db, "ord_1001", 6_000, "rk_001", now_ts=NOW)
    issue_refund(db, "ord_1001", 4_000, "rk_002", now_ts=NOW + DAY)
    assert ledger_amounts(db) == [6_000, 4_000]
