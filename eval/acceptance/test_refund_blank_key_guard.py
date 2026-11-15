"""RULE-004: issue_refund rejects empty or whitespace-only refund_key with ValueError (had01)."""

import pytest
from acceptance_support import NOW, ledger_amounts, open_db
from refunds import issue_refund


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


@pytest.mark.parametrize("refund_key", ["", "   ", "\t\n"])
def test_blank_refund_key_raises_value_error(db, refund_key):
    with pytest.raises(ValueError):
        issue_refund(db, "ord_1001", 2_500, refund_key, now_ts=NOW)
    assert ledger_amounts(db) == []


def test_non_blank_refund_key_is_accepted(db):
    refund = issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=NOW)
    assert refund["refund_key"] == "rk_001"
    assert ledger_amounts(db) == [2_500]
