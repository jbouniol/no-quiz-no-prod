"""RULE-001: issue_refund rejects non-int amount_cents (including bool and float) with ValueError (hn03)."""

import pytest
from acceptance_support import NOW, ledger_amounts, open_db
from refunds import issue_refund


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


@pytest.mark.parametrize("amount_cents", [2500.0, True, False, "2500"])
def test_non_int_amount_cents_raises_value_error(db, amount_cents):
    with pytest.raises(ValueError):
        issue_refund(db, "ord_1001", amount_cents, "rk_001", now_ts=NOW)
    assert ledger_amounts(db) == []


def test_integer_amount_cents_is_accepted(db):
    refund = issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=NOW)
    assert refund["amount_cents"] == 2_500
    assert ledger_amounts(db) == [2_500]
