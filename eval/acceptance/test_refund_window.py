"""RULE-004: a refund is accepted up to 30 days after capture and refused one second later (n12, n18)."""

import pytest
from acceptance_support import DAY, NOW, ledger_amounts, open_db
from refunds import issue_refund

CAPTURED_AT = NOW - 2 * DAY  # ord_1001
WINDOW = 30 * DAY


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


def test_refund_after_the_window_is_refused(db):
    with pytest.raises(Exception):  # noqa: B017 - the task says "rejected", not which error
        issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=CAPTURED_AT + WINDOW + 1)
    assert ledger_amounts(db) == []


def test_refund_on_the_last_second_of_the_window_is_accepted(db):
    issue_refund(db, "ord_1001", 2_500, "rk_001", now_ts=CAPTURED_AT + WINDOW)
    assert ledger_amounts(db) == [2_500]
