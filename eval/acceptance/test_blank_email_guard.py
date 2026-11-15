"""Blank or whitespace-only emails return None without querying the database (ad07)."""

import pytest
from acceptance_support import open_db
from auth import find_user_by_email


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_email_returns_none_without_querying_db(db, blank):
    queries: list[str] = []
    db.set_trace_callback(queries.append)
    assert find_user_by_email(db, blank) is None
    assert queries == []


def test_non_blank_email_still_finds_user(db):
    assert find_user_by_email(db, "alice@example.com") == {
        "id": 1,
        "email": "alice@example.com",
        "vip_tier": "VIP_GOLD",
    }
