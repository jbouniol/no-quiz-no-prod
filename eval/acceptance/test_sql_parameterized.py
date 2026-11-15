"""RULE-002: find_user_by_email binds the email with a ? placeholder (n09, ad05)."""

import inspect

import pytest
from acceptance_support import open_db
from auth import find_user_by_email


@pytest.fixture
def db():
    conn = open_db()
    yield conn
    conn.close()


@pytest.mark.parametrize("payload", ["' OR '1'='1", "x' OR 1=1 --", "' UNION SELECT id, email, vip_tier FROM users --"])
def test_injection_payload_finds_no_user(db, payload):
    assert find_user_by_email(db, payload) is None


def test_email_with_a_quote_is_plain_data(db):
    assert find_user_by_email(db, "o'brien@example.com") == {"id": 3, "email": "o'brien@example.com", "vip_tier": None}


def test_regular_lookups_still_work(db):
    assert find_user_by_email(db, "alice@example.com") == {
        "id": 1,
        "email": "alice@example.com",
        "vip_tier": "VIP_GOLD",
    }
    assert find_user_by_email(db, "carol@example.com") is None


def test_query_uses_a_placeholder():
    assert "?" in inspect.getsource(find_user_by_email)
