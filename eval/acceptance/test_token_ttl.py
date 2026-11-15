"""RULE-002: 0 < ttl_seconds <= 86400, and expired tokens are refused (n08)."""

import json

import pytest
from acceptance_support import NOW, SECRET, rejected
from auth import sign_token, verify_token

CLAIMS = {"sub": "user_42", "iat": NOW}


def token() -> str:
    return sign_token(json.dumps(CLAIMS), SECRET)


@pytest.mark.parametrize("ttl_seconds", [0, -5, 86_401])
def test_ttl_out_of_bounds_is_refused(ttl_seconds):
    assert rejected(lambda: verify_token(token(), SECRET, now_ts=NOW + 60, ttl_seconds=ttl_seconds))


def test_expired_token_is_refused():
    assert rejected(lambda: verify_token(token(), SECRET, now_ts=NOW + 3_601, ttl_seconds=3_600))


@pytest.mark.parametrize("ttl_seconds", [3_600, 86_400])
def test_fresh_token_within_bounds_is_accepted(ttl_seconds):
    assert verify_token(token(), SECRET, now_ts=NOW + 60, ttl_seconds=ttl_seconds) == CLAIMS
