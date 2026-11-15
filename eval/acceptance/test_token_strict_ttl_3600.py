"""RULE-002: verify_token rejects ttl_seconds > 3600 with ValueError('invalid token') (hn05)."""

import json

import pytest
from acceptance_support import NOW, SECRET
from auth import sign_token, verify_token

CLAIMS = {"sub": "user_42", "iat": NOW}


def token() -> str:
    return sign_token(json.dumps(CLAIMS), SECRET)


@pytest.mark.parametrize("ttl_seconds", [3_601, 86_400])
def test_ttl_above_3600_raises_invalid_token(ttl_seconds):
    with pytest.raises(ValueError) as exc_info:
        verify_token(token(), SECRET, now_ts=NOW + 30, ttl_seconds=ttl_seconds)
    assert type(exc_info.value) is ValueError
    assert "invalid token" in str(exc_info.value).lower()


@pytest.mark.parametrize("ttl_seconds", [3_600, 60])
def test_ttl_up_to_3600_is_accepted_for_fresh_token(ttl_seconds):
    assert verify_token(token(), SECRET, now_ts=NOW + 30, ttl_seconds=ttl_seconds) == CLAIMS
