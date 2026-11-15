"""RULE-002: verify_token rejects missing or future iat claims with ValueError('invalid token') (hn02)."""

import json

import pytest
from acceptance_support import NOW, SECRET
from auth import sign_token, verify_token


@pytest.mark.parametrize("claims", [{"sub": "user_42"}, {"sub": "user_42", "iat": NOW + 1}])
def test_missing_or_future_iat_raises_invalid_token(claims):
    token = sign_token(json.dumps(claims), SECRET)
    with pytest.raises(ValueError) as exc_info:
        verify_token(token, SECRET, now_ts=NOW)
    assert type(exc_info.value) is ValueError
    assert "invalid token" in str(exc_info.value).lower()


@pytest.mark.parametrize("iat", [NOW, NOW - 60])
def test_present_and_past_iat_within_ttl_are_accepted(iat):
    claims = {"sub": "user_42", "iat": iat}
    token = sign_token(json.dumps(claims), SECRET)
    assert verify_token(token, SECRET, now_ts=NOW) == claims
