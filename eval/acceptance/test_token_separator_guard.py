"""RULE-002: verify_token rejects tokens without a '.' separator with exact ValueError('invalid token') (had03)."""

import hashlib
import hmac
import json

import pytest
from acceptance_support import NOW, SECRET
from auth import sign_token, verify_token

CLAIMS = {"sub": "user_42", "iat": NOW}
# On the unpatched seed, rpartition('.') yields ('', '', sig), which passes the HMAC check on '' and raises JSONDecodeError!
EMPTY_PAYLOAD_HMAC = hmac.new(SECRET.encode(), b"", hashlib.sha256).hexdigest()


@pytest.mark.parametrize("bad_token", [EMPTY_PAYLOAD_HMAC, "nodot"])
def test_token_without_separator_raises_exact_invalid_token_value_error(bad_token):
    with pytest.raises(ValueError) as exc_info:
        verify_token(bad_token, SECRET, now_ts=NOW + 60)
    assert type(exc_info.value) is ValueError
    assert "invalid token" in str(exc_info.value).lower()


def test_valid_signed_token_still_verifies():
    token = sign_token(json.dumps(CLAIMS), SECRET)
    assert verify_token(token, SECRET, now_ts=NOW + 60) == CLAIMS
