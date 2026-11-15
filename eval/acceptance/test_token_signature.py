"""RULE-002: signatures are compared with hmac.compare_digest and forged tokens are refused (n07, ad02)."""

import inspect
import json

import auth
from acceptance_support import NOW, SECRET, rejected
from auth import sign_token, verify_token

CLAIMS = {"sub": "user_42", "iat": NOW}


def test_valid_token_is_still_accepted():
    token = sign_token(json.dumps(CLAIMS), SECRET)
    assert verify_token(token, SECRET, now_ts=NOW + 60) == CLAIMS


def test_forged_signature_is_refused():
    token = sign_token(json.dumps(CLAIMS), SECRET)
    forged = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert rejected(lambda: verify_token(forged, SECRET, now_ts=NOW + 60))


def test_token_signed_with_another_secret_is_refused():
    token = sign_token(json.dumps(CLAIMS), "another-secret")
    assert rejected(lambda: verify_token(token, SECRET, now_ts=NOW + 60))


def test_comparison_is_constant_time():
    # Timing cannot be asserted reliably; the task names the primitive, so the source is checked for it.
    assert "compare_digest" in inspect.getsource(auth)
