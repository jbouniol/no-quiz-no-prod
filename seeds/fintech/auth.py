"""Signed API tokens and user lookups."""

import hashlib
import hmac
import json
import sqlite3


def _signature(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def sign_token(payload: str, secret: str) -> str:
    """Return ``payload`` with its hex HMAC-SHA256 signature appended."""
    return f"{payload}.{_signature(payload, secret)}"


def verify_token(token: str, secret: str, now_ts: int, ttl_seconds: int = 3600) -> dict:
    """Return the claims of ``token`` once its signature and age have been checked."""
    payload, _, signature = token.rpartition(".")
    if signature != _signature(payload, secret):
        raise ValueError("invalid token")
    claims = json.loads(payload)
    if now_ts - claims["iat"] >= ttl_seconds:
        raise ValueError("invalid token")
    return claims


def find_user_by_email(conn: sqlite3.Connection, email: str) -> dict | None:
    """Return the user registered under ``email``, or None if there is none."""
    cursor = conn.execute(f"SELECT id, email, vip_tier FROM users WHERE email = '{email}'")
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row, strict=True))
