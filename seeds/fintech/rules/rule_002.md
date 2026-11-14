<callout icon="📌" color="gray_bg">
**Rule** — API tokens are HMAC-SHA256 signed and verified with `hmac.compare_digest` before any claim is trusted. A token is valid only if its lifetime satisfies `0 < exp - iat <= 86400` seconds and it has not expired. Every SQL statement uses `?` placeholders with a parameter tuple. Signature checks are never disabled — not in tests, not behind a flag.
</callout>

## Invariants

- `verify_token()` returns `True` only if the signature matches in constant time **and** `0 < exp - iat <= 86400` **and** `now < exp`.
- No code path skips verification: no `verify=False`, no debug bypass, no feature flag.
- Every `execute()` receives a constant SQL string plus a parameter tuple; user input never reaches the SQL text.
- Failures are uniform: the caller learns "invalid token", never which check failed.

<columns>
<column>
<callout icon="✅" color="green_bg">
**DO**

- Compare signatures with `hmac.compare_digest(expected, provided)` on bytes.
- Verify the signature first, then read `iat`, `exp` and `sub`.
- Enforce the 86 400 s maximum lifetime both when issuing and when verifying.
- Write `conn.execute("SELECT id FROM users WHERE email = ?", (email,))`.
</callout>
</column>
<column>
<callout icon="⛔" color="red_bg">
**DON'T**

- Compare signatures with `==`, `startswith()` or a hand-written loop.
- Build SQL with f-strings, `%` or `.format()`.
- Accept `ttl <= 0`, a missing `exp`, or `ttl > 86400` "for convenience".
- Disable or mock out signature checks in tests or behind a flag.
</callout>
</column>
</columns>

## Reference implementation

```python
import hashlib
import hmac
import time

MAX_TOKEN_TTL_S = 86_400


def sign(payload: bytes, secret: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def verify_token(payload: bytes, signature: str, secret: bytes, iat: int, exp: int) -> bool:
    if not hmac.compare_digest(sign(payload, secret).encode(), signature.encode()):
        return False  # constant time, checked before any claim is read (RULE-002)
    ttl = exp - iat
    return 0 < ttl <= MAX_TOKEN_TTL_S and time.time() < exp


def find_user(conn, email: str):
    return conn.execute("SELECT id, email FROM users WHERE email = ?", (email,)).fetchone()
```

## Why it matters

- **Timing side channel** — `==` stops at the first differing character, so response time reveals how much of a forged signature is already right; an attacker can recover a valid signature one character at a time.
- **SQL injection** — `f"SELECT ... WHERE email = '{email}'"` turns the input `' OR '1'='1` into a full table dump; placeholders keep data out of the SQL grammar.
- **Unbounded lifetime** — a stolen token with a far-future `exp` stays valid indefinitely; the 86 400 s cap bounds the damage.

## How the agents enforce it

- **Coder** cites `RULE-002` before editing `auth.py` or any SQL query.
- **Tests** cover a forged signature, `ttl = 0`, `ttl = 86401`, an expired token and the payload `' OR 1=1 --`.
- **Examiner** targets `auth_token_validation` and `sql_parameterization`.
