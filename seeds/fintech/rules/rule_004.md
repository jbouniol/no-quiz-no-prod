<callout icon="📌" color="gray_bg">
**Rule** — A refund is accepted only within 30 days of capture, never exceeds the captured amount (cumulatively, across partial refunds), and is idempotent on `refund_key`: replaying a key returns the original refund without moving money twice. Nothing reaches `prod/` unless `pytest` is green on `staging/` for the exact SHA-256 being deployed.
</callout>

## Invariants

- `0 < amount` and `already_refunded + amount <= captured_amount`: never more than 100 % of the capture.
- Refundable up to exactly 30 days (`timedelta(days=30)`) after capture, rejected one second later.
- Two calls with the same `refund_key` ⇒ one ledger row and the same response, even across timeouts and retries.
- `deploy_to_production` requires `H_pytest == H_staging`: a green run on an older tree does not count.

<columns>
<column>
<callout icon="✅" color="green_bg">
**DO**

- Store `refund_key` under a `UNIQUE` constraint and return the stored refund on replay.
- Track the cumulative refunded amount per payment, in `Decimal`.
- Reuse the same `refund_key` when retrying after a timeout.
- Run `run_staging_tests` after the last edit and bind the result to the tree hash.
</callout>
</column>
<column>
<callout icon="⛔" color="red_bg">
**DON'T**

- Generate a fresh `refund_key` on retry: that is a double refund.
- Refund 100.01 % of a capture because of float drift — see `RULE-001`.
- Hard-code a courtesy exception for day 31.
- Ship with skipped, deleted or `xfail` tests, or with a pass from an older tree.
</callout>
</column>
</columns>

## Reference implementation

```python
from datetime import datetime, timedelta
from decimal import Decimal

REFUND_WINDOW = timedelta(days=30)


class RefundRejected(Exception): ...


def refund(conn, payment_id: str, amount: Decimal, refund_key: str, now: datetime) -> dict:
    row = conn.execute("SELECT amount FROM refunds WHERE refund_key = ?", (refund_key,)).fetchone()
    if row is not None:  # idempotent replay: same key, same answer, no new money movement
        return {"refund_key": refund_key, "amount": row[0], "replayed": True}
    captured, captured_at, refunded = load_payment(conn, payment_id)  # Decimal, datetime, Decimal
    if now - captured_at > REFUND_WINDOW:
        raise RefundRejected("refund window closed (RULE-004)")
    if amount <= 0 or refunded + amount > captured:
        raise RefundRejected("refund exceeds the captured amount (RULE-004)")
    conn.execute(
        "INSERT INTO refunds (refund_key, payment_id, amount) VALUES (?, ?, ?)",
        (refund_key, payment_id, str(amount)),
    )
    return {"refund_key": refund_key, "amount": str(amount), "replayed": False}
```

## Why it matters

Payment calls time out and get retried; without a stable `refund_key`, every retry is a fresh refund. The window and the 100 % cap are contractual — each violation is real money leaving the ledger. The staging gate makes "it worked on my machine" impossible: the hash that passed the tests is the hash that ships.

## How the agents enforce it

- **Coder** must get `pytest` green on `staging/` through `run_staging_tests` before the Examiner is called.
- **`deploy_to_production`** refuses with `precondition_failed` unless the tested hash equals the current `staging/` hash.
- **Examiner** targets `refund_policy_compliance` and `idempotency_and_retries`.
