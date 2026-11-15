"""RULE-003: a coupon whose expires_at <= now_ts is not applied (n05)."""

from decimal import Decimal

import pytest
from acceptance_support import NOW
from discounts import compute_discount_rate

SUBTOTAL = Decimal("120.00")
VIP_ONLY = Decimal("0.05")


def coupon(expires_at: int) -> dict:
    return {"code": "PROMO", "rate": "0.10", "expires_at": expires_at, "min_subtotal": "50.00"}


def applied_rate(expires_at: int) -> Decimal | None:
    """The rate the function grants, or None when it refuses the coupon with an error."""
    try:
        return compute_discount_rate(coupon(expires_at), "VIP_SILVER", now_ts=NOW, subtotal=SUBTOTAL)
    except Exception:
        return None


@pytest.mark.parametrize("expires_at", [NOW, NOW - 1])
def test_expired_coupon_is_not_applied(expires_at):
    assert applied_rate(expires_at) in (None, VIP_ONLY)


def test_coupon_one_second_before_expiry_is_applied():
    assert applied_rate(NOW + 1) == Decimal("0.15")
