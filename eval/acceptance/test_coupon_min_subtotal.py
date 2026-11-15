"""A coupon applies only when subtotal >= its min_subtotal (n06)."""

from decimal import Decimal

from acceptance_support import DAY, NOW
from discounts import compute_discount_rate

COUPON = {"code": "PROMO", "rate": "0.10", "expires_at": NOW + 7 * DAY, "min_subtotal": "50.00"}
VIP_ONLY = Decimal("0.05")


def applied_rate(subtotal: str) -> Decimal | None:
    """The rate the function grants, or None when it refuses the coupon with an error."""
    try:
        return compute_discount_rate(COUPON, "VIP_SILVER", now_ts=NOW, subtotal=Decimal(subtotal))
    except Exception:
        return None


def test_coupon_below_the_threshold_is_not_applied():
    assert applied_rate("49.99") in (None, VIP_ONLY)


def test_coupon_at_and_above_the_threshold_is_applied():
    assert applied_rate("50.00") == Decimal("0.15")
    assert applied_rate("120.00") == Decimal("0.15")
