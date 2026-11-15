"""RULE-003: a VIP tier stacked with a coupon is capped at 30% (n04, n13, ad06, hn04)."""

from decimal import Decimal

from acceptance_support import DAY, NOW
from discounts import compute_discount_rate

SUBTOTAL = Decimal("120.00")


def coupon(rate: str) -> dict:
    return {"code": "PROMO", "rate": rate, "expires_at": NOW + 7 * DAY, "min_subtotal": "50.00"}


def test_stack_above_the_cap_is_capped():
    assert compute_discount_rate(coupon("0.25"), "VIP_GOLD", now_ts=NOW, subtotal=SUBTOTAL) == Decimal("0.30")


def test_stack_exactly_at_the_cap_is_kept():
    assert compute_discount_rate(coupon("0.25"), "VIP_SILVER", now_ts=NOW, subtotal=SUBTOTAL) == Decimal("0.30")


def test_stack_below_the_cap_is_unchanged():
    assert compute_discount_rate(coupon("0.10"), "VIP_GOLD", now_ts=NOW, subtotal=SUBTOTAL) == Decimal("0.22")
    assert compute_discount_rate(None, "VIP_SILVER", now_ts=NOW) == Decimal("0.05")
