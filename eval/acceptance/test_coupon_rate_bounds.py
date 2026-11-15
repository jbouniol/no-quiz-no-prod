"""RULE-003: compute_discount_rate rejects coupon rates outside (0, 1] with ValueError (hn01)."""

from decimal import Decimal

import pytest
from acceptance_support import DAY, NOW
from discounts import compute_discount_rate

SUBTOTAL = Decimal("120.00")


def coupon(rate: str) -> dict:
    return {"code": "PROMO", "rate": rate, "expires_at": NOW + 7 * DAY, "min_subtotal": "50.00"}


@pytest.mark.parametrize("rate", ["0", "0.00", "-0.10", "1.01"])
def test_coupon_rate_out_of_bounds_raises_value_error(rate):
    with pytest.raises(ValueError):
        compute_discount_rate(coupon(rate), None, now_ts=NOW, subtotal=SUBTOTAL)


def test_valid_coupon_rates_are_accepted():
    assert compute_discount_rate(coupon("0.10"), None, now_ts=NOW, subtotal=SUBTOTAL) == Decimal("0.10")
    assert compute_discount_rate(coupon("1.00"), None, now_ts=NOW, subtotal=SUBTOTAL) in (
        Decimal("1.00"),
        Decimal("0.30"),
    )
