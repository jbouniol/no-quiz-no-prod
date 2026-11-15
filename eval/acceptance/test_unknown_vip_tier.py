"""RULE-003: an unknown VIP tier raises a ValueError naming the tier rather than a KeyError (n15)."""

from decimal import Decimal

import pytest
from acceptance_support import NOW
from discounts import compute_discount_rate


@pytest.mark.parametrize("tier", ["VIP_PLATINUM", "VIP_BRONZE"])
def test_unknown_vip_tier_raises_value_error_naming_the_tier(tier):
    with pytest.raises(ValueError) as exc_info:
        compute_discount_rate(None, tier, now_ts=NOW)
    assert tier in str(exc_info.value)


def test_known_vip_tiers_and_none_keep_their_decimal_rates():
    silver = compute_discount_rate(None, "VIP_SILVER", now_ts=NOW)
    gold = compute_discount_rate(None, "VIP_GOLD", now_ts=NOW)
    guest = compute_discount_rate(None, None, now_ts=NOW)
    assert isinstance(silver, Decimal) and silver == Decimal("0.05")
    assert isinstance(gold, Decimal) and gold == Decimal("0.12")
    assert isinstance(guest, Decimal) and guest == Decimal("0")
