"""RULE-003: calculate_order_total rejects discount_rate outside [0, 1] with ValueError (n17)."""

from decimal import Decimal

import pytest
from orders import calculate_order_total

ITEMS = [{"price": "10.00", "quantity": 1}]


@pytest.mark.parametrize("discount_rate", ["-0.01", "-0.50", "1.01", -0.1, 1.5])
def test_discount_rate_out_of_bounds_raises_value_error(discount_rate):
    with pytest.raises(ValueError):
        calculate_order_total(ITEMS, discount_rate=discount_rate)


def test_boundary_discount_rates_are_accepted():
    full_price = calculate_order_total(ITEMS, discount_rate="0.00")
    free_order = calculate_order_total(ITEMS, discount_rate="1.00")
    assert Decimal(str(full_price)) == Decimal("12.00")
    assert Decimal(str(free_order)) == Decimal("0.00")
