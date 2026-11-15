"""Order items with a negative price or a non-positive quantity are refused (n02)."""

from decimal import Decimal

import pytest
from orders import calculate_order_total


@pytest.mark.parametrize(
    "item",
    [
        {"price": "-1.00", "quantity": 1},
        {"price": "10.00", "quantity": 0},
        {"price": "10.00", "quantity": -2},
    ],
)
def test_invalid_item_raises_value_error(item):
    with pytest.raises(ValueError):
        calculate_order_total([{"price": "5.00", "quantity": 1}, item])


def test_valid_items_still_priced():
    items = [{"price": "10.00", "quantity": 2}, {"price": "5.00", "quantity": 1}]
    assert calculate_order_total(items) == Decimal("30.00")
