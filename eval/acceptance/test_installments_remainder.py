"""RULE-001: installments add up to the total, remainder cents go to the earliest ones (n03, ad04)."""

from decimal import Decimal

import pytest
from orders import split_into_installments


@pytest.mark.parametrize(
    ("total", "count", "expected"),
    [
        ("100.00", 3, ["33.34", "33.33", "33.33"]),
        ("0.05", 3, ["0.02", "0.02", "0.01"]),
        ("10.00", 4, ["2.50", "2.50", "2.50", "2.50"]),
        ("90.00", 3, ["30.00", "30.00", "30.00"]),
    ],
)
def test_parts_sum_to_total_with_early_remainder(total, count, expected):
    parts = split_into_installments(total, count)
    assert sum(parts) == Decimal(total)
    assert list(parts) == [Decimal(value) for value in expected]
