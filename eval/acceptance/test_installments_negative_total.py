"""RULE-001: split_into_installments rejects negative totals with ValueError (had02)."""

from decimal import Decimal

import pytest
from orders import split_into_installments


@pytest.mark.parametrize("total", ["-0.01", "-90.00", Decimal("-10.00")])
def test_negative_total_raises_value_error(total):
    with pytest.raises(ValueError):
        split_into_installments(total, 3)


def test_positive_total_still_splits():
    assert split_into_installments("90.00", 3) == [Decimal("30.00")] * 3
