"""A plan needs at least one installment (n03)."""

import pytest
from orders import split_into_installments


@pytest.mark.parametrize("count", [0, -1])
def test_non_positive_count_raises_value_error(count):
    with pytest.raises(ValueError):
        split_into_installments("90.00", count)
