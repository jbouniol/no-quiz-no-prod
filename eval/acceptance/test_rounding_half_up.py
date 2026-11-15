"""RULE-001: order totals are Decimal amounts rounded half up to the cent (n01, n13, ad01)."""

from decimal import Decimal

from orders import calculate_order_total


def test_vat_tie_rounds_up_on_reduced_rate():
    # 19.00 * 0.055 = 1.045: half up gives 1.05; float round() and half-even both give 1.04.
    assert calculate_order_total([{"price": "19.00", "quantity": 1}], vat_rate="0.055") == Decimal("20.05")


def test_vat_tie_rounds_up_on_small_amount():
    # 0.50 * 0.05 = 0.025: half up gives 0.03; half-even gives 0.02.
    assert calculate_order_total([{"price": "0.50", "quantity": 1}], vat_rate="0.05") == Decimal("0.53")


def test_vat_tie_after_an_exact_discount():
    # 20.00 less 5% is exactly 19.00, then the same 1.045 tie.
    items = [{"price": "10.00", "quantity": 2}]
    assert calculate_order_total(items, vat_rate="0.055", discount_rate="0.05") == Decimal("20.05")


def test_totals_that_never_tied_are_unchanged():
    items = [{"price": "40.00", "quantity": 2}, {"price": "20.00", "quantity": 1}]
    assert calculate_order_total(items, vat_rate="0.20", discount_rate="0.10") == Decimal("108.00")
    assert calculate_order_total([{"price": "50.00", "quantity": 1}], vat_rate="0.055") == Decimal("52.75")
