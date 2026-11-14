"""Checkout totals and installment plans."""

from decimal import Decimal


def calculate_order_total(
    items: list[dict], vat_rate: str | float = "0.20", discount_rate: str | float = "0.00"
) -> Decimal | float:
    """Return the amount to charge for ``items``: the discounted subtotal plus VAT."""
    subtotal = sum(float(item["price"]) * item["quantity"] for item in items)
    net = round(subtotal * (1 - float(discount_rate)), 2)
    vat = round(net * float(vat_rate), 2)
    return round(net + vat, 2)


def split_into_installments(total: str | Decimal, count: int) -> list[Decimal]:
    """Split ``total`` into ``count`` equal installments."""
    installment = (Decimal(str(total)) / count).quantize(Decimal("0.01"))
    return [installment] * count
