"""Discount rates for VIP tiers and promotional coupons."""

from decimal import Decimal

VIP_TIERS = {"VIP_SILVER": Decimal("0.05"), "VIP_GOLD": Decimal("0.12")}


def compute_discount_rate(
    coupon: dict | None, vip_tier: str | None, now_ts: int, subtotal: Decimal = Decimal("0")
) -> Decimal:
    """Return the order discount rate: the VIP tier rate plus the coupon rate, if any."""
    rate = VIP_TIERS[vip_tier] if vip_tier is not None else Decimal("0")
    if coupon is not None:
        if coupon["expires_at"] < now_ts:
            raise ValueError(f"coupon {coupon['code']} has expired")
        rate += Decimal(str(coupon["rate"]))
    return rate
