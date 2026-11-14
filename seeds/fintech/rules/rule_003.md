<callout icon="📌" color="gray_bg">
**Rule** — VIP discounts are fixed by tier: `VIP_SILVER` = 5 %, `VIP_GOLD` = 12 %. If a request does not name the tier, ask — never guess. All stacked discounts (VIP + coupons) are capped at 30 % of the subtotal. Expired coupons are rejected.
</callout>

## Invariants

- `VIP_RATES` holds exactly two tiers; an unknown tier is an error, never a default.
- Missing tier ⇒ the agent stops with a clarification question (`stop_reason = clarification_needed`) and writes nothing.
- `combined_rate = min(vip_rate + sum(coupon_rates), 0.30)`, applied once to the subtotal.
- A coupon is valid through its expiry date and rejected from the next day on.
- The discount amount is quantized per `RULE-001`, and a total never goes negative.

<columns>
<column>
<callout icon="✅" color="green_bg">
**DO**

- Read the tier explicitly from the request or the customer record.
- Ask "Which tier: `VIP_SILVER` (5 %) or `VIP_GOLD` (12 %)?" when it is missing.
- Sum every discount first, then apply the 30 % cap.
- Reject expired coupons with an explicit error.
</callout>
</column>
<column>
<callout icon="⛔" color="red_bg">
**DON'T**

- Default to `VIP_GOLD`, or to any tier, when the request is silent.
- Let a stack exceed the cap: 12 % + a 25 % coupon gives 30 %, not 37 %.
- Accept a coupon after its expiry date because the request insists.
- Invent tiers or rates that this rule does not define.
</callout>
</column>
</columns>

## Reference implementation

```python
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
VIP_RATES = {"VIP_SILVER": Decimal("0.05"), "VIP_GOLD": Decimal("0.12")}
MAX_STACKED_RATE = Decimal("0.30")


class ClarificationNeeded(Exception): ...


class CouponExpired(Exception): ...


@dataclass(frozen=True)
class Coupon:
    code: str
    rate: Decimal
    expires_on: date


def stacked_discount(subtotal: Decimal, tier: str | None, coupons: list[Coupon], today: date) -> Decimal:
    if tier is None:
        raise ClarificationNeeded("Which VIP tier: VIP_SILVER (5%) or VIP_GOLD (12%)?")
    rate = VIP_RATES[tier]  # unknown tier -> KeyError, never a silent default
    for coupon in coupons:
        if today > coupon.expires_on:
            raise CouponExpired(coupon.code)
        rate += coupon.rate
    return (subtotal * min(rate, MAX_STACKED_RATE)).quantize(CENT, rounding=ROUND_HALF_UP)
```

## Why it matters

Guessing a tier silently changes what a customer pays — a 7-point swing between Silver and Gold — and an uncapped stack turns promotions into negative margins. These are the **ambiguous** scenarios of the Golden Set: the only correct outcome is a clarification question, with `staging/`, `prod/` and Notion left untouched.

## How the agents enforce it

- **Coder** must stop and ask when the tier is missing; the eval harness scores it on the final state.
- **Tests** cover the 30 % cap, the coupon expiry boundary and unknown tiers.
- **Examiner** targets `boundary_conditions`.
