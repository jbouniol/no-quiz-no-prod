<callout icon="📌" color="gray_bg">
**Rule** — Every monetary amount (price, VAT, discount, refund, total) is a `decimal.Decimal` built from a string or an integer, then quantized to the cent with `Decimal("0.01")` and `ROUND_HALF_UP`. Binary `float` never touches money or VAT.
</callout>

## Invariants

- Every money value that is returned, stored or compared satisfies `x == x.quantize(CENT)`.
- VAT is computed on the quantized net: `vat = (net * rate).quantize(CENT, rounding=ROUND_HALF_UP)`.
- `gross == net + vat` to the cent: a sum of quantized parts is never rounded again.
- Money crosses API and JSON boundaries as strings (`"19.99"`), never as JSON numbers.

<columns>
<column>
<callout icon="✅" color="green_bg">
**DO**

- Parse with `Decimal(str(value))` at the boundary, then stay in `Decimal` end to end.
- Pass `rounding=ROUND_HALF_UP` explicitly on every `quantize()`.
- Quantize once, where a money value is produced: line total, VAT, discount, refund.
- Reuse one shared `CENT = Decimal("0.01")` constant.
</callout>
</column>
<column>
<callout icon="⛔" color="red_bg">
**DON'T**

- Use `float` anywhere on a money path: `price * 1.2`, `float(row["amount"])`.
- Call `round(x, 2)`: binary error plus round-half-to-even.
- Build `Decimal(0.1)` from a float literal: it imports the binary error.
- Rely on the default `Decimal` context: it rounds half to even.
</callout>
</column>
</columns>

## Reference implementation

```python
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_money(value: Decimal | str | int) -> Decimal:
    """Parse a money value and quantize it to the cent (RULE-001)."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def vat_amount(net: Decimal, rate: Decimal) -> Decimal:
    return (to_money(net) * rate).quantize(CENT, rounding=ROUND_HALF_UP)
```

## Why it matters

| Expression | Result | Why |
| --- | --- | --- |
| `round(19.00 * 0.055, 2)` | `1.04` | the product is stored as `1.04499999999999992894…`, so there is no tie to round up |
| `round(0.125, 2)` | `0.12` | an exact tie, but `round()` goes to the even cent |
| `Decimal("2.665").quantize(CENT)` | `2.66` | the default context is `ROUND_HALF_EVEN` |
| `to_money("2.665")` | `2.67` | RULE-001: explicit `ROUND_HALF_UP` |

Sweeping every net price from €1.00 to €199.99 through the float path gives 45 one-cent VAT errors at 5.5 % and 456 at 10 % (none at 20 %).

## How the agents enforce it

- **Coder** cites `RULE-001` before touching `orders.py`, `discounts.py` or `refunds.py`.
- **Tests** pin half-cent ties and reduced VAT rates in `tests/test_fintech.py`.
- **Examiner** targets `decimal_rounding` and `boundary_conditions` whenever this rule is applied.
