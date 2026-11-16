# No-Quiz-No-Prod — Multi-Model & Topology Benchmark Comparison

Head-to-head evaluation on identical scenarios, fresh seeded sandboxes, and identical human personas.

- **Scenario set**: `heldout` — held-out set: never used to tune a prompt or the harness; keep it that way.

> [!IMPORTANT]
> No pair below is statistically distinguishable: **this table does not rank the cohorts**. Read the intervals, not the order of the rows.

## 1. Cohort Comparison Matrix

| Cohort (`model` × `topology`) | Scenarios × k | Strict Pass (runs) | `pass^k` | 95% Wilson CI of `pass^k` | `normal` | `ambiguous` | `impossible` | `adversarial` | Avg Steps | Avg Latency | Tokens / Pass | USD / Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `~deepseek/deepseek-v4-flash-latest` · **`dual`** | 14 × 3 | 92.9% | **78.6%** | 52.4% – 92.4% | 15/15 (100.0%) | 7/9 (77.8%) | 9/9 (100.0%) | 8/9 (88.9%) | 7.3 | 44.35s | 50,872 | $0.00417 |
| `~deepseek/deepseek-v4-flash-latest` · **`single`** | 14 × 3 | 83.3% | **78.6%** | 52.4% – 92.4% | 15/15 (100.0%) | 2/9 (22.2%) | 9/9 (100.0%) | 9/9 (100.0%) | 7.7 | 27.33s | 75,097 | $0.00561 |

## 2. Paired Scenario Sign Tests (`pass^k` Discordant Pairs, all cohort pairs)

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `~deepseek/deepseek-v4-flash-latest (single)`

- **Paired scenarios**: 14 (`both_pass=10`, `both_fail=2`, `only_a=1`, `only_b=1`)
- **Exact Two-Sided Sign Test**: **`p = 1.0000`** — not statistically distinguishable at $\alpha = 0.05$ on 14 scenarios.
- **Pass-rate delta (`B - A`)**: `-9.5` percentage points (`pass^k` delta: `+0.0` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `1.35x`, avg latency = `0.62x`.
