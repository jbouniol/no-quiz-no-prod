# No-Quiz-No-Prod — Multi-Model & Topology Benchmark Comparison

Head-to-head evaluation on identical scenarios, fresh seeded sandboxes, and identical human personas.

- **Scenario set**: `dev` — development set: the prompts and the harness were tuned on these scenarios, so every rate below is an optimistic estimate.

> [!WARNING]
> **`google/gemini-2.5-flash` · `single` is invalid**: 85 of its 114 runs are `api_error` (OpenRouter HTTP 402, credits ran out mid-campaign). Its row and every pair that involves it measure the provider, not the agent: do not cite them. Rerun that cohort alone. `google/gemini-2.5-flash` · `dual` has 5 `api_error` runs (HTTP 429), counted as failures.

## 1. Cohort Comparison Matrix

| Cohort (`model` × `topology`) | Scenarios × k | Strict Pass (runs) | `pass^k` | 95% Wilson CI of `pass^k` | `normal` | `ambiguous` | `impossible` | `adversarial` | Avg Steps | Avg Latency | Tokens / Pass | USD / Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `~deepseek/deepseek-v4-flash-latest` · **`dual`** | 38 × 3 | 93.0% | **86.8%** | 72.7% – 94.2% | 56/57 (98.2%) | 12/18 (66.7%) | 17/18 (94.4%) | 21/21 (100.0%) | 8.7 | 54.89s | 69,945 | $0.00585 |
| `~deepseek/deepseek-v4-flash-latest` · **`single`** | 38 × 3 | 93.0% | **86.8%** | 72.7% – 94.2% | 55/57 (96.5%) | 14/18 (77.8%) | 17/18 (94.4%) | 20/21 (95.2%) | 8.2 | 39.88s | 74,906 | $0.00416 |
| `google/gemini-2.5-flash` · **`dual`** | 38 × 3 | 76.3% | **68.4%** | 52.5% – 80.9% | 52/57 (91.2%) | 11/18 (61.1%) | 12/18 (66.7%) | 12/21 (57.1%) | 8.4 | 29.86s | 70,790 | $0.02018 |
| `google/gemini-2.5-flash` · **`single`** | 38 × 3 | 20.2% | **15.8%** | 7.4% – 30.4% | 23/57 (40.4%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/21 (0.0%) | 3.3 | 9.99s | 137,452 | $0.03350 |

## 2. Paired Scenario Sign Tests (`pass^k` Discordant Pairs, all cohort pairs)

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `~deepseek/deepseek-v4-flash-latest (single)`

- **Paired scenarios**: 38 (`both_pass=30`, `both_fail=2`, `only_a=3`, `only_b=3`)
- **Exact Two-Sided Sign Test**: **`p = 1.0000`** — not statistically distinguishable at $\alpha = 0.05$ on 38 scenarios.
- **Pass-rate delta (`B - A`)**: `+0.0` percentage points (`pass^k` delta: `+0.0` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `0.71x`, avg latency = `0.73x`.

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `google/gemini-2.5-flash (dual)`

- **Paired scenarios**: 38 (`both_pass=24`, `both_fail=3`, `only_a=9`, `only_b=2`)
- **Exact Two-Sided Sign Test**: **`p = 0.0654`** — not statistically distinguishable at $\alpha = 0.05$ on 38 scenarios.
- **Pass-rate delta (`B - A`)**: `-16.7` percentage points (`pass^k` delta: `-18.4` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `3.45x`, avg latency = `0.54x`.

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `google/gemini-2.5-flash (single)`

- **Paired scenarios**: 38 (`both_pass=6`, `both_fail=5`, `only_a=27`, `only_b=0`)
- **Exact Two-Sided Sign Test**: **`p = 0.0000`** — statistically significant at $\alpha = 0.05$.
- **Pass-rate delta (`B - A`)**: `-72.8` percentage points (`pass^k` delta: `-71.1` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `5.73x`, avg latency = `0.18x`.

### `~deepseek/deepseek-v4-flash-latest (single)` vs `google/gemini-2.5-flash (dual)`

- **Paired scenarios**: 38 (`both_pass=25`, `both_fail=4`, `only_a=8`, `only_b=1`)
- **Exact Two-Sided Sign Test**: **`p = 0.0391`** — statistically significant at $\alpha = 0.05$.
- **Pass-rate delta (`B - A`)**: `-16.7` percentage points (`pass^k` delta: `-18.4` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `4.85x`, avg latency = `0.75x`.

### `~deepseek/deepseek-v4-flash-latest (single)` vs `google/gemini-2.5-flash (single)`

- **Paired scenarios**: 38 (`both_pass=6`, `both_fail=5`, `only_a=27`, `only_b=0`)
- **Exact Two-Sided Sign Test**: **`p = 0.0000`** — statistically significant at $\alpha = 0.05$.
- **Pass-rate delta (`B - A`)**: `-72.8` percentage points (`pass^k` delta: `-71.1` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `8.05x`, avg latency = `0.25x`.

### `google/gemini-2.5-flash (dual)` vs `google/gemini-2.5-flash (single)`

- **Paired scenarios**: 38 (`both_pass=6`, `both_fail=12`, `only_a=20`, `only_b=0`)
- **Exact Two-Sided Sign Test**: **`p = 0.0000`** — statistically significant at $\alpha = 0.05$.
- **Pass-rate delta (`B - A`)**: `-56.1` percentage points (`pass^k` delta: `-52.6` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `1.66x`, avg latency = `0.33x`.
