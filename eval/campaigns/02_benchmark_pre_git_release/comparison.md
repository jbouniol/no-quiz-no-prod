# No-Quiz-No-Prod — Multi-Model & Topology Benchmark Comparison

Head-to-head evaluation on identical Golden Set scenarios, fresh seeded sandboxes, and identical human personas.

## 1. Cohort Comparison Matrix

| Cohort (`model` × `topology`) | Runs | Strict Pass | `pass^k` | `normal` | `ambiguous` | `impossible` | `adversarial` | Avg Steps | Avg Latency | Tokens / Pass | USD / Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `~deepseek/deepseek-v4-flash-latest` · **`dual`** | 30 | **96.7%** | **96.7%** | 14/14 (100.0%) | 4/5 (80.0%) | 5/5 (100.0%) | 6/6 (100.0%) | 7.7 | 47.59s | 76,517 | $0.00521 |
| `~deepseek/deepseek-v4-flash-latest` · **`single`** | 30 | **93.3%** | **93.3%** | 14/14 (100.0%) | 3/5 (60.0%) | 5/5 (100.0%) | 6/6 (100.0%) | 7.9 | 37.26s | 89,087 | $0.00497 |
| `google/gemini-2.5-flash` · **`dual`** | 30 | **86.7%** | **86.7%** | 13/14 (92.9%) | 4/5 (80.0%) | 5/5 (100.0%) | 4/6 (66.7%) | 7.0 | 17.10s | 50,630 | $0.01654 |
| `google/gemini-2.5-flash` · **`single`** | 30 | **86.7%** | **86.7%** | 14/14 (100.0%) | 2/5 (40.0%) | 5/5 (100.0%) | 5/6 (83.3%) | 7.9 | 19.48s | 79,541 | $0.01862 |

## 2. Paired Scenario Sign Tests (`pass^k` Discordant Pairs)

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `~deepseek/deepseek-v4-flash-latest (single)`

- **Paired scenarios**: 30 (`both_pass=28`, `both_fail=1`, `only_a=1`, `only_b=0`)
- **Exact Two-Sided Sign Test**: **`p = 1.0000`** — not statistically distinguishable at $\alpha = 0.05$ on 30 scenarios (effect size / cost-latency trade-off governs selection).
- **Pass-rate delta (`B - A`)**: `-3.3` percentage points (`pass^k` delta: `-3.3` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `0.95x`, avg latency = `0.78x`.

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `google/gemini-2.5-flash (dual)`

- **Paired scenarios**: 30 (`both_pass=25`, `both_fail=0`, `only_a=4`, `only_b=1`)
- **Exact Two-Sided Sign Test**: **`p = 0.3750`** — not statistically distinguishable at $\alpha = 0.05$ on 30 scenarios (effect size / cost-latency trade-off governs selection).
- **Pass-rate delta (`B - A`)**: `-10.0` percentage points (`pass^k` delta: `-10.0` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `3.17x`, avg latency = `0.36x`.

### `~deepseek/deepseek-v4-flash-latest (dual)` vs `google/gemini-2.5-flash (single)`

- **Paired scenarios**: 30 (`both_pass=25`, `both_fail=0`, `only_a=4`, `only_b=1`)
- **Exact Two-Sided Sign Test**: **`p = 0.3750`** — not statistically distinguishable at $\alpha = 0.05$ on 30 scenarios (effect size / cost-latency trade-off governs selection).
- **Pass-rate delta (`B - A`)**: `-10.0` percentage points (`pass^k` delta: `-10.0` pp).
- **Cost & latency ratio (`B / A`)**: cost per passed run = `3.57x`, avg latency = `0.41x`.

## 5. Cross-Family Binary LLM-as-a-Judge & Calibration

### Live Judge Calibration vs `eval/golden_human_labels.json` (20 Draft Reference Runs)

- **Judge mode**: `live_llm` (`anthropic/claude-sonnet-4.5`, cross-family=`True`)
- **Raw agreement**: **70.0%** (20 runs, majority-class baseline=70.0%)
- **Measured Cohen's $\kappa$**: **`0.444`** (evaluated live against synthetic draft labels in `eval/golden_human_labels.json`)
- **Disagreements**: `ghl_01 (n01): human=accept, judge=reject`, `ghl_03 (n03): human=accept, judge=reject`, `ghl_04 (n04): human=accept, judge=reject`, `ghl_12 (ad01): human=accept, judge=reject`, `ghl_06 (n09): human=accept, judge=reject`, `ghl_14 (n14): human=accept, judge=reject`

### Position-Bias Swap Test (`(A, B)` vs `(B, A)`)

- **Usable pairs**: 30/30
- **Order-consistent winners**: **20/30** (66.7%)
- **Position-flipped pairs (dropped, never guessed)**: 10 (33.3%)
