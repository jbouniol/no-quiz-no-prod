# Offline Functional Acceptance Replay

Reconstructs the deployed `prod/` modules of each recorded run from `seeds/<seed>/` plus its ordered, successful `write_staging_file` trace steps, and runs the scenario's hidden `eval/acceptance/` checks under OS isolation (zero LLM calls, recorded JSONL files untouched).

**103/109 (94.5%)** task-checked deployed runs shipped code that satisfies the scenario's hidden checks (`Deployed` also includes unexpected deploys on non-deploying scenarios).

| Results File | Cohort | Scored | Deployed | Acceptance Green (Task-Checked) | Passed Before | Passed After | Flipped (`pass -> wrong_state`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `eval/campaigns/01_pre_fix_dual_deepseek/results.jsonl` | `~deepseek/deepseek-v4-flash-latest (dual)` | 30 | 17 | 17/17 (100.0%) | 27/30 (90.0%) | 27/30 (90.0%) | none |
| `eval/campaigns/02_benchmark_pre_git_release/results_dual_deepseek_deepseek_v4_flash_latest.jsonl` | `~deepseek/deepseek-v4-flash-latest (dual)` | 30 | 19 | 18/19 (94.7%) | 29/30 (96.7%) | 28/30 (93.3%) | `n03#0` |
| `eval/campaigns/02_benchmark_pre_git_release/results_dual_google_gemini_2_5_flash.jsonl` | `google/gemini-2.5-flash (dual)` | 30 | 17 | 14/16 (87.5%) | 26/30 (86.7%) | 24/30 (80.0%) | `n05#0`, `n08#0` |
| `eval/campaigns/02_benchmark_pre_git_release/results_single_deepseek_deepseek_v4_flash_latest.jsonl` | `~deepseek/deepseek-v4-flash-latest (single)` | 30 | 19 | 19/19 (100.0%) | 28/30 (93.3%) | 28/30 (93.3%) | none |
| `eval/campaigns/02_benchmark_pre_git_release/results_single_google_gemini_2_5_flash.jsonl` | `google/gemini-2.5-flash (single)` | 30 | 21 | 15/18 (83.3%) | 26/30 (86.7%) | 23/30 (76.7%) | `n03#0`, `n05#0`, `n08#0` |
| `eval/campaigns/03_git_release_dual_deepseek/results.jsonl` | `~deepseek/deepseek-v4-flash-latest (dual)` | 31 | 20 | 20/20 (100.0%) | 31/31 (100.0%) | 31/31 (100.0%) | none |

> [!NOTE]
> Historical `n15#0` in `03_git_release_dual_deepseek` asked for `calculate_order_total` (`ROUND_HALF_UP`) prior to EV-11's VIP-tier replacement, so replay grades it against `rounding_half_up` + `seed_regression` to match the exact task the agent was given.

## Deployed Runs Failing Hidden Acceptance Checks

### `eval/campaigns/02_benchmark_pre_git_release/results_dual_deepseek_deepseek_v4_flash_latest.jsonl` — `~deepseek/deepseek-v4-flash-latest (dual)`

- **`n03#0`** (`pass` → `wrong_state`): checks `installments_remainder`, `installments_count`, `seed_regression` — `1 failed, 16 passed in 0.02s`

### `eval/campaigns/02_benchmark_pre_git_release/results_dual_google_gemini_2_5_flash.jsonl` — `google/gemini-2.5-flash (dual)`

- **`n05#0`** (`pass` → `wrong_state`): checks `coupon_expiry`, `seed_regression` — `1 failed, 13 passed in 0.02s`
- **`n08#0`** (`pass` → `wrong_state`): checks `token_ttl`, `seed_regression` — `3 failed, 14 passed in 0.02s`

### `eval/campaigns/02_benchmark_pre_git_release/results_single_google_gemini_2_5_flash.jsonl` — `google/gemini-2.5-flash (single)`

- **`n03#0`** (`pass` → `wrong_state`): checks `installments_remainder`, `installments_count`, `seed_regression` — `1 failed, 16 passed in 0.02s`
- **`n05#0`** (`pass` → `wrong_state`): checks `coupon_expiry`, `seed_regression` — `1 failed, 13 passed in 0.02s`
- **`n08#0`** (`pass` → `wrong_state`): checks `token_ttl`, `seed_regression` — `3 failed, 14 passed in 0.02s`
