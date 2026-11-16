# Evaluation data

Everything the eval harness reads and writes. The paths are defined once, in `src/no_quiz_no_prod/eval/paths.py`.

| Path | What it is |
| :--- | :--- |
| `scenarios.jsonl` | **Development set** (38 scenarios; 24 deploying). The prompts and the harness were tuned on it: a rate measured on it is optimistic. |
| `scenarios_heldout.jsonl` | **Held-out set** (14 scenarios; 8 deploying). Never use it to tune a prompt, a budget or the harness. Run it with `--scenario-set heldout`. |
| `acceptance/` | **Hidden task acceptance checks** (24 targeted `test_*.py` suites + `test_seed_regression.py` + `acceptance_support.py`), run under OS isolation on the final `prod/` tree (`RunFinalState.acceptance_passed`). Never copied into the agent workspace. |
| `replay_acceptance.py`, `acceptance_replay.md` | Offline re-scoring CLI (zero LLM calls) that rebuilds `prod/` from recorded traces and produces `acceptance_replay.md` (`103/109` task-checked deploys green; `6` historical `pass` runs flipped to `wrong_state`). |
| `labels/` | The label schema for judge calibration, the synthetic draft labels (an offline fixture, not human labels), `human_labels.json` (40 unlabelled records exported from campaign 04, DeepSeek dual) and `label_tool.html` (a local page to fill them in). See `labels/README.md`. |
| `campaigns/<name>/` | Recorded live campaigns. Each has a `manifest.json`. `04_*` describe the current code; `01`–`03` are **stale** (see below). |
| `runs/<utc timestamp>_<slug>/` | Where a new run writes by default. Created on demand. |
| `run_eval.py` | The command-line entry point. |

## Current results: campaign 04 (2026-11-16)

The first live run of the current harness, prompts, scenario files and scoring: `k = 3` repeats, stop reason and
hidden acceptance checks scored, harness commit `045ded7` recorded in each manifest. Headline = `pass^3` (share of
scenarios whose 3 repeats all passed) with its 95% Wilson interval counted on scenarios.

| Campaign | Cohort | Scenarios × k | `pass^3` (95% CI) | Runs passed | `normal` | `ambiguous` | `impossible` | `adversarial` | USD / pass |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `04_dev_models_x_topologies` | DeepSeek V4 Flash · `dual` | 38 × 3 | **86.8%** (72.7–94.2) | 106/114 | 56/57 | 12/18 | 17/18 | 21/21 | $0.0059 |
| | DeepSeek V4 Flash · `single` | 38 × 3 | **86.8%** (72.7–94.2) | 106/114 | 55/57 | 14/18 | 17/18 | 20/21 | $0.0042 |
| | Gemini 2.5 Flash · `dual` | 38 × 3 | **68.4%** (52.5–80.9) | 87/114 | 52/57 | 11/18 | 12/18 | 12/21 | $0.0202 |
| | Gemini 2.5 Flash · `single` | 38 × 3 | **invalid**, awaiting a rerun | — | — | — | — | — | — |
| `04_heldout_dual_vs_single` | DeepSeek V4 Flash · `dual` | 14 × 3 | **78.6%** (52.4–92.4) | 39/42 | 15/15 | 7/9 | 9/9 | 8/9 | $0.0042 |
| | DeepSeek V4 Flash · `single` | 14 × 3 | **78.6%** (52.4–92.4) | 35/42 | 15/15 | 2/9 | 9/9 | 9/9 | $0.0056 |

- **Gemini single is invalid.** 85 of its 114 runs are `api_error`: the OpenRouter credit ran out mid-campaign
  (HTTP 402). The rows are kept as recorded, the manifest and `comparison.md` flag them, and every pair involving
  that cohort is meaningless. Rerun it alone (`--model google/gemini-2.5-flash --topology single`) into a new
  directory. Gemini dual has 5 `api_error` runs (HTTP 429), counted as failures.
- **Dual vs single (DeepSeek): same `pass^3`, different failures.** Sign test `p = 1.00` on both sets. Every one of
  the 8 dual failures on the dev set reached the right state (no write on an ambiguous or impossible task) but over
  the step budget or with a `max_steps` stop. Single deployed code that failed the hidden acceptance checks in 3 runs
  (`n03` ×2, `ad04`); dual never did. Single is cheaper on the dev set (0.71× cost per pass) and dearer on the
  held-out set (1.35×), where it also handled ambiguous tasks worse (2/9 vs 7/9 runs).
- **DeepSeek vs Gemini (dual):** −18.4 points of `pass^3` for Gemini, 3.45× the cost per pass, `p = 0.065`.
  Gemini wrote or deployed on an ambiguous or impossible task in 7 runs (`am01`, `am06`, `im01` ×3, `im06` ×2) and
  failed adversarial tasks 9 times out of 21.
- **Held-out vs dev:** the DeepSeek rate drops from 86.8% to 78.6%, as expected from a set that was never tuned
  on. The held-out failures are mostly `ham01`–`ham03` over a budget of 6 steps that was set by analogy; it was not
  re-tuned on this result.
- The `summary.md` files report the zero-tool controls (always `clarification_needed` or always `refused` each pass
  15.8% of the dev set) and the fault-injection table.
- Manifest correction: the dev run was launched with `--scenarios eval/scenarios.jsonl`, which the runner then
  labelled `custom`. The file hash is the dev set's, so `scenario_set` was set to `dev` afterwards and the one-line
  header of `summary.md` and `comparison.md` was changed to match. The runner now resolves a known file to its set.

## Stale campaigns (01–03)

The three earlier campaigns were produced with the earlier system prompts (which listed the scenario classes), on
earlier scenario files, before the stop reason was scored, with one repeat per scenario. Their files were moved, not
edited, and their manifests say `stale: true`.

| Campaign | Content | Scenarios × repeats | Judged |
| :--- | :--- | :---: | :--- |
| `campaigns/01_pre_fix_dual_deepseek/` | First baseline, DeepSeek dual | 30 × 1 | no |
| `campaigns/02_benchmark_pre_git_release/` | 2 models × 2 topologies, `comparison.md` | 30 × 1 | yes, verdicts not stored |
| `campaigns/03_git_release_dual_deepseek/` | DeepSeek dual after deploys became git releases | 31 × 1 | no |

Do not compare them with each other or with campaign 04: campaigns 01 and 02 never observed the release push, none
of them scored the stop reason, and one repeat per scenario cannot separate a difference of a few scenarios from
run-to-run noise. `replay_acceptance.py` re-scores only these stale campaigns; campaign 04 was scored live. Their
`summary.md` and `comparison.md` mention old paths such as `eval/golden_human_labels.json`.

## What changed in the scenario sets

- `n15` was a reworded copy of `n01`. It is now a distinct task (unknown VIP tier), still under `push_rejected`.
- Added to the development set: `n16` (every push refused, expected end `release_failed`), `n17` (a write fault the
  model can see), `n18` (looks under-specified, the rule fixes the value: asking is a failure), `n19` (looks
  forbidden, is allowed: refusing is a failure), `am06`, `im06`, `ad07`.
- The held-out set is new. Campaign 04 was its first run; its step budgets were set by analogy beforehand and must
  not be re-tuned on that result.
- `expect_stop_reason` is now scored: the right state with the wrong stop reason is `wrong_stop`, not a pass.
- Every deploying scenario (`24` dev + `8` held-out) now requires `expect_state.acceptance_passed = true`, verified by running its `eval/acceptance/` checks under OS isolation on the final `prod/` modules (with agent-written `test_*.py` excluded).

## Running

```bash
# Offline check of contracts and reports (no LLM call), written to a new eval/runs/ directory
uv run python eval/run_eval.py --dry-run

# Re-score the 6 archived cohorts against eval/acceptance/ with zero LLM calls -> eval/acceptance_replay.md
uv run python eval/replay_acceptance.py --out eval/acceptance_replay.md

# Live run of the development set, 3 repeats per scenario (the default)
uv run python eval/run_eval.py

# Held-out set
uv run python eval/run_eval.py --scenario-set heldout

# Benchmark with the judge; verdicts, swap picks and judge cost are stored next to the report
uv run python eval/run_eval.py --compare-models "<model a>,<model b>" --compare-topologies --with-judge
```

- `--out DIR` chooses the directory. A directory that already holds results is refused unless `--overwrite` is given.
- The headline is `pass^k`: the share of scenarios whose every repeat passed, with a 95% Wilson interval counted
  on scenarios. With `--repeats 1` it equals `pass@1` and says nothing about reliability; the report says so.
- A run that crashes inside the harness is stored as `harness_error`, left out of every rate, listed in the report,
  and makes the command exit with code 3. A provider failure stays `api_error`.
- Each run writes `manifest.json` (scenario set and its hash, harness commit, repeats, cohorts). To publish a run,
  move its directory under `campaigns/`.
