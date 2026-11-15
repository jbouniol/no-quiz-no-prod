# Evaluation data

Everything the eval harness reads and writes. The paths are defined once, in `src/no_quiz_no_prod/eval/paths.py`.

| Path | What it is |
| :--- | :--- |
| `scenarios.jsonl` | **Development set** (38 scenarios). The prompts and the harness were tuned on it: a rate measured on it is optimistic. |
| `scenarios_heldout.jsonl` | **Held-out set** (14 scenarios). Never use it to tune a prompt, a budget or the harness. Run it with `--scenario-set heldout`. |
| `labels/` | The label schema for judge calibration, and the synthetic draft labels (an offline fixture, not human labels). See `labels/README.md`. |
| `campaigns/<name>/` | Recorded live campaigns. Each has a `manifest.json`. All three are **stale** (see below). |
| `runs/<utc timestamp>_<slug>/` | Where a new run writes by default. Created on demand. |
| `run_eval.py` | The command-line entry point. |

## No result describes the current code

No live run was made after the fix phase, so there is **no measured number for the current prompts, scenarios and
scoring**. The three recorded campaigns were produced with the earlier system prompts (which listed the scenario
classes), on earlier scenario files, before the stop reason was scored. Their files were moved, not edited.

| Campaign | Content | Scenarios × repeats | Judged |
| :--- | :--- | :---: | :--- |
| `campaigns/01_pre_fix_dual_deepseek/` | First baseline, DeepSeek dual | 30 × 1 | no |
| `campaigns/02_benchmark_pre_git_release/` | 2 models × 2 topologies, `comparison.md` | 30 × 1 | yes, verdicts not stored |
| `campaigns/03_git_release_dual_deepseek/` | DeepSeek dual after deploys became git releases | 31 × 1 | no |

Do not compare these campaigns with each other or with a new run: campaigns 01 and 02 never observed the release
push, none of them scored the stop reason, and each has one repeat per scenario, so a difference of a few scenarios
between two of them is within run-to-run noise. The `summary.md` and `comparison.md` files inside a campaign are
the reports generated at the time; they mention old paths such as `eval/golden_human_labels.json`.

## What changed in the scenario sets

- `n15` was a reworded copy of `n01`. It is now a distinct task (unknown VIP tier), still under `push_rejected`.
- Added to the development set: `n16` (every push refused, expected end `release_failed`), `n17` (a write fault the
  model can see), `n18` (looks under-specified, the rule fixes the value: asking is a failure), `n19` (looks
  forbidden, is allowed: refusing is a failure), `am06`, `im06`, `ad07`.
- The held-out set is new. Its scenarios, and the eight above, have **never been run against a model**. Their step
  budgets were set by analogy with measured scenarios and may need a first calibration run.
- `expect_stop_reason` is now scored: the right state with the wrong stop reason is `wrong_stop`, not a pass.

## Running

```bash
# Offline check of contracts and reports (no LLM call), written to a new eval/runs/ directory
uv run python eval/run_eval.py --dry-run

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
