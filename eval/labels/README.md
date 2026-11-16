# Labels for judge calibration

The LLM judge is only worth reading once it agrees with people. This directory holds what is needed to check that.

## What exists today

`synthetic_draft_labels.json` (formerly `eval/golden_human_labels.json`): 20 made-up runs with draft labels and two
recorded verdicts, all written by the harness author. **Nobody labelled them.** They are a fixture for offline
tests and for `--dry-run --with-judge`. A kappa computed against them is not a human-agreement figure, and every
report says so. The keys `human_acceptable` and `human_rationale` inside the file are historical names.

`human_labels.json`: **40 unlabelled records** (10 per scenario kind) exported from campaign 04,
`04_dev_models_x_topologies/results_dual_deepseek_deepseek_v4_flash_latest.jsonl`. Nobody has filled them in yet,
so there are **no human labels yet**: until the file is labelled and the judge reaches the bar, reports print
"Judge validated against human labels: no".

`label_tool.html`: a single local page to fill that file in (no server, no network). Open it in Chrome, click
"Open human_labels.json": Chrome saves back in place (`⌘S`); other browsers download a copy. Each record
shows the task, the tool trace and the final message; answer the four criteria with `1`–`4` then `y`/`n`, write a
rationale, move with `←`/`→`. `human_acceptable` is computed from the criteria. Unsaved work is kept in the
browser per record.

## Producing real labels

1. Export a batch of real runs. Nothing is pre-filled:

   ```bash
   uv run python eval/run_eval.py \
     --export-labeling-batch eval/runs/<run>/results.jsonl \
     --labeling-out eval/labels/human_labels.json --labeling-count 40
   ```

   The sample is deterministic, balanced over the four scenario kinds, and takes failed runs first so the batch is
   not all "accept". Use runs of the current harness: the recorded campaigns were made with the old prompts.

2. A person fills in the file (schema below), ideally someone who did not write the prompts.

3. Calibrate the live judge against it:

   ```bash
   uv run python eval/run_eval.py --with-judge --human-labels eval/labels/human_labels.json
   ```

The judge counts as validated when it is compared with at least 20 human-labelled runs and Cohen's kappa is at
least 0.60 (`MIN_HUMAN_LABELS`, `JUDGE_KAPPA_BAR` in `src/no_quiz_no_prod/eval/judge.py`).

## Schema of a human label file

```json
{
  "meta": {
    "schema_version": 1,
    "provenance": "human",
    "rubric": {"<criterion>": "<what it means>"},
    "sample_seed": 0,
    "instructions": "..."
  },
  "records": [
    {
      "record_id": "hl_001",
      "source": "eval/runs/<run>/results.jsonl",
      "run_key": "n03#1",
      "record_sha256": "<sha256 of record_text>",
      "record_text": "<exactly what the judge reads>",
      "human_acceptable": null,
      "criteria": {"<criterion>": null},
      "rationale": "",
      "labeler": "",
      "labeled_at": null
    }
  ]
}
```

| Field | Filled by | Rule |
| :--- | :--- | :--- |
| `record_id`, `source`, `run_key`, `record_sha256`, `record_text` | the export | Do not edit. A changed `record_text` no longer matches `record_sha256` and the file is refused. |
| `criteria` | the labeller | `true` or `false` for each criterion of `meta.rubric`, judged from `record_text` only. |
| `human_acceptable` | the labeller | `true` only if every criterion is `true`. |
| `rationale` | the labeller | One sentence. |
| `labeler`, `labeled_at` | the labeller | Name or initials, ISO date. A record with an empty `labeler` counts as not labelled. |

`record_text` is the judge's own input: the task, the tool trace, the workspace summary and the final message. It
does not name the scenario kind or the expected state, so the labeller and the judge decide from the same evidence.
