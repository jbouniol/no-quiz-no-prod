"""CLI entry point of the No-Quiz-No-Prod evaluation harness (see eval/README.md).

Usage:
    uv run python eval/run_eval.py --dry-run
    uv run python eval/run_eval.py --scenario-set heldout
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from no_quiz_no_prod.eval.runner import build_arg_parser, main, run_cohort, simulate_dry_run_scenario  # noqa: E402

__all__ = ["build_arg_parser", "main", "run_cohort", "simulate_dry_run_scenario"]


if __name__ == "__main__":
    raise SystemExit(main())
