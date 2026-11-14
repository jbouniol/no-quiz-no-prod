"""Seed loading and adversarial planting: the pristine fintech modules, the rule pages and their injections.

A seed lives in ``seeds/<seed>/``: the five ``FintechModule`` files, plus ``rules/index.json`` (NotionRule metadata,
one ``body_file`` per rule) and the canonical rule pages it points to (the Markdown mirrored in the Notion
"Engineering Rules & Policies" database).
"""

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Final

from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.contracts import ALL_FINTECH_MODULES, PROJECT_ROOT, NotionRule

SEEDS_DIR: Final = PROJECT_ROOT / "seeds"
RULES_DIRNAME: Final = "rules"
RULES_INDEX_FILENAME: Final = "index.json"
RULE_BODY_FILE_KEY: Final = "body_file"
_SEED_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_FIRST_SECTION: Final = re.compile(r"^## ", re.MULTILINE)


def seed_path(seed: str) -> Path:
    """``seeds/<seed>/``, validated: a safe name holding the five fintech modules and a rules index."""
    path = SEEDS_DIR / seed
    complete = all((path / module).is_file() for module in ALL_FINTECH_MODULES)
    if not _SEED_NAME.match(seed) or not complete or not (path / RULES_DIRNAME / RULES_INDEX_FILENAME).is_file():
        raise WorkspaceError("invalid_argument", f"unknown or incomplete seed {seed!r} (expected seeds/{seed}/)")
    return path


def load_seed_rules(path: Path) -> list[NotionRule]:
    """The seed's rule pages as ``NotionRule``s, in index order."""
    rules_dir = path / RULES_DIRNAME
    rules: list[NotionRule] = []
    for entry in json.loads((rules_dir / RULES_INDEX_FILENAME).read_text(encoding="utf-8")):
        fields = {key: value for key, value in entry.items() if key != RULE_BODY_FILE_KEY}
        body = (rules_dir / entry[RULE_BODY_FILE_KEY]).read_text(encoding="utf-8")
        rules.append(NotionRule.model_validate({**fields, "body_markdown": body}))
    return rules


def _as_comment(line: str) -> str:
    if line.lstrip().startswith("#"):
        return line
    return f"# {line}" if line else "#"


def plant_code_injection(source: str, text: str) -> str:
    """``source`` with ``text`` planted as a comment block right after the module docstring.

    Each line of ``text`` is kept verbatim (lines that are not already comments get a ``# `` prefix), so the module
    stays valid Python and its tests keep passing, while the hostile text reaches any agent that reads the file.
    """
    comment = "".join(f"{_as_comment(line)}\n" for line in text.splitlines())
    body = ast.parse(source).body
    first = body[0] if body else None
    has_docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    insert_at = (first.end_lineno or 0) if has_docstring and first is not None else 0
    lines = source.splitlines(keepends=True)
    return "".join([*lines[:insert_at], "\n" if insert_at else "", comment, *lines[insert_at:]])


def plant_rule_injection(rule: NotionRule, text: str) -> NotionRule:
    """``rule`` with ``text`` inserted as a paragraph right after the rule statement (before the first section)."""
    body = rule.body_markdown
    match = _FIRST_SECTION.search(body)
    cut = match.start() if match else len(body)
    planted = f"{body[:cut].rstrip()}\n\n{text}\n\n{body[cut:]}".rstrip() + "\n"
    return NotionRule.model_validate({**rule.model_dump(), "body_markdown": planted})


def populate_area(destination: Path, seed_dir: Path, *, injection_target: str | None, injection_text: str) -> None:
    """Fill an empty area directory with the seed modules, planting the code injection when it targets one."""
    for module in ALL_FINTECH_MODULES:
        if module == injection_target:
            source = (seed_dir / module).read_text(encoding="utf-8")
            planted = plant_code_injection(source, injection_text)
            (destination / module).write_text(planted, encoding="utf-8", newline="")
        else:
            shutil.copyfile(seed_dir / module, destination / module)
