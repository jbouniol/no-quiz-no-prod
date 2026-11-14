"""The Coder: fixes the seeded fintech code in staging, guided by the Notion engineering rules.

Its toolset is exactly the eight tools below. ``read_audit_credentials`` is deliberately visible (adversarial
scenarios test real injection resistance, the prompt forbids it) and ``deploy_to_production`` is absent: the Coder
cannot even name the destructive tool. Promotion to prod belongs to the harness, behind the four SHA-256 locks.
"""

from typing import Any, Final

from no_quiz_no_prod.agents.prompts import CODER_SYSTEM_PROMPT
from no_quiz_no_prod.agents.tools import AGENT_MAX_OUTPUT_TOKENS, AgentSpec

CODER_TOOLS: Final = (
    "search_notion_rules",
    "read_notion_rule",
    "list_workspace_files",
    "read_code_file",
    "write_staging_file",
    "run_staging_tests",
    "get_staging_diff",
    "read_audit_credentials",
)
CODER_MAX_TOKENS: Final = AGENT_MAX_OUTPUT_TOKENS  # a complete module rewrite travels inside one tool call

CODER: Final = AgentSpec(
    role="coder", system_prompt=CODER_SYSTEM_PROMPT, tools=CODER_TOOLS, max_tokens=CODER_MAX_TOKENS
)


def developer_request(task: str, *, follow_up: bool = False) -> dict[str, str]:
    """The developer's words, verbatim: the only user instruction an agent receives besides harness checks."""
    label = "Developer follow-up" if follow_up else "Developer request"
    return {"role": "user", "content": f"{label}:\n{task.strip()}"}


def build_coder_messages(task: str) -> list[dict[str, Any]]:
    """Fresh Coder conversation: hardened system prompt + the developer request."""
    return [{"role": "system", "content": CODER.system_prompt}, developer_request(task)]


__all__ = ["CODER", "CODER_MAX_TOKENS", "CODER_TOOLS", "build_coder_messages", "developer_request"]
