"""The single-agent baseline: one ReAct agent, one shared conversation, Coder + Examiner tools.

``--topology single`` against ``dual`` isolates the effect of context separation: the same model, the same tools and
the same harness gates, but the documentation and the quiz are written by an agent whose context already holds every
file read, every test log and every rule page (more tokens, and more exposure to injected text).
"""

from typing import Any, Final

from no_quiz_no_prod.agents.coder import CODER_MAX_TOKENS, CODER_TOOLS, developer_request
from no_quiz_no_prod.agents.examiner import EXAMINER_TOOLS, remediation_brief
from no_quiz_no_prod.agents.prompts import SINGLE_SYSTEM_PROMPT, harness_message
from no_quiz_no_prod.agents.tools import AgentSpec
from no_quiz_no_prod.contracts import QuizGradeResult

SINGLE_TOOLS: Final = tuple(dict.fromkeys((*CODER_TOOLS, *EXAMINER_TOOLS)))

SINGLE: Final = AgentSpec(
    role="single", system_prompt=SINGLE_SYSTEM_PROMPT, tools=SINGLE_TOOLS, max_tokens=CODER_MAX_TOKENS
)


def build_single_messages(task: str, memory_profile_block: str) -> list[dict[str, Any]]:
    """Fresh single-agent conversation: system prompt, the developer request, then the Layer 4 profile."""
    profile = harness_message(
        f"Developer profile for the documentation and quiz phase:\n\n{memory_profile_block.strip()}"
    )
    return [{"role": "system", "content": SINGLE.system_prompt}, developer_request(task), profile]


def build_single_remediation_message(grade: QuizGradeResult) -> dict[str, str]:
    """The remediation order, appended to the SAME conversation (the single agent keeps its whole history)."""
    return harness_message(remediation_brief(grade))


__all__ = ["SINGLE", "SINGLE_TOOLS", "build_single_messages", "build_single_remediation_message"]
