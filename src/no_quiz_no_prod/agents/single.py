"""The single-agent baseline: one ReAct agent, one shared conversation, Coder + Examiner tools.

``--topology single`` against ``dual`` isolates the effect of context separation: the same model, the same tools and
the same harness gates, but the documentation and the quiz are written by an agent whose context already holds every
file read, every test log and every rule page (more tokens, and more exposure to injected text).
"""

from typing import Any, Final

from no_quiz_no_prod.agents.coder import CODER_MAX_TOKENS, CODER_TOOLS, developer_request
from no_quiz_no_prod.agents.examiner import EXAMINER_TOOLS, GRADER_EXPLANATION_SOURCE, remediation_order
from no_quiz_no_prod.agents.prompts import SINGLE_SYSTEM_PROMPT, harness_message, untrusted_message
from no_quiz_no_prod.agents.tools import TOOLS, AgentSpec
from no_quiz_no_prod.contracts import QuizGradeResult

SINGLE_TOOLS: Final = tuple(dict.fromkeys((*CODER_TOOLS, *EXAMINER_TOOLS)))

SINGLE: Final = AgentSpec(
    role="single", system_prompt=SINGLE_SYSTEM_PROMPT, tools=SINGLE_TOOLS, max_tokens=CODER_MAX_TOKENS
)
# Remediation documents the SAME staged diff again (doc v2 + a new question): the code is frozen, so the single agent
# loses every tool that writes to staging or re-grants a lock. A write there would change the diff under the quiz.
SINGLE_REMEDIATION_TOOLS: Final = tuple(
    name for name in SINGLE_TOOLS if name in EXAMINER_TOOLS or not TOOLS[name].writes
)
SINGLE_REMEDIATION: Final = AgentSpec(
    role="single", system_prompt=SINGLE_SYSTEM_PROMPT, tools=SINGLE_REMEDIATION_TOOLS, max_tokens=CODER_MAX_TOKENS
)
CODE_FROZEN: Final = (
    "The staged code is frozen during remediation: you have no write or test tool until this quiz is answered."
)


def build_single_messages(task: str, memory_profile_block: str) -> list[dict[str, Any]]:
    """Fresh single-agent conversation: system prompt, the developer request, then the Layer 4 profile.

    The profile holds model-written text (past misconceptions): it travels as untrusted data in its own turn, never
    inside the trusted ``[harness]`` message that announces it.
    """
    return [
        {"role": "system", "content": SINGLE.system_prompt},
        developer_request(task),
        harness_message("The developer profile for the documentation and quiz phase follows as data."),
        untrusted_message("developer_profile", memory_profile_block),
    ]


def build_single_remediation_messages(grade: QuizGradeResult) -> list[dict[str, str]]:
    """The remediation order (trusted) then the grader's explanation (untrusted data), appended to the SAME
    conversation: the single agent keeps its whole history."""
    return [
        harness_message(f"{remediation_order(grade)}\n{CODE_FROZEN}"),
        untrusted_message(GRADER_EXPLANATION_SOURCE, grade.explanation),
    ]


__all__ = [
    "SINGLE",
    "SINGLE_REMEDIATION",
    "SINGLE_REMEDIATION_TOOLS",
    "SINGLE_TOOLS",
    "build_single_messages",
    "build_single_remediation_messages",
]
