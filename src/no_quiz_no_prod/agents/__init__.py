"""LLM agents of No-Quiz-No-Prod: the Coder, the Pedagogical Examiner and the single-agent baseline.

``AGENTS`` maps each LLM ``AgentRole`` to its spec (system prompt, visible tools, output budget). The import-time
guard proves no agent can see a harness-only tool (``deploy_to_production``).
"""

from collections.abc import Mapping
from typing import Final

from no_quiz_no_prod.agents.coder import CODER, CODER_TOOLS, build_coder_messages, developer_request
from no_quiz_no_prod.agents.examiner import (
    EXAMINER,
    DiffTooLargeError,
    EXAMINER_TOOLS,
    bind_quiz_args,
    bind_ship_log_args,
    build_examiner_brief,
    build_examiner_messages,
    missing_deliverables,
    render_applied_rules,
)
from no_quiz_no_prod.agents.prompts import (
    AgentStatus,
    harness_message,
    parse_agent_status,
    render_tool_result,
    strip_status_line,
)
from no_quiz_no_prod.agents.single import (
    SINGLE,
    SINGLE_REMEDIATION,
    SINGLE_TOOLS,
    build_single_messages,
    build_single_remediation_messages,
)
from no_quiz_no_prod.agents.tools import (
    HARNESS_ONLY_TOOLS,
    SEALED,
    TOOLS,
    AgentSpec,
    ToolDef,
    check_toolsets,
    server_for,
)
from no_quiz_no_prod.contracts import AgentRole

AGENTS: Final[Mapping[AgentRole, AgentSpec]] = {spec.role: spec for spec in (CODER, EXAMINER, SINGLE)}

check_toolsets((*AGENTS.values(), SINGLE_REMEDIATION))

__all__ = [
    "AGENTS",
    "CODER",
    "CODER_TOOLS",
    "EXAMINER",
    "EXAMINER_TOOLS",
    "HARNESS_ONLY_TOOLS",
    "SEALED",
    "SINGLE",
    "SINGLE_REMEDIATION",
    "SINGLE_TOOLS",
    "TOOLS",
    "AgentSpec",
    "AgentStatus",
    "DiffTooLargeError",
    "ToolDef",
    "bind_quiz_args",
    "bind_ship_log_args",
    "build_coder_messages",
    "build_examiner_brief",
    "build_examiner_messages",
    "build_single_messages",
    "build_single_remediation_messages",
    "developer_request",
    "harness_message",
    "missing_deliverables",
    "parse_agent_status",
    "render_applied_rules",
    "render_tool_result",
    "server_for",
    "strip_status_line",
]
