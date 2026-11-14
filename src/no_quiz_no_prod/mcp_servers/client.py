"""Client-side helpers: launch a server on stdio and decode its tool results into a payload or a typed error."""

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.types import CallToolResult, TextContent

from no_quiz_no_prod.contracts import DEFAULT_GIT_BRANCH, MCP_SERVER_MODULES, EnvVar, McpServerName, ToolErrorPayload

WRAPPED_RESULT_KEY = "result"  # MCPServer wraps a non-object return value (a list) as {"result": value}
# A launched server reads .env itself: without these, a GitHub remote and token configured there would be used.
HERMETIC_RELEASE_ENV = {
    EnvVar.NO_QUIZ_GIT_REMOTE_URL.value: "",
    EnvVar.NO_QUIZ_GITHUB_TOKEN.value: "",
    EnvVar.NO_QUIZ_GIT_BRANCH.value: DEFAULT_GIT_BRANCH,
}


def server_parameters(
    server: McpServerName, *, workspace_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> StdioServerParameters:
    """Launch parameters of one server under the current interpreter, bound to ``workspace_dir`` when given.

    ``env`` is merged over the SDK's default environment, e.g. ``{"NOTION_API_KEY": ""}`` keeps an eval run off the
    real Notion workspace (an empty value counts as unset). The release remote is hermetic by default: the URL and
    token are blanked (deploys go to the workspace's local bare remote) unless ``env`` sets them explicitly, since an
    empty variable wins over the ``.env`` the server loads.
    """
    extra = HERMETIC_RELEASE_ENV | dict(env or {})
    if workspace_dir is not None:
        extra[EnvVar.NO_QUIZ_WORKSPACE_DIR.value] = str(workspace_dir)
    return StdioServerParameters(command=sys.executable, args=["-m", MCP_SERVER_MODULES[server]], env=extra)


@dataclass(frozen=True)
class ToolOutcome:
    """One decoded tool result: the JSON payload, or the typed error."""

    payload: Any = None
    error: ToolErrorPayload | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def replayed(self) -> bool:
        """The server answered from its idempotency store: nothing happened twice."""
        return isinstance(self.payload, dict) and self.payload.get("replayed") is True


def decode_tool_result(result: CallToolResult) -> ToolOutcome:
    """Payload of a successful call (structured content, unwrapped), or its ``ToolErrorPayload``."""
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    if result.is_error:
        error = ToolErrorPayload.from_text(text)
        return ToolOutcome(error=error or ToolErrorPayload(code="internal_error", message=text or "tool failed"))
    structured = result.structured_content
    if isinstance(structured, dict) and structured.keys() == {WRAPPED_RESULT_KEY}:
        return ToolOutcome(payload=structured[WRAPPED_RESULT_KEY])
    if structured is not None:
        return ToolOutcome(payload=structured)
    try:
        return ToolOutcome(payload=json.loads(text))
    except ValueError:
        return ToolOutcome(payload=text)
