"""Shared plumbing of both MCP servers: tool annotations, harness-owned arguments, typed errors, workspace binding."""

import functools
import logging
from typing import Annotated, Any, Final

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult, ToolAnnotations
from pydantic import Field, ValidationError

from no_quiz_no_prod import __version__
from no_quiz_no_prod.backend import SandboxWorkspace, WorkspaceError
from no_quiz_no_prod.contracts import McpServerName, RuntimeSettings, ToolErrorPayload

logger = logging.getLogger(__name__)

# ---- tool annotations (hints for the client: what a tool does to its environment) -------------------------------
READ_ONLY: Final = ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False, openWorldHint=False)
# Safe to repeat: the server replays a logical write by its idempotency key and never applies it twice.
IDEMPOTENT_WRITE: Final = ToolAnnotations(
    readOnlyHint=False, idempotentHint=True, destructiveHint=False, openWorldHint=False
)
DESTRUCTIVE_WRITE: Final = ToolAnnotations(
    readOnlyHint=False, idempotentHint=True, destructiveHint=True, openWorldHint=False
)

# ---- arguments the harness owns (the model should leave them alone) ----------------------------------------------
HARNESS_ARG_MAX_CHARS: Final = 128
IdempotencyKey = Annotated[
    str | None,
    Field(max_length=HARNESS_ARG_MAX_CHARS, description="Set by the harness on writes; never set it yourself."),
]
ThreadId = Annotated[
    str,
    Field(min_length=1, max_length=HARNESS_ARG_MAX_CHARS, description="Set by the harness; leave the default."),
]


def _validation_problems(error: ValidationError) -> tuple[list[str], str]:
    fields = sorted({".".join(str(part) for part in item["loc"]) or "arguments" for item in error.errors()})
    summary = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'arguments'}: {item['msg']}" for item in error.errors()
    )
    return fields, summary


def invalid_argument(error: ValidationError, what: str) -> WorkspaceError:
    """A model-facing ``invalid_argument`` error listing the offending fields of a pydantic ``ValidationError``."""
    fields, summary = _validation_problems(error)
    return WorkspaceError("invalid_argument", f"Invalid {what}: {summary}", details={"fields": fields})


def typed_error(tool: str, error: ToolError) -> ToolErrorPayload:
    """The ``ToolErrorPayload`` of any tool failure: backend refusal, bad arguments, unknown tool or crash."""
    cause = error.__cause__
    if isinstance(cause, WorkspaceError):
        return cause.payload
    if isinstance(cause, ValidationError) and not isinstance(error, UnexpectedToolError):
        return invalid_argument(cause, f"arguments for {tool}").payload
    if (payload := ToolErrorPayload.from_text(str(error))) is not None:
        return payload
    if isinstance(error, UnexpectedToolError):
        logger.error("tool %s crashed", tool, exc_info=cause)  # the traceback stays in the server log
        return ToolErrorPayload(code="internal_error", message=f"{tool} failed unexpectedly: report it, do not retry")
    return ToolErrorPayload(code="invalid_argument", message=str(error))  # e.g. an unknown tool name


class NoQuizMCPServer(MCPServer):
    """``MCPServer`` whose every tool failure reaches the client as a ``ToolErrorPayload`` JSON (Session 2 lab).

    ``message`` is for the model; ``retryable`` / ``outcome_unknown`` / ``details`` are for the harness, which
    recovers the payload with ``ToolErrorPayload.from_text``.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context[Any, Any] | None = None
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as error:
            raise ToolError(typed_error(name, error).model_dump_json()) from error.__cause__


def create_server(name: McpServerName, instructions: str) -> NoQuizMCPServer:
    """A named, versioned server of this project (the version shows up in the client's ``server_info``)."""
    return NoQuizMCPServer(name, instructions=instructions, version=__version__)


class WorkspaceBinding:
    """The ``SandboxWorkspace`` a server works on: fixed ``settings``, or ``RuntimeSettings.from_env()`` per call.

    Resolving per call lets an in-process caller retarget a module-level server by setting ``NO_QUIZ_WORKSPACE_DIR``
    before a call; a spawned stdio server simply inherits it from its environment.
    """

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self._settings = settings

    def __call__(self) -> SandboxWorkspace:
        return _workspace_for(self._settings or RuntimeSettings.from_env())


@functools.lru_cache(maxsize=16)
def _workspace_for(settings: RuntimeSettings) -> SandboxWorkspace:
    return SandboxWorkspace(settings)
