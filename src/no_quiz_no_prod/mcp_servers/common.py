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
from no_quiz_no_prod.contracts import McpServerName, RuntimeSettings, ToolErrorPayload, project_environment

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
UNKNOWN_TOOL_PREFIX: Final = "Unknown tool"  # the SDK's message for a tool name no server registered
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
    if str(error).startswith(UNKNOWN_TOOL_PREFIX):
        return ToolErrorPayload(code="not_found", message=f"{tool!r} is not a tool of this server")
    # Anything else is a failure of the server itself (a crash in a tool, or an SDK error that names no argument):
    # never labelled as the caller's mistake. The traceback stays in the server log.
    logger.error("tool %s failed without a typed error", tool, exc_info=cause or error)
    return ToolErrorPayload(code="internal_error", message=f"{tool} failed unexpectedly: report it, do not retry")


class NoQuizMCPServer(MCPServer):
    """``MCPServer`` whose every tool failure reaches the client as a ``ToolErrorPayload`` JSON.

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


def server_settings(server: McpServerName) -> RuntimeSettings:
    """Settings of a launched server, read at call time and reduced to what that server may hold.

    They come from ``project_environment()`` (the project ``.env`` overlaid by the process environment), which
    exports nothing to ``os.environ``: a secret never enters the environment a test child could inherit or read
    back. ``restricted_to`` then drops every secret outside the server's allow-list.
    """
    return RuntimeSettings.from_env(project_environment()).restricted_to(server)


class WorkspaceBinding:
    """The ``SandboxWorkspace`` one server works on, under that server's least-privilege settings.

    Fixed ``settings`` (an in-process server), or ``server_settings(server)`` per call: resolving per call lets an
    in-process caller retarget a module-level server by setting ``NO_QUIZ_WORKSPACE_DIR`` before a call, and follows
    a ``.env`` rewritten while the server runs; a spawned stdio server gets its workspace from its environment.
    """

    def __init__(self, server: McpServerName, settings: RuntimeSettings | None = None) -> None:
        self._server = server
        self._settings = settings.restricted_to(server) if settings is not None else None

    def __call__(self) -> SandboxWorkspace:
        return _workspace_for(self._settings or server_settings(self._server))


@functools.lru_cache(maxsize=16)
def _workspace_for(settings: RuntimeSettings) -> SandboxWorkspace:
    return SandboxWorkspace(settings)
