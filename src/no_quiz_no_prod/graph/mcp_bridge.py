"""MCPToolBridge: one synchronous facade over the two MCP servers of #14 §3, whatever the transport.

- ``"stdio"`` (production, eval): spawns ``python -m <MCP_SERVER_MODULES[server]>`` for code-mcp and notion-quiz-mcp
  with ``mcp.client.stdio.stdio_client`` and ``NO_QUIZ_WORKSPACE_DIR=<settings.workspace_dir>``, so parallel eval runs
  stay isolated. Server stderr goes to ``<workspace_dir>/mcp_servers.log``.
- ``"inprocess"`` with an injected ``ToolDispatcher`` (tests): ``dispatcher(server, tool, args)`` returns a result or
  raises ``ToolCallError(ToolErrorPayload)``. Fully deterministic, no subprocess, no MCP server needed.
- ``"inprocess"`` without a dispatcher: the real server objects (``build_server(settings)`` when the server module
  offers it) over the SDK in-memory transport, ``mcp.Client(<MCPServer>)``.

The async MCP sessions live on one private event-loop thread; graph nodes stay synchronous and call ``call()``
from any thread. Every failure comes back as a typed ``ToolErrorPayload`` (never an exception), so the harness can
retry what is retryable and the model can read the rest.

Least privilege: each stdio server gets ``settings.environment_for(server)`` and nothing else of the project: the
release token and remote only reach the server that publishes, the Notion key only the Notion server, no LLM key
reaches either. Lifecycle: a server that died is reconnected once and the call repeated (writes carry their
idempotency key, so a repeat is a replay). The deploy call waits longer than the server's own release deadline.
"""

import asyncio
import concurrent.futures
import importlib
import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Literal, Self

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver import MCPServer
from mcp_types import CONNECTION_CLOSED
from pydantic_core import to_jsonable_python

from no_quiz_no_prod.agents.tools import server_for
from no_quiz_no_prod.contracts import (
    ALL_MCP_SERVERS,
    MCP_CALL_TIMEOUT_S,
    MCP_CONNECT_TIMEOUT_S,
    MCP_DEPLOY_CALL_TIMEOUT_S,
    MCP_SERVER_MODULES,
    PROJECT_ROOT,
    EnvVar,
    McpServerName,
    RuntimeSettings,
    ToolErrorPayload,
)

logger = logging.getLogger(__name__)

McpTransport = Literal["stdio", "inprocess"]
ToolDispatcher = Callable[[McpServerName, str, dict[str, Any]], Any]

MCP_SHUTDOWN_TIMEOUT_S: Final = 10.0
MCP_SERVER_LOG_FILENAME: Final = "mcp_servers.log"
# How long the bridge waits for one call: the deploy runs a whole git release server-side, so its budget is above
# the server's own deadline (a bridge timeout there would report an unknown outcome for a deploy still running).
CALL_TIMEOUT_BY_TOOL: Final[Mapping[str, float]] = {"deploy_to_production": MCP_DEPLOY_CALL_TIMEOUT_S}
# What a dead server looks like from the client: its streams are closed or broken.
CONNECTION_LOST_ERRORS: Final = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    ConnectionError,
)
MAX_ERROR_TEXT: Final = 500


@dataclass(frozen=True)
class ToolOutcome:
    """The answer to one tool call: a decoded result, or a typed error."""

    result: Any = None
    error: ToolErrorPayload | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def replayed(self) -> bool:
        """The server answered from its idempotency store: nothing happened twice."""
        return isinstance(self.result, Mapping) and self.result.get("replayed") is True


class ToolCallError(Exception):
    """Raised by an injected dispatcher to answer a call with a typed ``ToolErrorPayload``."""

    def __init__(self, payload: ToolErrorPayload) -> None:
        super().__init__(payload.message)
        self.payload = payload


def error_from_exception(exc: BaseException) -> ToolErrorPayload:
    """Typed payload of a failed call: the server's own payload when the error text carries one."""
    if isinstance(exc, ToolCallError):
        return exc.payload
    if isinstance(exc, TimeoutError | concurrent.futures.TimeoutError):
        return ToolErrorPayload(code="timeout", message="no answer from the MCP server in time", outcome_unknown=True)
    text = str(exc)
    return ToolErrorPayload.from_text(text) or ToolErrorPayload(
        code="internal_error", message=f"{type(exc).__name__}: {text}"[:MAX_ERROR_TEXT]
    )


def call_timeout_s(tool: str) -> float:
    """Seconds the bridge waits for ``tool`` (see ``CALL_TIMEOUT_BY_TOOL``)."""
    return CALL_TIMEOUT_BY_TOOL.get(tool, MCP_CALL_TIMEOUT_S)


def connection_lost(exc: BaseException) -> bool:
    """True when ``exc`` means the server is gone (process died, pipe closed), not that the tool failed."""
    return isinstance(exc, CONNECTION_LOST_ERRORS) or getattr(exc, "code", None) == CONNECTION_CLOSED


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def decode_call_result(result: Any) -> ToolOutcome:
    """``CallToolResult`` -> ``ToolOutcome``: structured content first (a wrapped ``{"result": [...]}`` list is
    unwrapped), else the JSON text blocks; an error result becomes its ``ToolErrorPayload``."""
    texts = [block.text for block in result.content if getattr(block, "type", "") == "text"]
    if result.is_error:
        text = "\n".join(texts) or "the tool reported an error without a message"
        payload = ToolErrorPayload.from_text(text)
        return ToolOutcome(error=payload or ToolErrorPayload(code="internal_error", message=text[:MAX_ERROR_TEXT]))
    structured = result.structured_content
    if structured is not None:
        if isinstance(structured, Mapping) and set(structured) == {"result"}:
            return ToolOutcome(result=structured["result"])
        return ToolOutcome(result=structured)
    values = [_maybe_json(text) for text in texts]
    return ToolOutcome(result=values[0] if len(values) == 1 else values)


class MCPToolBridge(ABC):
    """Synchronous facade over the MCP servers. ``call`` never raises for a tool-level failure."""

    transport: McpTransport

    @abstractmethod
    def _call(self, server: McpServerName, tool: str, args: dict[str, Any]) -> ToolOutcome: ...

    def call(self, tool: str, args: Mapping[str, Any], *, server: McpServerName | None = None) -> ToolOutcome:
        """Call ``tool`` on its server (routing from the tool catalog unless ``server`` is given)."""
        try:
            target = server or server_for(tool)
        except KeyError:
            return ToolOutcome(error=ToolErrorPayload(code="not_found", message=f"unknown tool {tool!r}"))
        try:
            return self._call(target, tool, dict(args))
        except Exception as exc:  # the model must see a typed error, never a crash
            return ToolOutcome(error=error_from_exception(exc))

    def restart(self) -> None:
        """Drop live server connections (the next call reconnects): used after a workspace reset."""
        return None  # stateless transports hold no connection

    def close(self) -> None:
        """Release every resource (idempotent)."""
        return None  # stateless transports hold no resource

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DispatcherBridge(MCPToolBridge):
    """``"inprocess"`` transport around an injected dispatcher (deterministic tests, stubs).

    Results are normalized to JSON-compatible values, exactly what the stdio transport would deliver."""

    transport: McpTransport = "inprocess"

    def __init__(self, dispatcher: ToolDispatcher) -> None:
        self._dispatcher = dispatcher

    def _call(self, server: McpServerName, tool: str, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(result=to_jsonable_python(self._dispatcher(server, tool, args)))


class _SessionBridge(MCPToolBridge):
    """Owns one event-loop thread and one MCP session per server; sessions open lazily and reopen after restart."""

    def __init__(self, settings: RuntimeSettings, servers: Sequence[McpServerName] = ALL_MCP_SERVERS) -> None:
        self.settings = settings
        self._servers = tuple(servers)
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._owner: concurrent.futures.Future[None] | None = None
        self._stop: asyncio.Event | None = None
        self._sessions: dict[McpServerName, Any] = {}

    @abstractmethod
    async def _open(self, stack: AsyncExitStack, server: McpServerName) -> Any:
        """Enter one connected session (anything with an async ``call_tool(name, arguments)``) on ``stack``."""

    # -- lifecycle ---------------------------------------------------------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-bridge", daemon=True)
            self._thread.start()
        return self._loop

    async def _own_sessions(self, ready: concurrent.futures.Future[bool]) -> None:
        """Opens, holds and closes every session in ONE task (anyio cancel scopes demand it)."""
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                sessions = {server: await self._open(stack, server) for server in self._servers}
                self._sessions = sessions
                ready.set_result(True)
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP sessions closed with an error: %s", exc)
        finally:
            self._sessions = {}

    def _ensure_connected(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            loop = self._ensure_loop()
            if not self._sessions:
                ready: concurrent.futures.Future[bool] = concurrent.futures.Future()
                self._owner = asyncio.run_coroutine_threadsafe(self._own_sessions(ready), loop)
                ready.result(timeout=MCP_CONNECT_TIMEOUT_S)
            return loop

    def _call(self, server: McpServerName, tool: str, args: dict[str, Any]) -> ToolOutcome:
        """One call; a server found dead is reconnected ONCE and the same call repeated.

        The repeat is safe: reads are pure and every write carries its idempotency key in ``args``, so a write the
        dead server did complete is replayed, not redone. A timeout is never repeated here (the server is alive
        and may still be working): it surfaces as ``timeout`` with ``outcome_unknown``.
        """
        try:
            return self._call_once(server, tool, args)
        except Exception as exc:
            if not connection_lost(exc):
                raise
            logger.warning("MCP server %s is gone (%s): reconnecting once", server, type(exc).__name__)
        self.restart()
        return self._call_once(server, tool, args)

    def _call_once(self, server: McpServerName, tool: str, args: dict[str, Any]) -> ToolOutcome:
        with self._lock:
            loop = self._ensure_connected()
            session = self._sessions[server]
        future = asyncio.run_coroutine_threadsafe(session.call_tool(tool, args), loop)
        try:
            return decode_call_result(future.result(timeout=call_timeout_s(tool)))
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

    def list_tool_names(self) -> dict[McpServerName, list[str]]:
        """Tool names each server advertises (diagnostics: compare with the catalog)."""
        loop = self._ensure_connected()

        async def names(session: Any) -> list[str]:
            return [tool.name for tool in (await session.list_tools()).tools]

        return {
            server: asyncio.run_coroutine_threadsafe(names(session), loop).result(timeout=MCP_CALL_TIMEOUT_S)
            for server, session in self._sessions.items()
        }

    def restart(self) -> None:
        with self._lock:
            if self._loop is not None and self._stop is not None and self._owner is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
                try:
                    self._owner.result(timeout=MCP_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:  # a server that dies on shutdown must not block the next run
                    logger.warning("MCP shutdown: %s", exc)
            self._owner, self._stop, self._sessions = None, None, {}

    def close(self) -> None:
        with self._lock:
            self.restart()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=MCP_SHUTDOWN_TIMEOUT_S)
                self._loop.close()
            self._loop, self._thread = None, None


class StdioMCPBridge(_SessionBridge):
    """``"stdio"`` transport: the real MCP servers as subprocesses (``mcp.client.stdio.stdio_client``)."""

    transport: McpTransport = "stdio"

    def server_parameters(self, server: McpServerName) -> StdioServerParameters:
        """``python -m <module>`` bound to this run's workspace, with a least-privilege environment.

        ``settings.environment_for(server)`` gives each server only its allow-listed variables and sets every other
        project variable EMPTY, which also stops the server from reading it back from ``.env``: no LLM key for
        either server, the release remote and token only for the one that publishes, the Notion key only for the
        Notion server. Values always follow the harness settings (e.g. updated by the Studio), never the parent's
        environment; the token travels in the child's environment only, never in argv. ``NO_QUIZ_ENV_FILE`` (which
        ``.env`` to load, not a secret) is inherited when set.
        """
        env = self.settings.environment_for(server)
        if env_file := os.environ.get(EnvVar.NO_QUIZ_ENV_FILE.value):
            env[EnvVar.NO_QUIZ_ENV_FILE.value] = env_file
        return StdioServerParameters(
            command=sys.executable, args=["-m", MCP_SERVER_MODULES[server]], env=env, cwd=str(PROJECT_ROOT)
        )

    async def _open(self, stack: AsyncExitStack, server: McpServerName) -> ClientSession:
        self.settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        errlog = stack.enter_context(
            (self.settings.workspace_dir / MCP_SERVER_LOG_FILENAME).open("a", encoding="utf-8")
        )
        read, write = await stack.enter_async_context(stdio_client(self.server_parameters(server), errlog=errlog))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session


def find_mcp_server(module: ModuleType, settings: RuntimeSettings) -> MCPServer:
    """The server object of a server module: ``build_server(settings)`` (bound to this workspace) when the module
    offers it, else its module-level ``MCPServer`` (``mcp`` / ``server`` / ``app`` first, then any)."""
    factory = getattr(module, "build_server", None)
    if callable(factory) and isinstance(built := factory(settings), MCPServer):
        return built
    os.environ[EnvVar.NO_QUIZ_WORKSPACE_DIR.value] = str(settings.workspace_dir)  # module-level servers read env
    candidates = [getattr(module, name, None) for name in ("mcp", "server", "app")] + list(vars(module).values())
    for candidate in candidates:
        if isinstance(candidate, MCPServer):
            return candidate
    raise LookupError(f"{module.__name__} exposes no MCPServer instance")


class InMemoryMCPBridge(_SessionBridge):
    """``"inprocess"`` transport without a dispatcher: the real server objects over the SDK in-memory transport."""

    transport: McpTransport = "inprocess"

    async def _open(self, stack: AsyncExitStack, server: McpServerName) -> Any:
        from mcp import Client  # local: only this transport needs the high-level client

        instance = find_mcp_server(importlib.import_module(MCP_SERVER_MODULES[server]), self.settings)
        return await stack.enter_async_context(Client(instance))


def create_bridge(
    transport: McpTransport, settings: RuntimeSettings, *, dispatcher: ToolDispatcher | None = None
) -> MCPToolBridge:
    """The bridge for ``transport``; ``dispatcher`` (inprocess only) replaces the MCP servers entirely."""
    if transport == "stdio":
        if dispatcher is not None:
            raise ValueError("a tool dispatcher only applies to the 'inprocess' transport")
        return StdioMCPBridge(settings)
    if transport == "inprocess":
        return DispatcherBridge(dispatcher) if dispatcher is not None else InMemoryMCPBridge(settings)
    raise ValueError(f"unknown MCP transport {transport!r}: expected 'stdio' or 'inprocess'")


__all__ = [
    "CALL_TIMEOUT_BY_TOOL",
    "MCP_SERVER_LOG_FILENAME",
    "DispatcherBridge",
    "InMemoryMCPBridge",
    "MCPToolBridge",
    "McpTransport",
    "StdioMCPBridge",
    "ToolCallError",
    "ToolDispatcher",
    "ToolOutcome",
    "call_timeout_s",
    "connection_lost",
    "create_bridge",
    "decode_call_result",
    "error_from_exception",
    "find_mcp_server",
]
