"""Typed backend failures: one exception carrying the frozen ``ToolErrorPayload``.

The sandbox backend raises ``WorkspaceError`` for every anticipated refusal (gate locked, stale hash, quiz exhausted,
injected fault...). The MCP servers forward ``error.payload`` verbatim as their tool error, and the harness (graph, UI,
CLI) can map ``error.code`` to its own transport (HTTP status, stop reason) without parsing any text.
"""

from collections.abc import Mapping
from typing import Any

from no_quiz_no_prod.contracts import ToolErrorCode, ToolErrorPayload


class WorkspaceError(Exception):
    """An anticipated refusal or failure of the sandbox backend, with its typed payload."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        retryable: bool = False,
        outcome_unknown: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.payload = ToolErrorPayload(
            code=code,
            message=message,
            retryable=retryable,
            outcome_unknown=outcome_unknown,
            details=dict(details or {}),
        )
        super().__init__(message)

    @property
    def code(self) -> ToolErrorCode:
        return self.payload.code
