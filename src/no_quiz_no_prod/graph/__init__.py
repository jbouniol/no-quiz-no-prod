"""LangGraph state machine of No-Quiz-No-Prod and its MCP tool bridge.

Import the runner from ``no_quiz_no_prod.graph.workflow`` (``NoQuizWorkflow``, ``WorkflowOutcome``); the names are
also re-exported here, lazily, so importing the package stays cheap.
"""

from typing import Any

_LAZY = {
    "NoQuizWorkflow": "no_quiz_no_prod.graph.workflow",
    "WorkflowOutcome": "no_quiz_no_prod.graph.workflow",
    "make_graph": "no_quiz_no_prod.graph.workflow",
    "MCPToolBridge": "no_quiz_no_prod.graph.mcp_bridge",
    "create_bridge": "no_quiz_no_prod.graph.mcp_bridge",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY)
