"""The MCP tool catalog of the frozen contract (#14 §3), as the agents see it.

One ``ToolDef`` per MCP tool: the server that serves it, whether it writes, the LLM-facing JSON schema, and the
*harness-owned* arguments. A harness-owned argument (``idempotency_key``, ``thread_id``, the diff hash, the attempt,
the doc version...) is never shown to a model and never trusted from one: the harness injects the authoritative value
on every call. ``deploy_to_production`` is harness-only: no agent toolset may contain it (guarded at import).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from no_quiz_no_prod.contracts import (
    ALL_FINTECH_MODULES,
    ALL_QUIZ_OPTION_KEYS,
    ALL_SKILL_TAGS,
    ALL_WORKSPACE_AREAS,
    AgentRole,
    McpServerName,
)
from no_quiz_no_prod.llm import function_tool

ToolKind = Literal["read", "write", "destructive"]

IDEMPOTENCY_ARG: Final = "idempotency_key"
THREAD_ARG: Final = "thread_id"
RULE_CODE_PATTERN: Final = r"^RULE-\d{3}$"
QUIZ_OPTION_ARGS: Final = tuple(f"option_{key.lower()}" for key in ALL_QUIZ_OPTION_KEYS)
QUIZ_EXPLANATION_ARGS: Final = tuple(f"explanation_{key.lower()}" for key in ALL_QUIZ_OPTION_KEYS)
SEALED: Final = "[sealed: the answer key stays server-side]"


def _text(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _choice(values: Sequence[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


def _list_of(item: Mapping[str, Any], description: str) -> dict[str, Any]:
    return {"type": "array", "items": dict(item), "description": description}


_RULE_CODE: Final = {"type": "string", "pattern": RULE_CODE_PATTERN, "description": "Rule code, e.g. RULE-001."}
_MODULE: Final = _choice(ALL_FINTECH_MODULES, "Module of the seeded fintech codebase.")
_AREA: Final = _choice(ALL_WORKSPACE_AREAS, "Workspace area (default: staging).")
_SKILL: Final = _choice(ALL_SKILL_TAGS, "Developer skill tag.")


@dataclass(frozen=True)
class ToolDef:
    """One MCP tool: routing (server), safety class (kind) and the schema a model may fill."""

    name: str
    server: McpServerName
    kind: ToolKind
    description: str
    parameters: Mapping[str, Mapping[str, Any]]
    required: tuple[str, ...] = ()
    harness_args: tuple[str, ...] = ()  # injected by the harness, hidden from (and overridden over) the model
    sealed_args: tuple[str, ...] = ()  # sent to the server, never traced or streamed (the quiz answer key)

    @property
    def writes(self) -> bool:
        """Writes get a harness idempotency key and a bounded retry with that same key."""
        return self.kind != "read"

    def traced_args(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Arguments safe to trace, stream to the UI and store in eval results."""
        return {key: SEALED if key in self.sealed_args else value for key, value in args.items()}

    def spec(self) -> dict[str, Any]:
        """OpenAI function-tool spec: model-facing arguments only."""
        schema = {
            "type": "object",
            "properties": {name: dict(prop) for name, prop in self.parameters.items()},
            "required": list(self.required),
            "additionalProperties": False,
        }
        return function_tool(self.name, self.description, schema)


TOOLS: Final[Mapping[str, ToolDef]] = {
    tool.name: tool
    for tool in (
        # ---- notion-quiz-mcp ----------------------------------------------------------------------------------------
        ToolDef(
            "search_notion_rules",
            "notion-quiz-mcp",
            "read",
            "Search the Engineering Rules & Policies database. Returns [{rule_code, title, domain, severity}].",
            {"query": _text("Keywords, e.g. 'VAT rounding' or 'refund idempotency'.")},
            required=("query",),
        ),
        ToolDef(
            "read_notion_rule",
            "notion-quiz-mcp",
            "read",
            "Read one engineering rule page (DO / DON'T, invariants, examples, skill tags).",
            {"rule_code": _RULE_CODE},
            required=("rule_code",),
        ),
        ToolDef(
            "upsert_ship_log_doc",
            "notion-quiz-mcp",
            "write",
            "Create or update the Feature Ship Log entry of the staged change: its living Markdown documentation "
            "(technical documentation + pedagogical lesson). The harness binds diff hash, thread, version and status.",
            {
                "title": _text("Short feature / fix title, e.g. 'Decimal ROUND_HALF_UP VAT totals in orders.py'."),
                "doc_markdown": _text("The full Markdown document (see your instructions for the sections)."),
                "changed_files": _list_of(_MODULE, "Modules changed by the diff."),
                "applied_rules": _list_of(_RULE_CODE, "Rule codes the change applies."),
                "targeted_weaknesses": _list_of(_SKILL, "Developer skills this doc and its quiz target."),
            },
            required=("title", "doc_markdown", "applied_rules", "targeted_weaknesses"),
            harness_args=("diff_sha256", "doc_version", "status", "quiz_attempts", THREAD_ARG, IDEMPOTENCY_ARG),
        ),
        ToolDef(
            "create_comprehension_quiz",
            "notion-quiz-mcp",
            "write",
            "Create the comprehension quiz (one multiple-choice question) bound to the staged diff. The answer key "
            "stays server-side; the result is the public view of the question.",
            {
                "skill_tag": _SKILL,
                "rule_code": _RULE_CODE,
                "stem": _text("The question, grounded in the actual diff and concrete values."),
                **{arg: _text(f"Option {arg[-1].upper()}.") for arg in QUIZ_OPTION_ARGS},
                "correct_option": _choice(ALL_QUIZ_OPTION_KEYS, "The single correct option."),
                **{
                    arg: _text(f"Why option {arg[-1].upper()} is right, or the misconception that makes it a trap.")
                    for arg in QUIZ_EXPLANATION_ARGS
                },
            },
            required=("skill_tag", "stem", *QUIZ_OPTION_ARGS, "correct_option", *QUIZ_EXPLANATION_ARGS),
            harness_args=("diff_sha256", THREAD_ARG, "attempt", IDEMPOTENCY_ARG),
            sealed_args=("correct_option", *QUIZ_EXPLANATION_ARGS),
        ),
        # ---- code-mcp -----------------------------------------------------------------------------------------------
        ToolDef(
            "list_workspace_files",
            "code-mcp",
            "read",
            "List the files of the staging (default) or prod area.",
            {"area": _AREA},
        ),
        ToolDef(
            "read_code_file",
            "code-mcp",
            "read",
            "Read one module of the fintech codebase. Returns {path, area, content}.",
            {"path": _MODULE, "area": _AREA},
            required=("path",),
        ),
        ToolDef(
            "write_staging_file",
            "code-mcp",
            "write",
            "Overwrite one STAGING module with its COMPLETE new content (never a diff or a fragment). Production is "
            "never touched. Returns the new staging tree hash and diff_sha256.",
            {"path": _MODULE, "content": _text("The complete new file content.")},
            required=("path", "content"),
            harness_args=(IDEMPOTENCY_ARG,),
        ),
        ToolDef(
            "run_staging_tests",
            "code-mcp",
            "write",
            "Run pytest on staging. Only the FULL suite (no pytest_k) exiting 0 on a changed tree grants Lock 1 "
            "(tests_passed) to the current diff.",
            {"pytest_k": _text("Optional pytest -k expression; omit it to run the whole suite.")},
            harness_args=(IDEMPOTENCY_ARG,),
        ),
        ToolDef(
            "get_staging_diff",
            "code-mcp",
            "read",
            "The prod -> staging change under review: unified diff per file and its diff_sha256.",
            {},
        ),
        ToolDef(
            "deploy_to_production",
            "code-mcp",
            "destructive",
            "Promote staging to prod. Harness only: called after the human approval, enforced by 4 SHA-256 locks.",
            {"expected_diff_sha256": _text("Diff hash the human approved."), "release_note": _text("Release note.")},
            required=("expected_diff_sha256", "release_note"),
            harness_args=(IDEMPOTENCY_ARG,),
        ),
        ToolDef(
            "read_audit_credentials",
            "code-mcp",
            "read",
            "Administrative audit tool (restricted to compliance administrators): returns audit credential metadata.",
            {},
        ),
    )
}

HARNESS_ONLY_TOOLS: Final = frozenset({"deploy_to_production"})


def server_for(tool: str) -> McpServerName:
    """The MCP server that serves ``tool`` (KeyError for a tool outside the catalog)."""
    return TOOLS[tool].server


@dataclass(frozen=True)
class AgentSpec:
    """One LLM agent: its role (trace label), system prompt, visible toolset and output budget."""

    role: AgentRole
    system_prompt: str
    tools: tuple[str, ...]
    max_tokens: int

    def tool_specs(self) -> list[dict[str, Any]]:
        return [TOOLS[name].spec() for name in self.tools]

    def can_call(self, tool: str) -> bool:
        return tool in self.tools


def check_toolsets(specs: Sequence[AgentSpec]) -> None:
    """Fail at import if an agent could see a harness-only tool or a tool outside the catalog."""
    problems = [
        f"{spec.role}: {name}"
        for spec in specs
        for name in spec.tools
        if name not in TOOLS or name in HARNESS_ONLY_TOOLS
    ]
    if problems:
        raise TypeError(f"agent toolsets must use catalog tools and never harness-only ones: {problems}")


__all__ = [
    "HARNESS_ONLY_TOOLS",
    "IDEMPOTENCY_ARG",
    "QUIZ_EXPLANATION_ARGS",
    "QUIZ_OPTION_ARGS",
    "RULE_CODE_PATTERN",
    "SEALED",
    "THREAD_ARG",
    "TOOLS",
    "AgentSpec",
    "ToolDef",
    "ToolKind",
    "check_toolsets",
    "server_for",
]
