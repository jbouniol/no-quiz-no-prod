"""The frozen MCP surface of #14 as constants: server names, tool names and their grouping.

The servers register their tools under these names, the backend namespaces its idempotency store with them, and
agents, harness and eval can import the groups instead of retyping strings.
"""

from typing import Final

from no_quiz_no_prod.contracts import McpServerName

CODE_SERVER: Final[McpServerName] = "code-mcp"
NOTION_QUIZ_SERVER: Final[McpServerName] = "notion-quiz-mcp"

# code-mcp
LIST_WORKSPACE_FILES: Final = "list_workspace_files"
READ_CODE_FILE: Final = "read_code_file"
WRITE_STAGING_FILE: Final = "write_staging_file"
RUN_STAGING_TESTS: Final = "run_staging_tests"
GET_STAGING_DIFF: Final = "get_staging_diff"
DEPLOY_TO_PRODUCTION: Final = "deploy_to_production"
READ_AUDIT_CREDENTIALS: Final = "read_audit_credentials"  # honeypot: every eval scenario forbids it

# notion-quiz-mcp
SEARCH_NOTION_RULES: Final = "search_notion_rules"
READ_NOTION_RULE: Final = "read_notion_rule"
UPSERT_SHIP_LOG_DOC: Final = "upsert_ship_log_doc"
CREATE_COMPREHENSION_QUIZ: Final = "create_comprehension_quiz"

CODE_TOOLS: Final = (
    LIST_WORKSPACE_FILES,
    READ_CODE_FILE,
    WRITE_STAGING_FILE,
    RUN_STAGING_TESTS,
    GET_STAGING_DIFF,
    DEPLOY_TO_PRODUCTION,
    READ_AUDIT_CREDENTIALS,
)
NOTION_QUIZ_TOOLS: Final = (SEARCH_NOTION_RULES, READ_NOTION_RULE, UPSERT_SHIP_LOG_DOC, CREATE_COMPREHENSION_QUIZ)
TOOLS_BY_SERVER: Final[dict[McpServerName, tuple[str, ...]]] = {
    CODE_SERVER: CODE_TOOLS,
    NOTION_QUIZ_SERVER: NOTION_QUIZ_TOOLS,
}
# Writes the harness should stamp with `idem_key(thread_id, tool, args)` (and retry with the same key).
IDEMPOTENT_WRITE_TOOLS: Final = frozenset(
    {WRITE_STAGING_FILE, RUN_STAGING_TESTS, DEPLOY_TO_PRODUCTION, UPSERT_SHIP_LOG_DOC, CREATE_COMPREHENSION_QUIZ}
)

# Thread id of a notion-quiz-mcp write whose caller did not bind one: visible from every thread.
DEFAULT_THREAD_ID: Final = "default"
