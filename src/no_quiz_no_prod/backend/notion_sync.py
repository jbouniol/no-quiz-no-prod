"""Best-effort mirror of the ship log and the skill memory into the live Notion databases (REST API 2022-06-28).

The local stores stay the source of truth; Notion is a mirror. ``NotionDatabaseSync`` writes one ``ShipLogEntry`` or
``SkillProfileRecord`` at a time and returns it with its Notion ids set: persist the returned model, so the next sync
updates the same page instead of creating another one. ``provision_notion_databases`` creates the three relational
databases under a parent page (the ``init-notion`` command).

Failure policy
--------------
- Disabled (``settings.notion_enabled`` is False): every sync method returns its input unchanged and sends nothing.
- The sync methods never raise. Any failure (network, timeout, non-2xx, bad JSON, unexpected shape) logs a warning,
  sets ``last_error`` and returns the input unchanged, so a Notion outage can never break the quiz and deploy flow.
  One exception prevents duplicate pages: once a page exists, a failed body append still returns its page id (with
  ``last_error`` set).
- ``NotionSyncError`` is raised only by ``provision_notion_databases``. No message ever contains the API key.

Schema-driven encoding
----------------------
Each database schema is read once (``GET /databases/{id}``) and every value is encoded for the type its column really
has. In the live hub, Ship Log "Status", Rules "Status" and Skills "Mastery" are ``status`` columns, while databases
created by ``provision_notion_databases`` use ``select`` (API 2022-06-28 cannot create a status property). Columns a
database lacks are skipped, and None becomes the empty value of the column type.
"""

import json
import logging
import re
import textwrap
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from enum import StrEnum
from http import HTTPStatus
from itertools import batched
from pathlib import Path
from typing import Any, Final, Self, get_args

import httpx
from pydantic import BaseModel

from no_quiz_no_prod.contracts import (
    ALL_DOC_VERSIONS,
    ALL_FINTECH_MODULES,
    ALL_MASTERY_LEVELS,
    ALL_RULE_DOMAINS,
    ALL_RULE_SEVERITIES,
    ALL_SHIP_LOG_STATUSES,
    ALL_SKILL_TAGS,
    NOTION_IDS_FILE,
    NotionRule,
    RuleStatus,
    RuntimeSettings,
    ShipLogEntry,
    SkillProfileRecord,
)

logger = logging.getLogger(__name__)

# ---- Notion API identity & limits ----------------------------------------------------------------------------------
NOTION_API_BASE: Final = "https://api.notion.com/v1"
NOTION_VERSION: Final = "2022-06-28"
NOTION_TIMEOUT_S: Final = 8.0  # per request: a slow Notion must not stall a tool call for long
NOTION_TEXT_CHUNK: Final = 2000  # max length of one rich_text item (Notion counts UTF-16 units: an emoji weighs 2)
NOTION_MAX_CHILDREN: Final = 100  # max blocks per create or append request
NOTION_MAX_RICH_TEXT_ITEMS: Final = 100  # max rich_text items in one block or one property value
NOTION_OPTION_NAME_MAX: Final = 100  # max length of a select option name (commas are not allowed)
NOTION_RATE_LIMIT_RETRIES: Final = 1  # retries on HTTP 429 (a rate-limited request was not processed: safe to repeat)
NOTION_MAX_RETRY_AFTER_S: Final = 2.0  # cap on the Retry-After wait

# ---- databases created by provision_notion_databases (same names as the live hub) ----------------------------------
RULES_DB_TITLE: Final = "Engineering Rules & Policies"
SHIP_LOG_DB_TITLE: Final = "Feature Ship Log & Pedagogical Docs"
SKILL_PROFILE_DB_TITLE: Final = "Developer Skill Profile (Layer 4 Memory)"
RULES_BACKLINK_PROPERTY: Final = "Linked Shipments"  # rules-side half of the two-way "Applied Rules" relation
DOC_VERSION_HEADING: Final = "Doc {doc_version}"  # heading above each doc version appended to an existing page
PLAIN_TEXT_LANGUAGE: Final = "plain text"  # code-block language when the fence names none that Notion knows

# Keys shared by docs/notion_workspace_ids.json, RuntimeSettings.from_env and the provisioning result.
KEY_PARENT_PAGE_ID: Final = "parent_page_id"
KEY_RULES_DB_ID: Final = "rules_db_id"
KEY_SHIP_LOG_DB_ID: Final = "ship_log_db_id"
KEY_SKILL_PROFILE_DB_ID: Final = "skill_profile_db_id"
_KEY_RULE_PAGE_IDS: Final = "rule_page_ids"  # {"RULE-001": "<page id>", ...}
_KEY_SKILL_PAGES: Final = "skill_profile_pages"  # {"decimal_rounding": {"page_id": "<page id>", "url": ...}, ...}
_KEY_PAGE_ID: Final = "page_id"

_RULE_STATUSES: Final = get_args(RuleStatus)
_DEFAULT_COLOR: Final = "default"
_DEFAULT_RETRY_AFTER_S: Final = 1.0
_ERROR_DETAIL_MAX: Final = 300  # characters of a Notion error message kept in last_error and the logs
_MAX_HEADING_LEVEL: Final = 3  # Notion has heading_1..3: deeper Markdown headings become heading_3
_BMP_MAX: Final = 0xFFFF  # code points above it take two UTF-16 units
_PAGE_ID_HEX_CHARS: Final = 32  # a Notion id without its dashes
_SKIP: Final = object()  # _encode_value result meaning "leave this column out of the request"

# Code-block languages accepted by the API (a conservative subset of its enum), and common fence aliases.
_CODE_LANGUAGES: Final = frozenset(
    {
        "bash",
        "c",
        "c#",
        "c++",
        "css",
        "diff",
        "docker",
        "go",
        "graphql",
        "html",
        "java",
        "javascript",
        "json",
        "kotlin",
        "makefile",
        "markdown",
        "mermaid",
        "php",
        PLAIN_TEXT_LANGUAGE,
        "powershell",
        "python",
        "ruby",
        "rust",
        "scss",
        "shell",
        "sql",
        "swift",
        "typescript",
        "xml",
        "yaml",
    }
)
_CODE_LANGUAGE_ALIASES: Final[Mapping[str, str]] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "sh": "shell",
    "zsh": "shell",
    "console": "shell",
    "yml": "yaml",
    "md": "markdown",
    "patch": "diff",
    "dockerfile": "docker",
    "cpp": "c++",
    "cs": "c#",
    "csharp": "c#",
    "golang": "go",
    "rs": "rust",
    "rb": "ruby",
    "kt": "kotlin",
    "ps1": "powershell",
    "htm": "html",
    "jsonc": "json",
    "text": PLAIN_TEXT_LANGUAGE,
    "txt": PLAIN_TEXT_LANGUAGE,
    "plaintext": PLAIN_TEXT_LANGUAGE,
}

_PAGE_ID_RE: Final = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.IGNORECASE)
_FENCE_RE: Final = re.compile(r"^(`{3,}|~{3,})(.*)$")
_HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE: Final = re.compile(r"^[-*+]\s+(.*)$")
_NUMBERED_RE: Final = re.compile(r"^\d+[.)]\s+(.*)$")
_QUOTE_RE: Final = re.compile(r"^>\s?(.*)$")
_DIVIDER_RE: Final = re.compile(r"^([-*_])(?:\s*\1){2,}$")
_HTML_LINE_RE: Final = re.compile(r"^</?[A-Za-z][^>]*>")


class _Prop(StrEnum):
    """Notion property types this module reads or writes."""

    TITLE = "title"
    RICH_TEXT = "rich_text"
    SELECT = "select"
    STATUS = "status"
    MULTI_SELECT = "multi_select"
    NUMBER = "number"
    DATE = "date"
    RELATION = "relation"


class NotionSyncError(RuntimeError):
    """A Notion request failed. Raised only by ``provision_notion_databases``: the sync methods swallow it.

    The message is safe to show and log: it never contains the API key. ``status_code`` is the HTTP status of the
    failed response, or None when no response came back.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---- pure helpers: text, blocks, ids -------------------------------------------------------------------------------
def _scrub(text: str, secret: str) -> str:
    """``text`` with every occurrence of ``secret`` masked (defense in depth: no message may carry the key)."""
    return text.replace(secret, "***") if secret else text


def _describe(exc: BaseException) -> str:
    """One-line description of a failure: our own messages as they are, anything else prefixed by its type."""
    return str(exc) if isinstance(exc, NotionSyncError) else f"{type(exc).__name__}: {exc}"


def _utf16_units(text: str) -> int:
    return len(text) + sum(ord(char) > _BMP_MAX for char in text)


def _chunk_text(text: str) -> list[str]:
    """``text`` cut into rich_text-sized pieces of at most ``NOTION_TEXT_CHUNK`` UTF-16 units, the unit Notion counts
    (so at most that many characters too). Empty text gives no piece."""
    if _utf16_units(text) <= NOTION_TEXT_CHUNK:
        return [text] if text else []
    chunks: list[str] = []
    start = units = 0
    for index, char in enumerate(text):
        width = 1 + (ord(char) > _BMP_MAX)
        if units + width > NOTION_TEXT_CHUNK:
            chunks.append(text[start:index])
            start, units = index, 0
        units += width
    chunks.append(text[start:])
    return chunks


def _text_item(content: str) -> dict[str, Any]:
    return {"type": "text", "text": {"content": content}}


def _rich_text(text: str) -> list[dict[str, Any]]:
    """Plain rich_text for a property value (beyond 100 chunks, i.e. 200k characters, the text is truncated)."""
    return [_text_item(chunk) for chunk in _chunk_text(text)[:NOTION_MAX_RICH_TEXT_ITEMS]]


def _text_blocks(block_type: str, text: str, **extra: Any) -> list[dict[str, Any]]:
    """Blocks of ``block_type`` holding ``text``: one block, or several when it needs over 100 rich_text items."""
    groups = list(batched(_chunk_text(text), NOTION_MAX_RICH_TEXT_ITEMS)) or [()]
    return [
        {"object": "block", "type": block_type, block_type: {"rich_text": [_text_item(c) for c in group], **extra}}
        for group in groups
    ]


def _divider_block() -> dict[str, Any]:
    return {"object": "block", "type": "divider", "divider": {}}


def _code_language(info: str) -> str:
    """Notion language of a code fence info string (``python``, ``py title=x.py``...), else "plain text"."""
    token = (info.split() or [""])[0].lower()
    language = _CODE_LANGUAGE_ALIASES.get(token, token)
    return language if language in _CODE_LANGUAGES else PLAIN_TEXT_LANGUAGE


def _code_blocks(lines: Sequence[str], language: str) -> list[dict[str, Any]]:
    return _text_blocks("code", textwrap.dedent("\n".join(lines)), language=language)


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    """Notion blocks for a Markdown document. Text stays plain: inline marks such as ``**bold**`` are kept as typed.

    Handles ``#``/``##``/``###`` headings (deeper levels become heading 3), ``-``/``*``/``+`` bullets, ``1.``
    numbered items, fenced code (language mapped to Notion's names, else "plain text"), ``>`` quotes, ``---``
    dividers and paragraphs (consecutive lines join with a newline). HTML-ish lines such as ``<callout>`` or
    ``</columns>`` become paragraphs of their own. Every rich_text chunk holds at most ``NOTION_TEXT_CHUNK`` chars.
    """
    blocks: list[dict[str, Any]] = []
    run_type: str | None = None  # "paragraph" or "quote": consecutive lines merge into one block
    run: list[str] = []
    fence: str | None = None  # the opening fence while inside a code block
    language = PLAIN_TEXT_LANGUAGE
    code: list[str] = []

    def flush() -> None:
        nonlocal run_type
        if run_type is not None:
            blocks.extend(_text_blocks(run_type, "\n".join(run)))
        run_type = None
        run.clear()

    def extend_run(block_type: str, text: str) -> None:
        nonlocal run_type
        if run_type != block_type:
            flush()
            run_type = block_type
        run.append(text)

    def emit(block_type: str, text: str) -> None:
        flush()
        blocks.extend(_text_blocks(block_type, text))

    for line in markdown.splitlines():
        stripped = line.strip()
        if fence is not None:
            if stripped and stripped == fence[0] * len(stripped) and len(stripped) >= len(fence):
                blocks.extend(_code_blocks(code, language))
                fence, code = None, []
            else:
                code.append(line)
        elif opening := _FENCE_RE.match(stripped):
            flush()
            fence, language = opening.group(1), _code_language(opening.group(2))
        elif not stripped:
            flush()
        elif _DIVIDER_RE.match(stripped):
            flush()
            blocks.append(_divider_block())
        elif heading := _HEADING_RE.match(stripped):
            emit(f"heading_{min(len(heading.group(1)), _MAX_HEADING_LEVEL)}", heading.group(2).strip())
        elif _HTML_LINE_RE.match(stripped):
            emit("paragraph", stripped)
        elif bullet := _BULLET_RE.match(stripped):
            emit("bulleted_list_item", bullet.group(1))
        elif numbered := _NUMBERED_RE.match(stripped):
            emit("numbered_list_item", numbered.group(1))
        elif quote := _QUOTE_RE.match(stripped):
            extend_run("quote", quote.group(1))
        else:
            extend_run("paragraph", stripped)
    if fence is not None:  # unclosed fence: keep the code anyway
        blocks.extend(_code_blocks(code, language))
    flush()
    return blocks


def _normalize_page_id(raw: Any) -> str | None:
    """Dashed lowercase form of a Notion id (32 hex chars, dashes optional), or None when ``raw`` is not one."""
    if not isinstance(raw, str) or not _PAGE_ID_RE.match(raw.strip()):
        return None
    return str(uuid.UUID(raw.strip().replace("-", "")))


def _parse_page_id(text: str) -> str | None:
    """A Notion page id from an id (dashes optional) or a page URL such as ``https://www.notion.so/Hub-<32 hex>``."""
    candidate = text.strip().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    segment = candidate.rsplit("/", 1)[-1]  # last path segment: "<id>" or "<Title>-<32 hex>"
    for option in (candidate, segment, segment[-_PAGE_ID_HEX_CHARS:]):
        if (page_id := _normalize_page_id(option)) is not None:
            return page_id
    return None


def _notion_page_ids(items: Iterable[Any]) -> list[str]:
    """The items that look like Notion page ids, normalized and de-duplicated (anything else is dropped)."""
    return list(dict.fromkeys(page_id for item in items if (page_id := _normalize_page_id(item))))


# ---- pure helpers: property encoding -------------------------------------------------------------------------------
def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, date):  # datetimes included: ISO-8601
        return value.isoformat()
    if isinstance(value, list | tuple):
        return ", ".join(_as_text(item) for item in value)
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _option_name(value: Any) -> str | None:
    """A valid select option name for ``value`` (no comma, at most 100 chars), or None for an empty value."""
    return _as_text(value).replace(",", " ").strip()[:NOTION_OPTION_NAME_MAX].strip() or None


def _encode_value(kind: str, value: Any) -> Any:
    """Page-property value of ``value`` for a column of type ``kind``, or ``_SKIP`` when it cannot be written.

    None becomes the type's empty value, except for ``status``, which always holds an option: it is left untouched.
    Relation values must already be page ids; anything else in the list is dropped.
    """
    match kind:
        case _Prop.TITLE | _Prop.RICH_TEXT:
            return _rich_text(_as_text(value))
        case _Prop.SELECT | _Prop.STATUS:
            items = _as_list(value)
            name = _option_name(items[0]) if items else None
            if name is not None:
                return {"name": name}
            return None if kind == _Prop.SELECT else _SKIP
        case _Prop.MULTI_SELECT:
            names = dict.fromkeys(name for item in _as_list(value) if (name := _option_name(item)))
            return [{"name": name} for name in names]
        case _Prop.NUMBER:
            if value is None:
                return None
            return value if isinstance(value, int | float) and not isinstance(value, bool) else _SKIP
        case _Prop.DATE:
            return None if value is None else {"start": _as_text(value)}
        case _Prop.RELATION:
            return [{"id": page_id} for page_id in _notion_page_ids(_as_list(value))]
        case _:
            return _SKIP  # formula, rollup, people...: never written


def _encode_properties(schema: Mapping[str, str], values: Mapping[str, Any]) -> dict[str, Any]:
    """``{column: {type: value}}`` for every value whose column exists in ``schema`` (``{column: type}``)."""
    properties: dict[str, Any] = {}
    for column, value in values.items():
        kind = schema.get(column)
        if kind is None:
            continue  # the database lacks this column
        encoded = _encode_value(kind, value)
        if encoded is not _SKIP:
            properties[column] = {kind: encoded}
    return properties


def _column_values(model: BaseModel, columns: Mapping[str, str]) -> dict[str, Any]:
    """``{Notion column: value}`` for a contract model's ``NOTION_PROPERTIES`` (``{field: column}``)."""
    return {column: getattr(model, field) for field, column in columns.items()}


def _title_column(schema: Mapping[str, str], preferred: str) -> str:
    """``preferred`` when the database has it, else its title column (so a renamed title is still written)."""
    if preferred in schema:
        return preferred
    return next((column for column, kind in schema.items() if kind == _Prop.TITLE), preferred)


def _with_title_column(values: dict[str, Any], schema: Mapping[str, str], preferred: str) -> dict[str, Any]:
    column = _title_column(schema, preferred)
    if column != preferred:
        values[column] = values.pop(preferred)
    return values


def _skill_values(record: SkillProfileRecord, schema: Mapping[str, str]) -> dict[str, Any]:
    """Column values of a skill row: the title column holds the skill tag."""
    columns = SkillProfileRecord.NOTION_PROPERTIES
    return _with_title_column(_column_values(record, columns), schema, columns["skill_tag"])


def _require_str(data: Any, key: str, what: str) -> str:
    value = data.get(key) if isinstance(data, Mapping) else None
    if not isinstance(value, str) or not value:
        raise NotionSyncError(f"unexpected Notion response: the {what} has no {key!r}")
    return value


# ---- pure helpers: the optional workspace ids file -----------------------------------------------------------------
def _read_workspace_ids(path: Path) -> dict[str, Any]:
    """The workspace ids file as a dict, or {} when it is missing or unreadable. Never raises."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _id_map(raw: Any) -> dict[str, str]:
    """``{key: normalized page id}`` for the entries of ``raw`` whose value is a page id."""
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): page_id for key, value in raw.items() if (page_id := _normalize_page_id(value))}


def _same_database(recorded: Any, configured: str | None) -> bool:
    """False when the ids file describes another database than the settings: its page ids would then be foreign
    (e.g. after ``init-notion`` provisioned fresh databases). A file that records no database id is trusted."""
    if recorded is None:
        return True
    expected = _normalize_page_id(recorded)
    return expected is not None and expected == _normalize_page_id(configured)


# ---- HTTP ------------------------------------------------------------------------------------------------------------
def _error_detail(response: httpx.Response) -> str:
    """Notion's ``code: message`` for a failed response (bounded), else the HTTP reason phrase."""
    try:
        data = response.json()
    except ValueError:
        data = None
    parts = [str(data[key]) for key in ("code", "message") if data.get(key)] if isinstance(data, dict) else []
    return (": ".join(parts) or response.reason_phrase)[:_ERROR_DETAIL_MAX]


def _retry_after_s(response: httpx.Response) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", _DEFAULT_RETRY_AFTER_S))
    except ValueError:
        seconds = _DEFAULT_RETRY_AFTER_S
    return min(max(seconds, 0.0), NOTION_MAX_RETRY_AFTER_S)


class _NotionApi:
    """JSON over HTTPS with one ``httpx.Client``: every failure becomes a ``NotionSyncError`` without the key."""

    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None, timeout_s: float) -> None:
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=NOTION_API_BASE,
            headers={"Authorization": f"Bearer {api_key}", "Notion-Version": NOTION_VERSION},
            timeout=timeout_s,
            transport=transport,
        )

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and return its JSON object; raise ``NotionSyncError`` on any failure."""
        where = f"{method} {path}"
        try:
            response = self._send(method, path, body)
        except httpx.HTTPError as exc:  # timeouts, connection errors, protocol errors
            raise NotionSyncError(_scrub(f"{where}: {type(exc).__name__}: {exc}", self._api_key)) from None
        if not response.is_success:
            detail = _scrub(_error_detail(response), self._api_key)
            raise NotionSyncError(f"{where}: HTTP {response.status_code} {detail}", status_code=response.status_code)
        try:
            data = response.json()
        except ValueError:
            raise NotionSyncError(f"{where}: the response is not JSON", status_code=response.status_code) from None
        if not isinstance(data, dict):
            raise NotionSyncError(f"{where}: unexpected response shape", status_code=response.status_code)
        return data

    def _send(self, method: str, path: str, body: Mapping[str, Any] | None) -> httpx.Response:
        response = self._client.request(method, path, json=body)
        for _ in range(NOTION_RATE_LIMIT_RETRIES):
            if response.status_code != HTTPStatus.TOO_MANY_REQUESTS:
                break
            time.sleep(_retry_after_s(response))
            response = self._client.request(method, path, json=body)
        return response

    def close(self) -> None:
        self._client.close()


# ---- write-through sync --------------------------------------------------------------------------------------------
class NotionDatabaseSync:
    """Best-effort write-through of ship-log entries and skill records to Notion (see the module docstring).

    One instance per process is enough: it opens one HTTP client on first use and caches the database schemas and the
    page ids it resolves. Use it as a context manager, or call ``close()``.
    """

    last_error: str | None  # last swallowed failure (never a secret); None after a fully successful sync

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = NOTION_TIMEOUT_S,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._timeout_s = timeout_s
        self._http: _NotionApi | None = None
        self._lock = threading.Lock()
        self._schemas: dict[str, dict[str, str]] = {}  # database id -> {column: property type}
        self._rule_pages: dict[str, str] = {}  # rule code -> page id
        self._skill_pages: dict[str, str] = {}  # skill tag -> page id
        self._workspace_ids: dict[str, Any] | None = None  # the workspace ids file, read once
        self.last_error = None

    @property
    def enabled(self) -> bool:
        """True when the Notion key and the three database ids are configured (``settings.notion_enabled``)."""
        return self._settings.notion_enabled

    # ---- public API ------------------------------------------------------------------------------------------------
    def sync_ship_log(self, entry: ShipLogEntry, *, include_body: bool = False) -> ShipLogEntry:
        """Create or update the entry's row in the Feature Ship Log; return it with ``notion_page_id``/``notion_url``.

        Without ``notion_page_id`` a page is created (its body is the doc when ``include_body``). Otherwise the page's
        properties are updated and, when ``include_body``, the doc is appended under a divider and a "Doc vN"
        heading: earlier versions stay, nothing is ever deleted. A page id that answers 404 gets a new page.
        "Applied Rules" links the rule pages of ``applied_rules`` (codes that resolve to no page are skipped).
        Never raises: on failure, the entry comes back unchanged and ``last_error`` says why.
        """
        database_id = self._settings.notion_ship_log_db_id
        if not self.enabled or not database_id:
            return entry
        return self._guarded("ship log sync", entry, lambda: self._sync_ship_log(database_id, entry, include_body))

    def sync_skill_record(self, record: SkillProfileRecord) -> SkillProfileRecord:
        """Upsert the record's row in the Developer Skill Profile; return it with ``notion_page_id`` set.

        The row is found by ``record.notion_page_id``, then by this session's cache, then by the workspace ids file,
        then by a query on the title ("Skill Tag" equals the tag); it is created when none matches. "Evidence
        Shipments" links the items of ``evidence_shipments`` that are Notion page ids. Never raises: on failure, the
        record comes back unchanged and ``last_error`` says why.
        """
        database_id = self._settings.notion_skills_db_id
        if not self.enabled or not database_id:
            return record
        return self._guarded("skill profile sync", record, lambda: self._sync_skill_record(database_id, record))

    def close(self) -> None:
        """Close the HTTP client (a later sync opens a new one)."""
        with self._lock:
            http, self._http = self._http, None
        if http is not None:
            http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- failure handling ------------------------------------------------------------------------------------------
    def _guarded[T](self, action: str, fallback: T, operation: Callable[[], T]) -> T:
        self.last_error = None
        try:
            return operation()
        except Exception as exc:  # a sync never raises: Notion is a mirror, not a dependency of the gate
            self._record_failure(action, exc)
            return fallback

    def _record_failure(self, action: str, exc: Exception) -> None:
        key = self._settings.notion_api_key
        self.last_error = _scrub(f"Notion {action} failed: {_describe(exc)}", key.get_secret_value() if key else "")
        if isinstance(exc, NotionSyncError) and exc.status_code == HTTPStatus.BAD_REQUEST:
            self._schemas.clear()  # a column may have been renamed or retyped: read the schemas again next time
        logger.warning("%s; continuing without Notion", self.last_error)

    # ---- requests --------------------------------------------------------------------------------------------------
    def _api(self) -> _NotionApi:
        with self._lock:
            if self._http is None:
                key = self._settings.notion_api_key
                if key is None:
                    raise NotionSyncError("NOTION_API_KEY is not set")
                self._http = _NotionApi(key.get_secret_value(), transport=self._transport, timeout_s=self._timeout_s)
            return self._http

    def _schema(self, database_id: str) -> dict[str, str]:
        """``{column: property type}`` of a database, fetched once per process."""
        schema = self._schemas.get(database_id)
        if schema is None:
            properties = self._api().request("GET", f"/databases/{database_id}").get("properties")
            if not isinstance(properties, dict):
                raise NotionSyncError("unexpected Notion response: the database has no 'properties'")
            schema = {
                column: spec["type"]
                for column, spec in properties.items()
                if isinstance(spec, dict) and isinstance(spec.get("type"), str)
            }
            self._schemas[database_id] = schema
        return schema

    def _query_first(self, database_id: str, condition: Mapping[str, Any]) -> str | None:
        """Id of the first page of ``database_id`` matching the filter ``condition``, or None."""
        body = {"filter": condition, "page_size": 1}
        results = self._api().request("POST", f"/databases/{database_id}/query", body).get("results")
        if not isinstance(results, list):
            raise NotionSyncError("unexpected Notion response: the query has no 'results'")
        return _require_str(results[0], "id", "query result") if results else None

    def _create_page(
        self, database_id: str, properties: Mapping[str, Any], blocks: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"parent": {"database_id": database_id}, "properties": properties}
        if blocks:
            body["children"] = list(blocks[:NOTION_MAX_CHILDREN])
        page = self._api().request("POST", "/pages", body)
        self._append_blocks(_require_str(page, "id", "created page"), blocks[NOTION_MAX_CHILDREN:])
        return page

    def _update_page(self, page_id: str, properties: Mapping[str, Any]) -> dict[str, Any] | None:
        """PATCH a page's properties; None when the page is gone (404), so the caller can fall back."""
        try:
            return self._api().request("PATCH", f"/pages/{page_id}", {"properties": properties})
        except NotionSyncError as exc:
            if exc.status_code != HTTPStatus.NOT_FOUND:
                raise
            logger.warning("Notion page %s not found (404); falling back", page_id)
            return None

    def _append_blocks(self, block_id: str, blocks: Sequence[dict[str, Any]]) -> None:
        """Append ``blocks`` to a page in batches of 100. The page itself is already saved, so a failure here only
        sets ``last_error``: the caller still returns the page id and never creates a duplicate page."""
        try:
            for batch in batched(blocks, NOTION_MAX_CHILDREN):
                self._api().request("PATCH", f"/blocks/{block_id}/children", {"children": list(batch)})
        except Exception as exc:
            self._record_failure("page body append", exc)

    # ---- page-id resolution ----------------------------------------------------------------------------------------
    def _workspace_file(self) -> dict[str, Any]:
        if self._workspace_ids is None:
            self._workspace_ids = _read_workspace_ids(NOTION_IDS_FILE)
        return self._workspace_ids

    def _file_rule_pages(self) -> dict[str, str]:
        data = self._workspace_file()
        if not _same_database(data.get(KEY_RULES_DB_ID), self._settings.notion_rules_db_id):
            return {}
        return _id_map(data.get(_KEY_RULE_PAGE_IDS))

    def _file_skill_pages(self) -> dict[str, str]:
        data = self._workspace_file()
        pages = data.get(_KEY_SKILL_PAGES)
        if not isinstance(pages, Mapping) or not _same_database(
            data.get(KEY_SKILL_PROFILE_DB_ID), self._settings.notion_skills_db_id
        ):
            return {}
        return _id_map({tag: entry.get(_KEY_PAGE_ID) for tag, entry in pages.items() if isinstance(entry, Mapping)})

    def _rule_page_ids(self, codes: Iterable[str]) -> list[str]:
        """Page ids of the rule pages of ``codes``: session cache, workspace ids file, then a "Rule Code" query."""
        page_ids: list[str] = []
        file_pages = self._file_rule_pages()
        for code in dict.fromkeys(codes):
            page_id = self._rule_pages.get(code) or file_pages.get(code) or self._query_rule_page(code)
            if page_id is None:
                logger.info("Notion: no rule page for %s; it is left out of the relation", code)
                continue
            self._rule_pages[code] = page_id
            page_ids.append(page_id)
        return page_ids

    def _query_rule_page(self, code: str) -> str | None:
        rules_db_id = self._settings.notion_rules_db_id
        if not rules_db_id:
            return None
        return self._query_first(
            rules_db_id, {"property": NotionRule.NOTION_PROPERTIES["rule_code"], _Prop.RICH_TEXT: {"equals": code}}
        )

    def _skill_candidates(self, record: SkillProfileRecord) -> list[str]:
        """Known page ids of the record's row, best first: its own, this session's, then the workspace ids file."""
        tag = record.skill_tag
        return _notion_page_ids((record.notion_page_id, self._skill_pages.get(tag), self._file_skill_pages().get(tag)))

    # ---- the two syncs ---------------------------------------------------------------------------------------------
    def _sync_ship_log(self, database_id: str, entry: ShipLogEntry, include_body: bool) -> ShipLogEntry:
        schema = self._schema(database_id)
        columns = ShipLogEntry.NOTION_PROPERTIES
        values = _with_title_column(_column_values(entry, columns), schema, columns["title"])
        if schema.get(columns["applied_rules"]) == _Prop.RELATION:
            values[columns["applied_rules"]] = self._rule_page_ids(entry.applied_rules)
        properties = _encode_properties(schema, values)
        blocks = markdown_to_blocks(entry.doc_markdown) if include_body else []
        known_page_id = _normalize_page_id(entry.notion_page_id)  # a malformed id is treated as no page yet
        page = self._update_page(known_page_id, properties) if known_page_id else None
        if page is None:
            page = self._create_page(database_id, properties, blocks)
        elif blocks:
            heading = _text_blocks("heading_2", DOC_VERSION_HEADING.format(doc_version=entry.doc_version))
            self._append_blocks(_require_str(page, "id", "page"), [_divider_block(), *heading, *blocks])
        url = page.get("url")
        updates = {
            "notion_page_id": _require_str(page, "id", "page"),
            "notion_url": url if isinstance(url, str) and url else entry.notion_url,
        }
        return type(entry).model_validate({**entry.model_dump(), **updates})

    def _sync_skill_record(self, database_id: str, record: SkillProfileRecord) -> SkillProfileRecord:
        schema = self._schema(database_id)
        properties = _encode_properties(schema, _skill_values(record, schema))
        page = None
        for page_id in self._skill_candidates(record):
            if (page := self._update_page(page_id, properties)) is not None:
                break
        if page is None:
            title = _title_column(schema, SkillProfileRecord.NOTION_PROPERTIES["skill_tag"])
            found = self._query_first(database_id, {"property": title, _Prop.TITLE: {"equals": record.skill_tag}})
            if found is not None:
                page = self._update_page(found, properties)
        if page is None:
            page = self._create_page(database_id, properties, [])
        page_id = _require_str(page, "id", "page")
        self._skill_pages[record.skill_tag] = page_id
        return type(record).model_validate({**record.model_dump(), "notion_page_id": page_id})


# ---- provisioning (init-notion) ------------------------------------------------------------------------------------
def _options(kind: _Prop, names: Sequence[str], colors: Sequence[str]) -> dict[str, Any]:
    """Schema of a select or multi_select column; colors follow the live hub (missing ones are "default")."""
    options = [
        {"name": name, "color": colors[index] if index < len(colors) else _DEFAULT_COLOR}
        for index, name in enumerate(names)
    ]
    return {kind: {"options": options}}


def _number() -> dict[str, Any]:
    return {_Prop.NUMBER: {"format": "number"}}


def _relation(database_id: str, *, dual: bool) -> dict[str, Any]:
    mode = "dual_property" if dual else "single_property"
    return {_Prop.RELATION: {"database_id": database_id, "type": mode, mode: {}}}


def _rules_schema() -> dict[str, Any]:
    columns = NotionRule.NOTION_PROPERTIES
    return {
        columns["title"]: {_Prop.TITLE: {}},
        columns["rule_code"]: {_Prop.RICH_TEXT: {}},
        columns["domain"]: _options(_Prop.SELECT, ALL_RULE_DOMAINS, ("blue", "red", "purple", "green")),
        columns["severity"]: _options(_Prop.SELECT, ALL_RULE_SEVERITIES, ("red", "orange", "gray")),
        columns["status"]: _options(_Prop.SELECT, _RULE_STATUSES, ("gray", "blue", "green")),
    }


def _ship_log_schema(rules_db_id: str) -> dict[str, Any]:
    columns = ShipLogEntry.NOTION_PROPERTIES
    return {
        columns["title"]: {_Prop.TITLE: {}},
        columns["status"]: _options(_Prop.SELECT, ALL_SHIP_LOG_STATUSES, ("gray", "yellow", "orange", "red", "green")),
        columns["doc_version"]: _options(_Prop.SELECT, ALL_DOC_VERSIONS, ("blue", "orange", "red")),
        columns["changed_files"]: _options(
            _Prop.MULTI_SELECT, ALL_FINTECH_MODULES, ("blue", "purple", "red", "green", "gray")
        ),
        columns["targeted_weaknesses"]: _options(
            _Prop.MULTI_SELECT, ALL_SKILL_TAGS, ("blue", "red", "orange", "purple", "yellow", "green")
        ),
        columns["quiz_attempts"]: _number(),
        columns["diff_sha256"]: {_Prop.RICH_TEXT: {}},
        columns["applied_rules"]: _relation(rules_db_id, dual=True),
        columns["deployed_at"]: {_Prop.DATE: {}},
    }


def _skill_profile_schema(ship_log_db_id: str) -> dict[str, Any]:
    columns = SkillProfileRecord.NOTION_PROPERTIES
    return {
        columns["skill_tag"]: {_Prop.TITLE: {}},
        columns["mastery"]: _options(_Prop.SELECT, ALL_MASTERY_LEVELS, ("gray", "red", "yellow", "green")),
        columns["fail_count"]: _number(),
        columns["pass_count"]: _number(),
        columns["last_misconception"]: {_Prop.RICH_TEXT: {}},
        columns["evidence_shipments"]: _relation(ship_log_db_id, dual=False),
    }


def _create_database(
    api: _NotionApi, parent_page_id: str, title: str, properties: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": _rich_text(title),
        "properties": properties,
    }
    database = api.request("POST", "/databases", body)
    return _require_str(database, "id", "created database"), database


def _rename_backlink(api: _NotionApi, rules_db_id: str, ship_log_database: Mapping[str, Any]) -> None:
    """Best effort: give the rules-side half of "Applied Rules" the live hub's name instead of Notion's default one.
    Cosmetic only, so a failure is logged and provisioning goes on."""
    column = ShipLogEntry.NOTION_PROPERTIES["applied_rules"]
    try:
        synced = ship_log_database["properties"][column][_Prop.RELATION]["dual_property"]
        key = synced.get("synced_property_name") or synced["synced_property_id"]  # a name is always accepted
        api.request("PATCH", f"/databases/{rules_db_id}", {"properties": {key: {"name": RULES_BACKLINK_PROPERTY}}})
    except (LookupError, TypeError, AttributeError, NotionSyncError) as exc:
        logger.warning("Notion: the rules backlink keeps its default name (%s)", _describe(exc))


def _provisioning_error(exc: Exception, created: Mapping[str, str], secret: str) -> NotionSyncError:
    status = exc.status_code if isinstance(exc, NotionSyncError) else None
    hint = " (is the parent page shared with the integration?)" if status == HTTPStatus.NOT_FOUND else ""
    done = ", ".join(f"{key}={value}" for key, value in created.items() if key != KEY_PARENT_PAGE_ID)
    already = f"; already created: {done}" if done else ""
    message = f"Notion provisioning failed: {_describe(exc)}{hint}{already}"
    return NotionSyncError(_scrub(message, secret), status_code=status)


def provision_notion_databases(
    parent_page_id: str, api_key: str, *, transport: httpx.BaseTransport | None = None
) -> dict[str, str]:
    """Create the three relational databases under a Notion page, then the six skill rows (Not Assessed, 0/0).

    ``parent_page_id`` may be a page id (dashes optional) or a page URL; the page must be shared with the
    integration. Status-like columns are ``select`` here because API 2022-06-28 cannot create ``status`` properties
    (the sync follows whatever type it reads). Returns exactly ``{"parent_page_id", "rules_db_id", "ship_log_db_id",
    "skill_profile_db_id"}``. Raises ``NotionSyncError`` (never containing the key) on any failure; its message lists
    the databases already created.
    """
    parent = _parse_page_id(parent_page_id)
    secret = api_key.strip()
    if parent is None or not secret:
        raise NotionSyncError("Notion provisioning needs a page id (or page URL) and an API key")
    created: dict[str, str] = {KEY_PARENT_PAGE_ID: parent}
    api = _NotionApi(secret, transport=transport, timeout_s=NOTION_TIMEOUT_S)
    try:
        created[KEY_RULES_DB_ID], _ = _create_database(api, parent, RULES_DB_TITLE, _rules_schema())
        ship_log_schema = _ship_log_schema(created[KEY_RULES_DB_ID])
        created[KEY_SHIP_LOG_DB_ID], ship_log = _create_database(api, parent, SHIP_LOG_DB_TITLE, ship_log_schema)
        _rename_backlink(api, created[KEY_RULES_DB_ID], ship_log)
        skills_schema = _skill_profile_schema(created[KEY_SHIP_LOG_DB_ID])
        skills_db_id, _ = _create_database(api, parent, SKILL_PROFILE_DB_TITLE, skills_schema)
        created[KEY_SKILL_PROFILE_DB_ID] = skills_db_id
        column_types = {column: next(iter(spec)) for column, spec in skills_schema.items()}  # {column: type}
        for tag in ALL_SKILL_TAGS:
            row = _skill_values(SkillProfileRecord(skill_tag=tag), column_types)
            body = {"parent": {"database_id": skills_db_id}, "properties": _encode_properties(column_types, row)}
            api.request("POST", "/pages", body)
    except Exception as exc:
        raise _provisioning_error(exc, created, secret) from None
    finally:
        api.close()
    return created


__all__ = [
    "DOC_VERSION_HEADING",
    "KEY_PARENT_PAGE_ID",
    "KEY_RULES_DB_ID",
    "KEY_SHIP_LOG_DB_ID",
    "KEY_SKILL_PROFILE_DB_ID",
    "NOTION_API_BASE",
    "NOTION_MAX_CHILDREN",
    "NOTION_MAX_RETRY_AFTER_S",
    "NOTION_MAX_RICH_TEXT_ITEMS",
    "NOTION_OPTION_NAME_MAX",
    "NOTION_RATE_LIMIT_RETRIES",
    "NOTION_TEXT_CHUNK",
    "NOTION_TIMEOUT_S",
    "NOTION_VERSION",
    "PLAIN_TEXT_LANGUAGE",
    "RULES_BACKLINK_PROPERTY",
    "RULES_DB_TITLE",
    "SHIP_LOG_DB_TITLE",
    "SKILL_PROFILE_DB_TITLE",
    "NotionDatabaseSync",
    "NotionSyncError",
    "markdown_to_blocks",
    "provision_notion_databases",
]
