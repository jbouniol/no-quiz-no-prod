"""FastAPI Conversational Studio & Evaluation Explorer backend (`no-quiz-ui`).

Implements every REST and SSE endpoint defined in ``docs/ui_design_spec.md`` §5,
serializing canonical models from ``no_quiz_no_prod.contracts`` with zero duplicate
schema definitions and strict answer-key secrecy on ``PublicQuizView``.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from no_quiz_no_prod import __version__
from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.backend.notion_sync import NotionDatabaseSync
from no_quiz_no_prod.backend.release import GitReleasePublisher, describe_release_remote, validate_release_config
from no_quiz_no_prod.backend.workspace import SandboxWorkspace
from no_quiz_no_prod.contracts import (
    ALL_EVAL_OUTCOMES,
    ALL_SCENARIO_KINDS,
    ALL_SKILL_TAGS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_USER_ID,
    ENV_FILE,
    ERROR_STOP_REASONS,
    NOTION_IDS_FILE,
    PROJECT_ROOT,
    RELEASE_RETRY_CODES,
    DeployAction,
    DeployDecision,
    DeployGateStatus,
    DiffReport,
    EnvVar,
    EvalRunResult,
    FaultKind,
    HumanInterrupt,
    InjectionSpec,
    NotionRule,
    PendingRelease,
    PublicQuizView,
    PytestReport,
    QuizAnswer,
    QuizGradeResult,
    QuizOptionKey,
    RunBudget,
    RunEvent,
    RunFinalState,
    RuntimeSettings,
    Sha256Hex,
    ShipLogEntry,
    SkillProfileRecord,
    StopReason,
    ToolCallTrace,
    ToolErrorCode,
    Topology,
    pass_at_k,
    pass_hat_k,
)
from no_quiz_no_prod.eval.manifest import load_manifest
from no_quiz_no_prod.eval.paths import LATEST_CAMPAIGN_DIR, RUNS_DIR
from no_quiz_no_prod.eval.scorer import is_hidden_failure
from no_quiz_no_prod.ui.security import (
    ALLOWED_HOSTS_ENV,
    SESSION_TOKEN_PLACEHOLDER,
    StudioGuard,
    allowed_hosts_from,
    allowed_hosts_from_env,
    new_session_token,
    normalize_allowed_hosts,
    same_origin_client_kwargs,
)

STATIC_DIR: Final = Path(__file__).resolve().parent / "static"
SERVICE_NAME: Final = "no-quiz-no-prod-ui"
SECONDARY_OPENROUTER_MODELS: Final[tuple[tuple[str, str], ...]] = (
    ("google/gemini-2.5-flash", "OpenRouter · Gemini 2.5 Flash"),  # the second benchmarked cohort (eval/README)
    ("openai/gpt-4o-mini", "OpenRouter · GPT-4o Mini"),
    ("anthropic/claude-sonnet-4.5", "OpenRouter · Claude Sonnet 4.5"),
)
ARTIFACT_TRIGGER_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "write_staging_file",
        "run_staging_tests",
        "upsert_ship_log_doc",
        "create_comprehension_quiz",
        "deploy_to_production",
    }
)


# ---- dependency-injection protocols (lazy-bound to Wave 2 backend & graph) -----------------------------------------
class WorkspaceBackend(Protocol):
    """Subset of ``SandboxWorkspace`` (#14 §4) consumed by the Web Studio."""

    def reset(
        self,
        seed: str = DEFAULT_SEED,
        *,
        fault: FaultKind | None = None,
        injection: InjectionSpec | None = None,
        preserve_skill_memory: bool = False,
    ) -> RunFinalState: ...

    def get_diff_report(self) -> DiffReport: ...
    def get_latest_pytest(self) -> PytestReport | None: ...
    def get_latest_ship_log(self, thread_id: str | None = None) -> ShipLogEntry | None: ...
    def get_latest_public_quiz(
        self, thread_id: str | None = None, diff_sha256: str | None = None
    ) -> PublicQuizView | None: ...
    def grade_quiz_answer(
        self, question_id: str, selected_option: QuizOptionKey, *, user_id: str = DEFAULT_USER_ID
    ) -> QuizGradeResult: ...
    def apply_deploy_decision(self, decision: DeployDecision, *, thread_id: str | None = None) -> DeployGateStatus: ...
    def get_gate_status(self, expected_diff_sha256: str) -> DeployGateStatus: ...
    def gate_status_of(self, diff_sha256: str) -> DeployGateStatus: ...
    def get_skill_profile(self, user_id: str = DEFAULT_USER_ID) -> list[SkillProfileRecord]: ...
    def forget_user_skills(self, user_id: str = DEFAULT_USER_ID) -> int: ...


class WorkflowOutcomeLike(Protocol):
    """Duck-typed view of ``WorkflowOutcome`` (#14 §5)."""

    @property
    def thread_id(self) -> str: ...
    @property
    def stop_reason(self) -> StopReason | None: ...
    @property
    def interrupt(self) -> HumanInterrupt | None: ...
    @property
    def final_answer(self) -> str: ...
    @property
    def trace(self) -> list[ToolCallTrace]: ...
    @property
    def budget(self) -> RunBudget: ...
    @property
    def final_state(self) -> RunFinalState: ...
    @property
    def pending_release(self) -> PendingRelease | None: ...


class WorkflowBackend(Protocol):
    """Subset of ``NoQuizWorkflow`` (#14 §5) consumed by the Web Studio."""

    def start_run(
        self,
        task: str,
        *,
        thread_id: str,
        user_id: str = DEFAULT_USER_ID,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> Any: ...

    def submit_quiz(
        self,
        thread_id: str,
        answer: QuizAnswer,
        *,
        user_id: str = DEFAULT_USER_ID,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> Any: ...

    def submit_deploy_decision(
        self,
        thread_id: str,
        decision: DeployDecision,
        *,
        user_id: str = DEFAULT_USER_ID,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> Any: ...


WorkflowFactory = Callable[[RuntimeSettings, Topology, str], WorkflowBackend]


def _load_default_workspace(settings: RuntimeSettings) -> WorkspaceBackend:
    return SandboxWorkspace(settings=settings)


def _default_workflow_factory(settings: RuntimeSettings, topology: Topology, model: str) -> WorkflowBackend:
    from no_quiz_no_prod.graph.workflow import NoQuizWorkflow  # deferred: importing LangGraph is slow

    return NoQuizWorkflow(settings=settings, topology=topology, model=model)


def _thread_state_error() -> type[Exception]:
    """The workflow's "this thread is not waiting for that decision" error (deferred import, as above)."""
    from no_quiz_no_prod.graph.workflow import ThreadStateError

    return ThreadStateError


# ---- typed errors over HTTP ------------------------------------------------------------------------------------------
BusyKind = Literal["run", "quiz", "deploy", "reset", "settings", "memory"]


class RunActiveError(Exception):
    """Another operation owns the workspace: one active run per workspace, answered with 409 ``run_active``."""

    def __init__(self, busy: BusyKind) -> None:
        self.busy: BusyKind = busy
        super().__init__(f"The workspace is busy ({busy} in progress). Retry once it is over.")


# One status per typed error code, so a refusal of the gate, a bad argument and a broken dependency never look alike.
_HTTP_STATUS_BY_ERROR_CODE: Final[dict[ToolErrorCode, int]] = {
    "invalid_argument": 422,
    "not_found": 404,
    "forbidden": 403,
    "gate_locked": 409,
    "quiz_exhausted": 409,
    "conflict": 409,
    "remote_diverged": 409,
    "run_active": 409,
    "rate_limited": 429,
    "timeout": 504,
    "upstream_error": 502,
    "sandbox_unavailable": 503,
    "internal_error": 500,
}


def _error_response(code: ToolErrorCode, message: str, *, status: int | None = None, **extra: Any) -> JSONResponse:
    """The one error body of the API: ``{"ok": false, "code", "message", ...}``."""
    return JSONResponse(
        status_code=status if status is not None else _HTTP_STATUS_BY_ERROR_CODE[code],
        content={"ok": False, "code": code, "message": message, **extra},
    )


# ---- request schemas -----------------------------------------------------------------------------------------------
class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: str = DEFAULT_SEED
    user_id: str = DEFAULT_USER_ID
    reset_skill_memory: bool = False


class RunStreamRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    thread_id: str | None = None
    user_id: str = DEFAULT_USER_ID
    model: str | None = None
    topology: Topology = "dual"
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=64)
    fault: FaultKind | None = None


class QuizSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_option: QuizOptionKey
    user_id: str = DEFAULT_USER_ID
    question_id: str | None = None
    model: str | None = None
    topology: Topology | None = None


class DeploySubmitRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: DeployAction
    expected_diff_sha256: Sha256Hex  # the diff the human was shown: mandatory, never defaulted server-side
    reason: str = ""
    user_id: str = DEFAULT_USER_ID
    model: str | None = None
    topology: Topology | None = None


ApiKeyProvider = Literal["openrouter", "anthropic", "notion", "github", "git_remote"]

_API_KEY_ENV_MAP: Final[dict[ApiKeyProvider, tuple[EnvVar, str]]] = {
    "openrouter": (EnvVar.OPENROUTER_API_KEY, "openrouter_api_key"),
    "anthropic": (EnvVar.ANTHROPIC_API_KEY, "anthropic_api_key"),
    "notion": (EnvVar.NOTION_API_KEY, "notion_api_key"),
    "github": (EnvVar.NO_QUIZ_GITHUB_TOKEN, "github_token"),
    "git_remote": (EnvVar.NO_QUIZ_GIT_REMOTE_URL, "git_remote_url"),  # not a secret, but handled write-only too
}
_PLAIN_SETTINGS: Final = frozenset({"git_remote_url"})  # every other updatable setting is a SecretStr
# The env var names are accepted as provider aliases (e.g. "NO_QUIZ_GITHUB_TOKEN").
_PROVIDER_ALIASES: Final[dict[str, ApiKeyProvider]] = {
    env_var.value: provider for provider, (env_var, _) in _API_KEY_ENV_MAP.items()
}


class ApiKeyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ApiKeyProvider
    api_key: str = Field(min_length=1, repr=False)
    # Changing where deploys are pushed must be an explicit act: the remote is saved only with this set to true.
    confirm_remote_change: bool = False

    @field_validator("provider", mode="before")
    @classmethod
    def _accept_env_var_names(cls, raw: Any) -> Any:
        return _PROVIDER_ALIASES.get(raw, raw) if isinstance(raw, str) else raw

    @field_validator("api_key")
    @classmethod
    def _validate_secret_token(cls, raw: str) -> str:
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("API key must not be empty or whitespace-only.")
        if "\n" in cleaned or "\r" in cleaned or "\x00" in cleaned:
            raise ValueError("API key must be a single-line token.")
        return cleaned


def _format_env_assignment(var_name: str, secret_value: str) -> str:
    if any(ch in secret_value for ch in (" ", "\t", "#", '"', "'", "\\")):
        escaped = secret_value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{var_name}="{escaped}"'
    return f"{var_name}={secret_value}"


def _upsert_env_secret(env_file: Path, env_var: EnvVar, secret_value: str) -> None:
    """Atomically write or update ``env_var`` in ``env_file`` with strict ``0o600`` permissions."""
    existing_lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []
    assignment = _format_env_assignment(env_var.value, secret_value)
    key_pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(env_var.value)}\s*=")

    updated_lines: list[str] = []
    replaced = False
    for line in existing_lines:
        if key_pattern.match(line):
            if not replaced:
                updated_lines.append(assignment)
                replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        updated_lines.append(assignment)

    rendered = "\n".join(updated_lines) + "\n"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = env_file.with_name(f".{env_file.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, env_file)
        os.chmod(env_file, 0o600)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---- pure serialization & helper functions -------------------------------------------------------------------------
def _normalize_skill_profile(records: list[SkillProfileRecord], user_id: str) -> list[SkillProfileRecord]:
    by_tag = {rec.skill_tag: rec for rec in records}
    return [by_tag.get(tag, SkillProfileRecord(skill_tag=tag, user_id=user_id)) for tag in ALL_SKILL_TAGS]


def _sanitize_public_quiz(quiz: PublicQuizView | dict[str, Any] | None) -> dict[str, Any] | None:
    """Guarantee ``correct_option`` and ``explanation_by_option`` never leave the server."""
    if quiz is None:
        return None
    if isinstance(quiz, PublicQuizView):
        public = quiz
    else:
        allowed = {k: v for k, v in quiz.items() if k in PublicQuizView.model_fields}
        public = PublicQuizView.model_validate(allowed)
    payload = public.model_dump(mode="json")
    payload.pop("correct_option", None)
    payload.pop("explanation_by_option", None)
    return payload


def _read_notion_metadata(notion_ids_file: Path) -> dict[str, str]:
    if not notion_ids_file.is_file():
        return {}
    try:
        raw = json.loads(notion_ids_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v.strip()}


def _resolve_hub_url(notion_ids: dict[str, str]) -> str | None:
    direct = notion_ids.get("hub_url") or notion_ids.get("HUB_URL")
    if direct:
        return direct
    page_id = notion_ids.get("hub_page_id") or notion_ids.get("HUB_PAGE_ID")
    if page_id:
        compact = page_id.replace("-", "")
        return f"https://www.notion.so/{compact}"
    return None


def _load_notion_rules(
    workspace: WorkspaceBackend | None = None, *, seed_index_override: Path | None = None
) -> list[NotionRule]:
    list_rules_fn = getattr(workspace, "list_rules", None) if workspace is not None else None
    if callable(list_rules_fn):
        items = list_rules_fn()
        return [r if isinstance(r, NotionRule) else NotionRule.model_validate(r) for r in items]

    seed_index = seed_index_override or (PROJECT_ROOT / "seeds" / DEFAULT_SEED / "rules" / "index.json")
    if seed_index.is_file():
        try:
            items = json.loads(seed_index.read_text(encoding="utf-8"))
            if isinstance(items, list):
                resolved: list[NotionRule] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    entry = dict(item)
                    body_file = entry.pop("body_file", None)
                    if not entry.get("body_markdown") and isinstance(body_file, str) and body_file.strip():
                        md_path = seed_index.parent / body_file
                        if md_path.is_file():
                            entry["body_markdown"] = md_path.read_text(encoding="utf-8")
                    allowed = {k: v for k, v in entry.items() if k in NotionRule.model_fields}
                    resolved.append(NotionRule.model_validate(allowed))
                return resolved
        except (OSError, ValueError, ValidationError):
            return []
    return []


def _build_models_catalogue(settings: RuntimeSettings) -> dict[str, Any]:
    providers = settings.providers
    default_model = (
        settings.openrouter_model if providers["openrouter"] or not providers["anthropic"] else settings.anthropic_model
    )
    options: list[dict[str, Any]] = [
        {
            "id": settings.openrouter_model,
            "label": f"OpenRouter · {settings.openrouter_model}",
            "provider": "openrouter",
            "enabled": providers["openrouter"],
        }
    ]
    seen_ids = {settings.openrouter_model}
    for model_id, label in SECONDARY_OPENROUTER_MODELS:
        if model_id not in seen_ids:
            options.append(
                {
                    "id": model_id,
                    "label": label,
                    "provider": "openrouter",
                    "enabled": providers["openrouter"],
                }
            )
            seen_ids.add(model_id)
    if settings.anthropic_model not in seen_ids:
        options.append(
            {
                "id": settings.anthropic_model,
                "label": f"Anthropic · {settings.anthropic_model}",
                "provider": "anthropic",
                "enabled": providers["anthropic"],
            }
        )
    return {
        "default_model": default_model,
        "default_topology": "dual",
        "default_max_steps": DEFAULT_MAX_STEPS,
        "available_providers": providers,
        "options": options,
    }


def _dump_or_none(model: BaseModel | None) -> dict[str, Any] | None:
    return model.model_dump(mode="json") if model is not None else None


def _pending_release_of(outcome: WorkflowOutcomeLike) -> PendingRelease | None:
    """The failed publish an outcome carries: on its ``release_retry`` interrupt, or on the outcome itself."""
    if outcome.interrupt is not None and outcome.interrupt.release is not None:
        return outcome.interrupt.release
    return outcome.pending_release


def _resolve_default_eval_dir() -> Path:
    """Newest run directory under ``eval/runs/`` that holds result rows, else ``LATEST_CAMPAIGN_DIR``."""
    if RUNS_DIR.is_dir():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if run_dir.is_dir() and ((run_dir / "results.jsonl").is_file() or any(run_dir.glob("results_*.jsonl"))):
                return run_dir
    return LATEST_CAMPAIGN_DIR


def _sse_frame(event: RunEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


async def _invoke_maybe_async(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# ---- studio runtime state manager ----------------------------------------------------------------------------------
@dataclass
class _ThreadSession:
    thread_id: str
    model: str
    topology: Topology
    user_id: str = DEFAULT_USER_ID
    interrupt: HumanInterrupt | None = None
    stop_reason: StopReason | None = None
    budget: RunBudget | None = None
    final_answer: str = ""
    expected_diff_sha256: str | None = None
    deployed_diff_sha256: str | None = None
    pending_release: PendingRelease | None = None  # a deploy whose push failed, waiting for re-approval or rejection
    workflow: WorkflowBackend | None = None


class StudioManager:
    """Stateful coordinator bridging HTTP/SSE requests to ``SandboxWorkspace`` and ``NoQuizWorkflow``."""

    def __init__(
        self,
        *,
        settings: RuntimeSettings | None = None,
        workspace: WorkspaceBackend | None = None,
        workflow_factory: WorkflowFactory | None = None,
        eval_dir: Path | None = None,
        notion_ids_file: Path | None = None,
        env_file: Path | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self._workspace = workspace
        self._workflow_factory = workflow_factory or _default_workflow_factory
        self._explicit_eval_dir = eval_dir
        self.eval_dir = eval_dir or _resolve_default_eval_dir()
        self.notion_ids_file = notion_ids_file or NOTION_IDS_FILE
        self.env_file = env_file or ENV_FILE
        self.sessions: dict[str, _ThreadSession] = {}
        self.latest_thread_id: str | None = None
        self._busy: BusyKind | None = None
        self._busy_lock = threading.Lock()
        self._background: set[asyncio.Task[Any]] = set()

    # ---- one active operation per workspace (run, quiz answer, deploy, reset, settings, memory) -------------------------
    @property
    def busy(self) -> BusyKind | None:
        return self._busy

    def begin(self, kind: BusyKind) -> None:
        """Take the workspace for one operation, or raise ``RunActiveError``. Never blocks, holds no lock afterwards."""
        with self._busy_lock:
            if self._busy is not None:
                raise RunActiveError(self._busy)
            self._busy = kind

    def end(self) -> None:
        with self._busy_lock:
            self._busy = None

    @contextmanager
    def exclusive(self, kind: BusyKind) -> Iterator[None]:
        self.begin(kind)
        try:
            yield
        finally:
            self.end()

    def keep_alive(self, task: asyncio.Task[Any]) -> None:
        """Hold a reference to a run that must finish even when its HTTP client is gone."""
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def record_outcome(self, session: _ThreadSession, outcome: WorkflowOutcomeLike) -> None:
        """What the thread is now waiting for (or how it ended), as every later request and reload must see it."""
        session.interrupt = outcome.interrupt
        session.stop_reason = outcome.stop_reason
        session.budget = outcome.budget
        session.final_answer = outcome.final_answer
        session.pending_release = _pending_release_of(outcome)
        gated = (outcome.interrupt.quiz or outcome.interrupt.gate) if outcome.interrupt else None
        if gated is not None:
            session.expected_diff_sha256 = gated.diff_sha256

    @property
    def provider_status(self) -> dict[str, bool]:
        llm = self.settings.providers
        return {
            "openrouter": bool(llm["openrouter"]),
            "anthropic": bool(llm["anthropic"]),
            "notion": self.settings.notion_api_key is not None,
        }

    def update_api_key(self, provider: ApiKeyProvider, api_key: str) -> dict[str, bool]:
        """Persist ``api_key`` to ``self.env_file`` (0o600) and refresh ``self.settings`` in memory."""
        env_var, settings_attr = _API_KEY_ENV_MAP[provider]
        _upsert_env_secret(self.env_file, env_var, api_key)
        if self.env_file == ENV_FILE and provider != "github":
            # The release token stays out of os.environ: no child process of the Studio may inherit it.
            os.environ[env_var.value] = api_key

        notion_ids = _read_notion_metadata(self.notion_ids_file)
        updates: dict[str, Any] = {settings_attr: api_key if settings_attr in _PLAIN_SETTINGS else SecretStr(api_key)}
        if not self.settings.notion_ship_log_db_id and notion_ids.get("ship_log_db_id"):
            updates["notion_ship_log_db_id"] = notion_ids["ship_log_db_id"]
        if not self.settings.notion_rules_db_id and notion_ids.get("rules_db_id"):
            updates["notion_rules_db_id"] = notion_ids["rules_db_id"]
        if not self.settings.notion_skills_db_id and notion_ids.get("skill_profile_db_id"):
            updates["notion_skills_db_id"] = notion_ids["skill_profile_db_id"]

        self.settings = self.settings.model_copy(update=updates)
        if self._workspace is not None:
            if hasattr(self._workspace, "settings"):
                self._workspace.settings = self.settings
            if hasattr(self._workspace, "notion"):
                old_notion = getattr(self._workspace, "notion", None)
                close_fn = getattr(old_notion, "close", None)
                if callable(close_fn):
                    close_fn()
                self._workspace.notion = NotionDatabaseSync(self.settings)
            if hasattr(self._workspace, "release"):
                self._workspace.release = GitReleasePublisher(self.settings)
        for session in self.sessions.values():
            session.workflow = None
        return self.provider_status

    @property
    def workspace(self) -> WorkspaceBackend:
        if self._workspace is None:
            self._workspace = _load_default_workspace(self.settings)
        return self._workspace

    def get_workflow_for_session(self, session: _ThreadSession) -> WorkflowBackend:
        if session.workflow is None:
            session.workflow = self._workflow_factory(self.settings, session.topology, session.model)
        return session.workflow

    @property
    def _threads_meta_path(self) -> Path:
        return self.settings.workspace_dir / "ui_threads.json"

    def persist_thread_meta(self, thread_id: str, *, model: str, topology: Topology) -> None:
        data: dict[str, dict[str, str]] = {}
        if self._threads_meta_path.is_file():
            try:
                loaded = json.loads(self._threads_meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError):
                data = {}
        data[thread_id] = {"model": model, "topology": topology}
        self._threads_meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._threads_meta_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    def resolve_thread_meta(
        self, thread_id: str, *, model_hint: str | None = None, topology_hint: Topology | None = None
    ) -> tuple[str, Topology]:
        saved: dict[str, str] = {}
        if self._threads_meta_path.is_file():
            try:
                loaded = json.loads(self._threads_meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get(thread_id), dict):
                    saved = loaded[thread_id]
            except (OSError, ValueError):
                saved = {}
        catalogue = _build_models_catalogue(self.settings)
        model = model_hint or saved.get("model") or str(catalogue["default_model"])
        saved_topo = saved.get("topology")
        topology: Topology = (
            topology_hint if topology_hint is not None else ("single" if saved_topo == "single" else "dual")
        )
        return model, topology

    def clear_thread_meta(self) -> None:
        if self._threads_meta_path.is_file():
            self._threads_meta_path.unlink()

    def snapshot_artifacts(
        self, thread_id: str | None = None, *, expected_diff_sha256: str | None = None
    ) -> tuple[DiffReport, PytestReport | None, ShipLogEntry | None, PublicQuizView | None, DeployGateStatus]:
        ws = self.workspace
        diff = ws.get_diff_report()
        pytest_rep = ws.get_latest_pytest()
        ship_log = ws.get_latest_ship_log(thread_id=thread_id)
        if ship_log is None and thread_id is not None:
            ship_log = ws.get_latest_ship_log(thread_id=None)

        target_diff_sha = expected_diff_sha256 or (ship_log.diff_sha256 if ship_log and ship_log.diff_sha256 else None)
        quiz = ws.get_latest_public_quiz(thread_id=thread_id, diff_sha256=target_diff_sha)
        if quiz is None and thread_id is not None:
            quiz = ws.get_latest_public_quiz(thread_id=thread_id)

        session = self.sessions.get(thread_id) if thread_id else None
        if session is not None and session.deployed_diff_sha256 and diff.trees_identical:
            locks = ws.gate_status_of(session.deployed_diff_sha256)  # the receipt of the diff this thread shipped
        else:
            # A read-only view: with no decision pending, the locks shown are those of what is staged right now.
            locks = ws.get_gate_status(expected_diff_sha256 or diff.diff_sha256)
        return diff, pytest_rep, ship_log, quiz, locks

    def build_state_payload(self, *, user_id: str = DEFAULT_USER_ID, thread_id: str | None = None) -> dict[str, Any]:
        active_tid = thread_id or self.latest_thread_id
        session = self.sessions.get(active_tid) if active_tid else None
        expected_sha = session.expected_diff_sha256 if session else None

        diff, pytest_rep, ship_log, quiz, locks = self.snapshot_artifacts(active_tid, expected_diff_sha256=expected_sha)
        notion_ids = _read_notion_metadata(self.notion_ids_file)
        rules = _load_notion_rules(self.workspace)
        skills = _normalize_skill_profile(self.workspace.get_skill_profile(user_id=user_id), user_id)

        interrupt_dump = session.interrupt.model_dump(mode="json") if session and session.interrupt else None
        if interrupt_dump and interrupt_dump.get("quiz") is not None:
            interrupt_dump["quiz"] = _sanitize_public_quiz(interrupt_dump["quiz"])

        return {
            "workspace": {
                "diff": diff.model_dump(mode="json"),
                "pytest": pytest_rep.model_dump(mode="json") if pytest_rep else None,
            },
            "models": _build_models_catalogue(self.settings),
            "notion": {
                "notion_enabled": self.settings.notion_enabled,
                "notion_key_configured": self.settings.notion_api_key is not None,
                "hub_url": _resolve_hub_url(notion_ids),
                "rules": [r.model_dump(mode="json") for r in rules],
            },
            "active_run": {
                "thread_id": active_tid,
                "interrupt": interrupt_dump,
                "stop_reason": session.stop_reason if session else None,
                "ship_log": ship_log.model_dump(mode="json") if ship_log else None,
                "quiz": _sanitize_public_quiz(quiz),
                "locks": locks.model_dump(mode="json"),
                "pending_release": _dump_or_none(session.pending_release if session else None),
                "busy": self.busy,
            },
            "skill_profile": [rec.model_dump(mode="json") for rec in skills],
            "git": describe_release_remote(self.settings),
        }

    def load_eval_results(self) -> dict[str, Any]:
        target_dir = self._explicit_eval_dir or _resolve_default_eval_dir()
        self.eval_dir = target_dir
        manifest = None
        if target_dir.is_dir():
            try:
                manifest = load_manifest(target_dir)
            except (OSError, ValidationError, ValueError):
                manifest = None
        stale = bool(manifest.stale) if manifest is not None else False
        stale_reason = manifest.stale_reason if manifest is not None else None

        candidate_files: list[Path] = []
        if target_dir.is_dir():
            primary = target_dir / "results.jsonl"
            if primary.is_file():
                candidate_files.append(primary)
            for extra in sorted(target_dir.glob("results_*.jsonl")):
                if extra.is_file() and extra != primary:
                    candidate_files.append(extra)

        seen_keys: set[tuple[str, Topology, str, int]] = set()
        runs: list[EvalRunResult] = []
        used_sources: list[str] = []
        invalid_rows = 0  # lines that are not a valid EvalRunResult: skipped and reported, never fatal
        for path in candidate_files:
            added = 0
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    parsed_run = EvalRunResult.model_validate_json(stripped)
                except ValidationError:
                    invalid_rows += 1
                    continue
                dedupe_key = (parsed_run.model, parsed_run.topology, parsed_run.id, parsed_run.repeat)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                runs.append(parsed_run)
                added += 1
            if added > 0:
                try:
                    rel = path.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rel = path.name
                used_sources.append(rel)

        if not runs:
            return {
                "available": False,
                "source_files": [],
                "invalid_rows": invalid_rows,
                "stale": stale,
                "stale_reason": stale_reason,
                "summary": {
                    "total_runs": 0,
                    "passed_runs": 0,
                    "pass_rate": 0.0,
                    "k_repeats": 0,
                    "pass_at_k": 0.0,
                    "pass_hat_k": 0.0,
                    "by_outcome": {outcome: 0 for outcome in ALL_EVAL_OUTCOMES},
                    "forbidden_calls_count": 0,
                    "hidden_failures_count": 0,
                    "avg_steps": 0.0,
                    "avg_latency_s": 0.0,
                    "avg_cost_usd": 0.0,
                    "by_kind": {kind: {"total": 0, "passed": 0, "pass_rate": 0.0} for kind in ALL_SCENARIO_KINDS},
                    "cohorts": [],
                    "showcase_passed_key": None,
                    "showcase_failed_or_blocked_key": None,
                },
                "runs": [],
            }

        # Group cohorts by (model, topology) preserving appearance order
        cohort_groups: dict[tuple[str, Topology], list[EvalRunResult]] = {}
        for run in runs:
            cohort_groups.setdefault((run.model, run.topology), []).append(run)

        primary_cohort_runs = next(iter(cohort_groups.values()))
        scenario_repeats: dict[str, int] = {}
        for run in primary_cohort_runs:
            scenario_repeats[run.id] = scenario_repeats.get(run.id, 0) + 1
        k_repeats = max(scenario_repeats.values(), default=1)

        total_runs = len(runs)
        passed_runs = sum(1 for r in runs if r.passed)
        by_outcome = {outcome: sum(1 for r in runs if r.outcome == outcome) for outcome in ALL_EVAL_OUTCOMES}

        by_kind: dict[str, dict[str, float | int]] = {}
        for kind in ALL_SCENARIO_KINDS:
            kind_runs = [r for r in runs if r.kind == kind]
            kind_total = len(kind_runs)
            kind_passed = sum(1 for r in kind_runs if r.passed)
            by_kind[kind] = {
                "total": kind_total,
                "passed": kind_passed,
                "pass_rate": round(kind_passed / kind_total, 4) if kind_total else 0.0,
            }

        cohorts_payload: list[dict[str, Any]] = []
        for (model_id, topo), c_runs in cohort_groups.items():
            c_total = len(c_runs)
            c_passed = sum(1 for r in c_runs if r.passed)
            c_by_kind: dict[str, float] = {}
            for kind in ALL_SCENARIO_KINDS:
                k_sub = [r for r in c_runs if r.kind == kind]
                c_by_kind[kind] = round(sum(1 for r in k_sub if r.passed) / len(k_sub), 4) if k_sub else 0.0
            cohorts_payload.append(
                {
                    "model": model_id,
                    "topology": topo,
                    "total": c_total,
                    "passed": c_passed,
                    "pass_rate": round(c_passed / c_total, 4) if c_total else 0.0,
                    "pass_at_k": round(pass_at_k(c_runs), 4),
                    "pass_hat_k": round(pass_hat_k(c_runs), 4),
                    "by_kind": c_by_kind,
                    "avg_steps": round(sum(r.steps for r in c_runs) / c_total, 2) if c_total else 0.0,
                    "avg_latency_s": round(sum(r.latency_s for r in c_runs) / c_total, 2) if c_total else 0.0,
                    "avg_cost_usd": round(sum(r.cost_usd for r in c_runs) / c_total, 6) if c_total else 0.0,
                }
            )

        showcase_passed = next(
            (r.key for r in runs if r.passed and r.kind == "normal" and r.final_state.deployed),
            next((r.key for r in runs if r.passed), None),
        )
        showcase_failed = next(
            (r.key for r in runs if not r.passed),
            next(
                (
                    r.key
                    for r in runs
                    if r.stop_reason in ("quiz_failed", "human_rejected", "refused", "release_failed")
                    or r.final_state.ship_log_status == "Blocked"
                ),
                None,
            ),
        )

        return {
            "available": True,
            "source_files": used_sources,
            "invalid_rows": invalid_rows,
            "stale": stale,
            "stale_reason": stale_reason,
            "summary": {
                "total_runs": total_runs,
                "passed_runs": passed_runs,
                "pass_rate": round(passed_runs / total_runs, 4),
                "k_repeats": k_repeats,
                "pass_at_k": round(pass_at_k(primary_cohort_runs), 4),
                "pass_hat_k": round(pass_hat_k(primary_cohort_runs), 4),
                "by_outcome": by_outcome,
                "forbidden_calls_count": sum(1 for r in runs if r.forbidden_called),
                "hidden_failures_count": sum(1 for r in runs if is_hidden_failure(r)),
                "avg_steps": round(sum(r.steps for r in runs) / total_runs, 2),
                "avg_latency_s": round(sum(r.latency_s for r in runs) / total_runs, 2),
                "avg_cost_usd": round(sum(r.cost_usd for r in runs) / total_runs, 6),
                "by_kind": by_kind,
                "cohorts": cohorts_payload,
                "showcase_passed_key": showcase_passed,
                "showcase_failed_or_blocked_key": showcase_failed,
            },
            # hidden_failure as the scorer counts it (deploy_push_mismatch included); legacy rows are unaffected
            "runs": [{**r.model_dump(mode="json"), "hidden_failure": is_hidden_failure(r)} for r in runs],
        }


# ---- FastAPI application factory -----------------------------------------------------------------------------------
def create_app(
    *,
    settings: RuntimeSettings | None = None,
    workspace: WorkspaceBackend | None = None,
    workflow_factory: WorkflowFactory | None = None,
    eval_dir: Path | None = None,
    notion_ids_file: Path | None = None,
    env_file: Path | None = None,
    allowed_hosts: Sequence[str] | None = None,
    session_token: str | None = None,
) -> FastAPI:
    """Create the FastAPI Studio server with injectable workspace/workflow backends for testing.

    ``allowed_hosts`` defaults to the loopback names plus ``NO_QUIZ_UI_ALLOWED_HOSTS``; ``session_token`` defaults
    to a fresh random token. Both feed ``StudioGuard``, which every request crosses before any route.
    """
    manager = StudioManager(
        settings=settings,
        workspace=workspace,
        workflow_factory=workflow_factory,
        eval_dir=eval_dir,
        notion_ids_file=notion_ids_file,
        env_file=env_file,
    )

    app = FastAPI(
        title="No-Quiz-No-Prod Studio",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.studio = manager
    app.state.session_token = session_token or new_session_token()
    app.state.allowed_hosts = (
        normalize_allowed_hosts(allowed_hosts) if allowed_hosts is not None else allowed_hosts_from_env()
    )
    app.add_middleware(StudioGuard, allowed_hosts=app.state.allowed_hosts, session_token=app.state.session_token)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(RunActiveError)
    async def on_run_active(_request: Request, error: RunActiveError) -> JSONResponse:
        return _error_response("run_active", str(error), busy=error.busy)

    @app.exception_handler(WorkspaceError)
    async def on_workspace_error(_request: Request, error: WorkspaceError) -> JSONResponse:
        # A typed backend refusal or failure keeps its own code and status: it is never relabelled "gate_locked".
        return _error_response(error.payload.code, error.payload.message)

    @app.get("/", response_model=None)
    async def serve_index() -> HTMLResponse:
        # The session token reaches the browser only inside this page, which another origin cannot read.
        page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(
            page.replace(SESSION_TOKEN_PLACEHOLDER, app.state.session_token),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/health")
    async def get_health() -> dict[str, Any]:
        try:
            ws_rel = manager.settings.workspace_dir.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            ws_rel = str(manager.settings.workspace_dir)
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": __version__,
            "workspace_dir": ws_rel,
            "providers": manager.settings.providers,
            "git": describe_release_remote(manager.settings),
        }

    @app.get("/api/state")
    async def get_state(
        user_id: str = Query(default=DEFAULT_USER_ID),
        thread_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return manager.build_state_payload(user_id=user_id, thread_id=thread_id)

    @app.post("/api/apikey", response_model=None)
    async def post_apikey(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            return _error_response("invalid_argument", "Content-Type must be application/json.", status=415)
        try:
            raw_body = await request.json()
        except ValueError:
            return _error_response("invalid_argument", "Invalid JSON request body.")
        if not isinstance(raw_body, dict):
            return _error_response("invalid_argument", "Expected JSON object payload.")
        try:
            req = ApiKeyUpdateRequest.model_validate(raw_body)
        except ValidationError as exc:
            first_msg = exc.errors()[0].get("msg", "Invalid API key update request.") if exc.errors() else "Invalid."
            return _error_response("invalid_argument", str(first_msg))
        if req.provider == "git_remote":
            # Validate against the configured allowed hosts (not defaults); WorkspaceError -> 422 invalid_argument.
            candidate = manager.settings.model_copy(update={"git_remote_url": req.api_key})
            GitReleasePublisher(candidate).resolve_remote()
            if not req.confirm_remote_change:
                return _error_response(
                    "invalid_argument",
                    "Changing the release remote redirects every later deploy: "
                    "resend with confirm_remote_change=true to confirm.",
                    confirmation_required=True,
                )

        # Settings are never swapped under a run, a quiz answer or a deploy (the remote least of all): 409 run_active.
        with manager.exclusive("settings"):
            providers = await _invoke_maybe_async(manager.update_api_key, req.provider, req.api_key)
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "provider": req.provider,
                "providers": providers,
                "notion_enabled": manager.settings.notion_enabled,
                "models": _build_models_catalogue(manager.settings),
                "git": describe_release_remote(manager.settings),
            },
        )

    @app.post("/api/reset")
    async def post_reset(req: ResetRequest) -> dict[str, Any]:
        with manager.exclusive("reset"):  # refused with 409 run_active while a run streams or a deploy is in flight
            final_state = await _invoke_maybe_async(
                manager.workspace.reset,
                req.seed,
                preserve_skill_memory=not req.reset_skill_memory,
            )
            manager.sessions.clear()
            manager.clear_thread_meta()
            manager.latest_thread_id = None
        state_payload = manager.build_state_payload(user_id=req.user_id, thread_id=None)
        return {
            "reset": True,
            "seed": req.seed,
            "prod_tree_sha256": final_state.prod_tree_sha256,
            "staging_tree_sha256": final_state.staging_tree_sha256,
            "skill_memory_reset": req.reset_skill_memory,
            "state": state_payload,
        }

    @app.post("/api/runs/stream", response_model=None)
    async def post_runs_stream(req: RunStreamRequest) -> Response:
        catalogue = _build_models_catalogue(manager.settings)
        chosen_model = req.model or str(catalogue["default_model"])
        thread_id = req.thread_id or f"thr_{uuid.uuid4().hex[:8]}"
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()

        # A release remote that cannot work is reported now, before any model call (typed error, see the handler).
        validate_release_config(manager.settings)

        def on_event(ev: RunEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        async def run_worker(session: _ThreadSession) -> tuple[WorkflowOutcomeLike | None, Exception | None]:
            """Drive the workflow to its next pause or end.

            The worker, not the response stream, owns the run slot and the recorded outcome: both stay right when
            the browser disconnects before the last frame.
            """
            try:
                wf = manager.get_workflow_for_session(session)
                outcome = await _invoke_maybe_async(
                    wf.start_run,
                    req.prompt,
                    thread_id=thread_id,
                    user_id=req.user_id,
                    max_steps=req.max_steps,
                    on_event=on_event,
                )
            except Exception as exc:  # noqa: BLE001 — not swallowed: recorded and sent to the client as a run_error frame
                session.stop_reason = "api_error"
                session.interrupt = None
                return None, exc
            else:
                manager.record_outcome(session, outcome)
                return outcome, None
            finally:
                manager.end()
                queue.put_nowait(None)

        manager.begin("run")  # 409 run_active when another run, quiz answer, deploy or reset owns the workspace
        try:
            session = manager.sessions.get(thread_id)
            if session is None:
                session = _ThreadSession(
                    thread_id=thread_id,
                    model=chosen_model,
                    topology=req.topology,
                    user_id=req.user_id,
                )
                manager.sessions[thread_id] = session
            else:
                session.model = chosen_model
                session.topology = req.topology
                session.user_id = req.user_id
                session.interrupt = None
                session.stop_reason = None
            manager.persist_thread_meta(thread_id, model=chosen_model, topology=req.topology)
            manager.latest_thread_id = thread_id

            # Arm transient fault when requested on a clean workspace via #14 contract reset()
            if req.fault is not None and manager.workspace.get_diff_report().trees_identical:
                await _invoke_maybe_async(
                    manager.workspace.reset,
                    DEFAULT_SEED,
                    fault=req.fault,
                    preserve_skill_memory=True,
                )
            worker_task = asyncio.create_task(run_worker(session))
        except BaseException:
            manager.end()  # the worker never started, so nothing else would release the slot
            raise
        manager.keep_alive(worker_task)

        async def event_generator() -> AsyncIterator[str]:
            seq = 0
            initial_budget = RunBudget(max_steps=req.max_steps)
            started_event = RunEvent(
                type="run_started",
                thread_id=thread_id,
                seq=seq,
                model=chosen_model,
                topology=req.topology,
                budget=initial_budget,
            )
            yield _sse_frame(started_event)

            deferred_terminal: RunEvent | None = None

            while True:
                incoming = await queue.get()
                if incoming is None:
                    break
                if incoming.type == "run_started":
                    continue
                if incoming.type in ("run_completed", "run_error"):
                    deferred_terminal = incoming
                    continue

                seq += 1
                resequenced = RunEvent.model_validate({**incoming.model_dump(), "seq": seq, "thread_id": thread_id})
                yield _sse_frame(resequenced)

                if incoming.type == "step" and incoming.step and incoming.step.tool in ARTIFACT_TRIGGER_TOOLS:
                    diff, pytest_rep, ship_log, quiz, locks = manager.snapshot_artifacts(thread_id)
                    seq += 1
                    art_event = RunEvent(
                        type="artifact_updated",
                        thread_id=thread_id,
                        seq=seq,
                        budget=incoming.budget,
                        diff=diff,
                        pytest=pytest_rep,
                        ship_log=ship_log,
                        quiz=quiz,
                        locks=locks,
                    )
                    yield _sse_frame(art_event)

            outcome, error = await worker_task
            if error is not None or outcome is None:
                seq += 1
                err_event = RunEvent(
                    type="run_error",
                    thread_id=thread_id,
                    seq=seq,
                    stop_reason="api_error",
                    text=str(error) or "Workflow execution failed.",
                )
                yield _sse_frame(err_event)
                return

            diff, pytest_rep, ship_log, quiz, locks = manager.snapshot_artifacts(
                thread_id, expected_diff_sha256=session.expected_diff_sha256
            )
            if quiz is None and outcome.interrupt and outcome.interrupt.quiz:
                quiz = outcome.interrupt.quiz

            seq += 1
            final_artifacts = RunEvent(
                type="artifact_updated",
                thread_id=thread_id,
                seq=seq,
                budget=outcome.budget,
                diff=diff,
                pytest=pytest_rep,
                ship_log=ship_log,
                quiz=quiz,
                locks=locks,
            )
            yield _sse_frame(final_artifacts)

            seq += 1
            if deferred_terminal is not None:
                term_event = RunEvent.model_validate(
                    {**deferred_terminal.model_dump(), "seq": seq, "thread_id": thread_id}
                )
            elif outcome.stop_reason in ERROR_STOP_REASONS:
                term_event = RunEvent(
                    type="run_error",
                    thread_id=thread_id,
                    seq=seq,
                    budget=outcome.budget,
                    stop_reason=outcome.stop_reason,
                    text=outcome.final_answer or "Workflow execution failed.",
                )
            elif outcome.interrupt is not None:
                term_event = RunEvent(
                    type="run_completed",
                    thread_id=thread_id,
                    seq=seq,
                    budget=outcome.budget,
                    interrupt=outcome.interrupt,
                    stop_reason=None,
                    text=outcome.final_answer or outcome.interrupt.message,
                )
            else:
                term_event = RunEvent(
                    type="run_completed",
                    thread_id=thread_id,
                    seq=seq,
                    budget=outcome.budget,
                    interrupt=None,
                    stop_reason=outcome.stop_reason or "completed",
                    text=outcome.final_answer,
                    pending_release=_pending_release_of(outcome),
                )
            yield _sse_frame(term_event)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/runs/{thread_id}/quiz", response_model=None)
    async def post_submit_quiz(
        thread_id: str,
        req: QuizSubmitRequest,
        request: Request,
        stream: bool = Query(default=False),
    ) -> Response:
        wants_sse = stream or "text/event-stream" in request.headers.get("accept", "")
        with manager.exclusive("quiz"):
            return await submit_quiz(thread_id, req, wants_sse=wants_sse)

    async def submit_quiz(thread_id: str, req: QuizSubmitRequest, *, wants_sse: bool) -> Response:
        answer = QuizAnswer(selected_option=req.selected_option)
        session = manager.sessions.get(thread_id)

        remediation_events: list[RunEvent] = []

        def collect_event(ev: RunEvent) -> None:
            remediation_events.append(ev)

        if session is None:
            resolved_model, resolved_topology = manager.resolve_thread_meta(
                thread_id, model_hint=req.model, topology_hint=req.topology
            )
            session = _ThreadSession(
                thread_id=thread_id,
                model=resolved_model,
                topology=resolved_topology,
                user_id=req.user_id,
            )

        wf = manager.get_workflow_for_session(session)
        try:
            grade, outcome = await _invoke_maybe_async(
                wf.submit_quiz,
                thread_id,
                answer,
                user_id=req.user_id,
                on_event=collect_event,
            )
        except _thread_state_error() as exc:
            # Only "this thread has no quiz to answer" is a 409; any other failure keeps its own type and status.
            return _error_response("conflict", f"Thread '{thread_id}' is not waiting for a quiz answer: {exc}")

        manager.sessions[thread_id] = session
        manager.record_outcome(session, outcome)
        session.expected_diff_sha256 = grade.diff_sha256
        interrupt = outcome.interrupt
        stop_reason = outcome.stop_reason
        budget = outcome.budget
        remediation_trace = list(outcome.trace)

        manager.latest_thread_id = thread_id
        diff, pytest_rep, ship_log, latest_quiz, locks = manager.snapshot_artifacts(
            thread_id, expected_diff_sha256=grade.diff_sha256
        )

        next_quiz_view: PublicQuizView | None = None
        if grade.remediation_required:
            if interrupt and interrupt.quiz:
                next_quiz_view = interrupt.quiz
            elif latest_quiz and latest_quiz.question_id != grade.question_id:
                next_quiz_view = latest_quiz

        skills = _normalize_skill_profile(manager.workspace.get_skill_profile(user_id=req.user_id), req.user_id)
        interrupt_dump = interrupt.model_dump(mode="json") if interrupt else None
        if interrupt_dump and interrupt_dump.get("quiz") is not None:
            interrupt_dump["quiz"] = _sanitize_public_quiz(interrupt_dump["quiz"])

        step_traces = [
            ev.step.model_dump(mode="json") for ev in remediation_events if ev.type == "step" and ev.step is not None
        ] or [t.model_dump(mode="json") for t in remediation_trace]

        response_payload: dict[str, Any] = {
            "grade": grade.model_dump(mode="json"),
            "ship_log": ship_log.model_dump(mode="json") if ship_log else None,
            "next_quiz": _sanitize_public_quiz(next_quiz_view),
            "interrupt": interrupt_dump,
            "locks": locks.model_dump(mode="json"),
            "stop_reason": stop_reason,
            "trace": step_traces,
            "budget": budget.model_dump(mode="json") if budget else None,
            "skill_profile": [rec.model_dump(mode="json") for rec in skills],
        }

        if not wants_sse:
            return JSONResponse(content=response_payload, status_code=200)

        async def quiz_sse_generator() -> AsyncIterator[str]:
            seq = 0
            for ev in remediation_events:
                if ev.type == "step" and ev.step is not None:
                    seq += 1
                    yield _sse_frame(
                        RunEvent(
                            type="step",
                            thread_id=thread_id,
                            seq=seq,
                            budget=ev.budget or budget,
                            step=ev.step,
                        )
                    )
            seq += 1
            yield _sse_frame(
                RunEvent(
                    type="artifact_updated",
                    thread_id=thread_id,
                    seq=seq,
                    budget=budget,
                    diff=diff,
                    pytest=pytest_rep,
                    ship_log=ship_log,
                    quiz=next_quiz_view,
                    locks=locks,
                )
            )
            seq += 1
            if interrupt is not None:
                yield _sse_frame(
                    RunEvent(
                        type="run_completed",
                        thread_id=thread_id,
                        seq=seq,
                        budget=budget,
                        interrupt=interrupt,
                        stop_reason=None,
                        text=grade.explanation,
                    )
                )
            else:
                yield _sse_frame(
                    RunEvent(
                        type="run_completed",
                        thread_id=thread_id,
                        seq=seq,
                        budget=budget,
                        interrupt=None,
                        stop_reason=stop_reason or "quiz_failed",
                        text=grade.explanation,
                    )
                )

        return StreamingResponse(
            quiz_sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{thread_id}/deploy", response_model=None)
    async def post_submit_deploy(thread_id: str, req: DeploySubmitRequest) -> Response:
        with manager.exclusive("deploy"):
            return await submit_deploy(thread_id, req)

    async def submit_deploy(thread_id: str, req: DeploySubmitRequest) -> Response:
        # The decision names the diff the human was shown (required by the request model): nothing is defaulted here.
        decision = DeployDecision(action=req.action, expected_diff_sha256=req.expected_diff_sha256, reason=req.reason)
        expected_sha = decision.expected_diff_sha256
        pre_diff = manager.workspace.get_diff_report()
        pre_locks = manager.workspace.get_gate_status(expected_sha)

        if decision.approved:
            pre_missing = [lock for lock in pre_locks.missing_locks if lock != "human_approved"]
            if pre_locks.blocked or pre_diff.trees_identical or pre_missing:
                reported_missing = (
                    pre_missing
                    if pre_missing
                    else (["tests_passed"] if pre_diff.trees_identical else list(pre_locks.missing_locks))
                )
                return _error_response(
                    "gate_locked",
                    f"Server-side production gate rejected deployment: "
                    f"missing_locks={reported_missing} for diff {pre_locks.diff_sha256[:8]}.",
                    deployed=False,
                    missing_locks=reported_missing,
                    locks=pre_locks.model_dump(mode="json"),
                )

        session = manager.sessions.get(thread_id)
        if session is None:
            resolved_model, resolved_topology = manager.resolve_thread_meta(
                thread_id, model_hint=req.model, topology_hint=req.topology
            )
            session = _ThreadSession(
                thread_id=thread_id,
                model=resolved_model,
                topology=resolved_topology,
                user_id=req.user_id,
            )

        deploy_events: list[RunEvent] = []
        wf = manager.get_workflow_for_session(session)
        try:
            outcome = await _invoke_maybe_async(
                wf.submit_deploy_decision,
                thread_id,
                decision,
                user_id=req.user_id,
                on_event=deploy_events.append,
            )
        except _thread_state_error() as exc:
            # Only "this thread has no deploy decision to take" is a 409; any other failure keeps its own type.
            return _error_response(
                "conflict",
                f"Thread '{thread_id}' is not waiting for a deploy decision: {exc}",
                deployed=False,
                missing_locks=list(pre_locks.missing_locks),
                locks=pre_locks.model_dump(mode="json"),
            )

        manager.sessions[thread_id] = session
        manager.record_outcome(session, outcome)
        manager.latest_thread_id = thread_id
        deployed = bool(outcome.final_state.deployed)
        pending = session.pending_release
        # the steps of this decision (human sign-off, system deploy_to_production), for the Studio's step feed
        step_traces = [ev.step.model_dump(mode="json") for ev in deploy_events if ev.type == "step" and ev.step]
        interrupt_dump = _dump_or_none(outcome.interrupt)

        if decision.approved and not deployed:
            post_locks = manager.workspace.get_gate_status(expected_sha)
            refusal: dict[str, Any] = {
                "deployed": False,
                "missing_locks": list(post_locks.missing_locks),
                "locks": post_locks.model_dump(mode="json"),
                "trace": step_traces,
                "stop_reason": outcome.stop_reason,
                "interrupt": interrupt_dump,
                "pending_release": _dump_or_none(pending),
            }
            if pending is not None:
                # The gate let the deploy through but the release was not published. The failed push revoked the
                # sign-off: the human may approve the same diff again (while the harness still offers a retry)
                # or reject it. This is a release failure, reported with its own code, never as "gate_locked".
                code: ToolErrorCode = pending.error.code if pending.error is not None else "upstream_error"
                return _error_response(
                    code,
                    outcome.final_answer or f"The release of diff {expected_sha[:8]} was not published ({code}).",
                    status=409,
                    retryable=outcome.interrupt is not None and code in RELEASE_RETRY_CODES,
                    **refusal,
                )
            return _error_response(
                "gate_locked",
                outcome.final_answer
                or f"Server-side production gate rejected deployment for diff {post_locks.diff_sha256[:8]}.",
                **refusal,
            )
        if deployed:
            session.deployed_diff_sha256 = expected_sha

        post_diff = manager.workspace.get_diff_report()
        ship_log = manager.workspace.get_latest_ship_log(thread_id=thread_id)
        if ship_log is None:
            ship_log = manager.workspace.get_latest_ship_log(thread_id=None)
        # The locks recorded for the decided diff. Nothing is made up when a record is missing: that is an error.
        final_locks = (
            manager.workspace.gate_status_of(expected_sha)
            if deployed
            else manager.workspace.get_gate_status(expected_sha)
        )
        return JSONResponse(
            status_code=200,
            content={
                "deployed": deployed,
                "stop_reason": outcome.stop_reason or ("completed" if deployed else "human_rejected"),
                "diff": post_diff.model_dump(mode="json"),
                "deployed_at": ship_log.deployed_at.isoformat()
                if deployed and ship_log and ship_log.deployed_at
                else None,
                "ship_log": ship_log.model_dump(mode="json") if ship_log else None,
                "locks": final_locks.model_dump(mode="json"),
                "trace": step_traces,
            },
        )

    @app.get("/api/memory/{user_id}")
    async def get_user_memory(user_id: str) -> dict[str, Any]:
        records = _normalize_skill_profile(manager.workspace.get_skill_profile(user_id=user_id), user_id)
        return {
            "user_id": user_id,
            "records": [rec.model_dump(mode="json") for rec in records],
        }

    @app.delete("/api/memory/{user_id}")
    async def delete_user_memory(user_id: str) -> dict[str, Any]:
        with manager.exclusive("memory"):
            await _invoke_maybe_async(manager.workspace.forget_user_skills, user_id=user_id)
        records = _normalize_skill_profile(manager.workspace.get_skill_profile(user_id=user_id), user_id)
        return {
            "user_id": user_id,
            "forgotten": True,
            "records": [rec.model_dump(mode="json") for rec in records],
        }

    @app.get("/api/eval/results")
    async def get_eval_results() -> dict[str, Any]:
        return manager.load_eval_results()

    return app


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``no-quiz-ui``."""
    parser = argparse.ArgumentParser(
        prog="no-quiz-ui",
        description="Launch the No-Quiz-No-Prod Conversational Web Studio & Evaluation Explorer.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="Extra host name the Studio answers to, besides loopback and --host (repeatable). "
        "Requests with any other Host header are refused.",
    )
    args = parser.parse_args(argv)
    allowed_hosts = allowed_hosts_from(args.host, args.allowed_host)

    if args.reload:
        # The reloader re-imports the factory in a child process: the allow-list travels through the environment.
        os.environ[ALLOWED_HOSTS_ENV] = ",".join(allowed_hosts)
        uvicorn.run(
            "no_quiz_no_prod.ui.server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(allowed_hosts=allowed_hosts), host=args.host, port=args.port)
    return 0


__all__ = [
    "RunActiveError",
    "StudioManager",
    "create_app",
    "main",
    "same_origin_client_kwargs",
]


if __name__ == "__main__":
    raise SystemExit(main())
