"""FastAPI Conversational Studio & Oral Defence Explorer backend (`no-quiz-ui`).

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
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from no_quiz_no_prod import __version__
from no_quiz_no_prod.contracts import (
    ALL_EVAL_OUTCOMES,
    ALL_SCENARIO_KINDS,
    ALL_SKILL_TAGS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_USER_ID,
    ENV_FILE,
    NOTION_IDS_FILE,
    PROJECT_ROOT,
    DeployAction,
    DeployDecision,
    DeployGateStatus,
    DiffReport,
    EnvVar,
    EvalRunResult,
    FaultKind,
    GateLedger,
    HumanInterrupt,
    InjectionSpec,
    NotionRule,
    PublicQuizView,
    PytestReport,
    QuizAnswer,
    QuizGradeResult,
    QuizOptionKey,
    RunBudget,
    RunEvent,
    RunFinalState,
    RuntimeSettings,
    ShipLogEntry,
    SkillProfileRecord,
    StopReason,
    ToolCallTrace,
    Topology,
    compute_tree_sha256,
    pass_at_k,
    pass_hat_k,
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
    def get_gate_status(self, expected_diff_sha256: str | None = None) -> DeployGateStatus: ...
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


# ---- lightweight standalone workspace fallback (only when backend module is absent) --------------------------------
class _StandaloneWorkspaceFallback:
    """Minimal contract-faithful workspace when running in an isolated worktree without ``backend`` installed."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._skills: dict[str, dict[str, SkillProfileRecord]] = {}
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.settings.area_dir("prod").mkdir(parents=True, exist_ok=True)
        self.settings.area_dir("staging").mkdir(parents=True, exist_ok=True)

    def reset(
        self,
        seed: str = DEFAULT_SEED,
        *,
        fault: FaultKind | None = None,
        injection: InjectionSpec | None = None,
        preserve_skill_memory: bool = False,
    ) -> RunFinalState:
        _ = (seed, fault, injection)
        self._ensure_dirs()
        if self.settings.gate_ledger_path.is_file():
            self.settings.gate_ledger_path.unlink()
        if not preserve_skill_memory:
            self._skills.clear()
        diff = self.get_diff_report()
        return RunFinalState(
            prod_tree_sha256=diff.prod_tree_sha256,
            staging_tree_sha256=diff.staging_tree_sha256,
            prod_changed=False,
        )

    def get_diff_report(self) -> DiffReport:
        self._ensure_dirs()
        prod_sha = compute_tree_sha256(self.settings.area_dir("prod"))
        staging_sha = compute_tree_sha256(self.settings.area_dir("staging"))
        return DiffReport(prod_tree_sha256=prod_sha, staging_tree_sha256=staging_sha, files=[])

    def get_latest_pytest(self) -> PytestReport | None:
        return None

    def get_latest_ship_log(self, thread_id: str | None = None) -> ShipLogEntry | None:
        _ = thread_id
        return None

    def get_latest_public_quiz(
        self, thread_id: str | None = None, diff_sha256: str | None = None
    ) -> PublicQuizView | None:
        _ = (thread_id, diff_sha256)
        return None

    def grade_quiz_answer(
        self, question_id: str, selected_option: QuizOptionKey, *, user_id: str = DEFAULT_USER_ID
    ) -> QuizGradeResult:
        _ = (question_id, selected_option, user_id)
        raise KeyError("No active quiz in standalone fallback workspace.")

    def apply_deploy_decision(self, decision: DeployDecision, *, thread_id: str | None = None) -> DeployGateStatus:
        _ = thread_id
        diff = self.get_diff_report()
        ledger = GateLedger.load(self.settings.gate_ledger_path)
        target_sha = decision.expected_diff_sha256 or diff.diff_sha256
        entry = "human_approved" if decision.approved else "blocked"
        updated = ledger.record(entry, target_sha)
        updated.save(self.settings.gate_ledger_path)
        return updated.status(diff.diff_sha256, decision.expected_diff_sha256)

    def get_gate_status(self, expected_diff_sha256: str | None = None) -> DeployGateStatus:
        diff = self.get_diff_report()
        ledger = GateLedger.load(self.settings.gate_ledger_path)
        return ledger.status(diff.diff_sha256, expected_diff_sha256)

    def get_skill_profile(self, user_id: str = DEFAULT_USER_ID) -> list[SkillProfileRecord]:
        bucket = self._skills.get(user_id, {})
        return [bucket.get(tag, SkillProfileRecord(skill_tag=tag, user_id=user_id)) for tag in ALL_SKILL_TAGS]

    def forget_user_skills(self, user_id: str = DEFAULT_USER_ID) -> int:
        removed = len(self._skills.pop(user_id, {}))
        return removed


def _load_default_workspace(settings: RuntimeSettings) -> WorkspaceBackend:
    try:
        from no_quiz_no_prod.backend.workspace import SandboxWorkspace  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name in {"no_quiz_no_prod.backend", "no_quiz_no_prod.backend.workspace"}:
            return _StandaloneWorkspaceFallback(settings)
        raise
    return SandboxWorkspace(settings=settings)


def _default_workflow_factory(settings: RuntimeSettings, topology: Topology, model: str) -> WorkflowBackend:
    try:
        from no_quiz_no_prod.graph.workflow import NoQuizWorkflow  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name in {"no_quiz_no_prod.graph", "no_quiz_no_prod.graph.workflow"}:
            raise RuntimeError(
                "LangGraph workflow package (`no_quiz_no_prod.graph.workflow`) is not installed in this worktree."
            ) from exc
        raise
    return NoQuizWorkflow(settings=settings, topology=topology, model=model)


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
    expected_diff_sha256: str | None = None
    reason: str = ""
    user_id: str = DEFAULT_USER_ID
    model: str | None = None
    topology: Topology | None = None


ApiKeyProvider = Literal["openrouter", "anthropic", "notion"]

_API_KEY_ENV_MAP: Final[dict[ApiKeyProvider, tuple[EnvVar, str]]] = {
    "openrouter": (EnvVar.OPENROUTER_API_KEY, "openrouter_api_key"),
    "anthropic": (EnvVar.ANTHROPIC_API_KEY, "anthropic_api_key"),
    "notion": (EnvVar.NOTION_API_KEY, "notion_api_key"),
}


class ApiKeyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ApiKeyProvider
    api_key: str = Field(min_length=1, repr=False)

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
        self.eval_dir = eval_dir or (PROJECT_ROOT / "eval")
        self.notion_ids_file = notion_ids_file or NOTION_IDS_FILE
        self.env_file = env_file or ENV_FILE
        self.sessions: dict[str, _ThreadSession] = {}
        self.latest_thread_id: str | None = None

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
        if self.env_file == ENV_FILE:
            os.environ[env_var.value] = api_key

        notion_ids = _read_notion_metadata(self.notion_ids_file)
        updates: dict[str, Any] = {settings_attr: SecretStr(api_key)}
        if not self.settings.notion_ship_log_db_id and notion_ids.get("ship_log_db_id"):
            updates["notion_ship_log_db_id"] = notion_ids["ship_log_db_id"]
        if not self.settings.notion_rules_db_id and notion_ids.get("rules_db_id"):
            updates["notion_rules_db_id"] = notion_ids["rules_db_id"]
        if not self.settings.notion_skills_db_id and notion_ids.get("skill_profile_db_id"):
            updates["notion_skills_db_id"] = notion_ids["skill_profile_db_id"]

        self.settings = self.settings.model_copy(update=updates)
        if self._workspace is not None:
            if hasattr(self._workspace, "settings"):
                self._workspace.settings = self.settings  # type: ignore[attr-defined]
            if hasattr(self._workspace, "notion"):
                old_notion = getattr(self._workspace, "notion", None)
                close_fn = getattr(old_notion, "close", None)
                if callable(close_fn):
                    close_fn()
                from no_quiz_no_prod.backend.notion_sync import NotionDatabaseSync

                self._workspace.notion = NotionDatabaseSync(self.settings)  # type: ignore[attr-defined]
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
            topology_hint if topology_hint is not None else (saved_topo if saved_topo in ("dual", "single") else "dual")
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
        if (
            session is not None
            and session.deployed_diff_sha256
            and diff.trees_identical
            and self.settings.gate_ledger_path.is_file()
        ):
            ledger = GateLedger.load(self.settings.gate_ledger_path)
            locks = ledger.status(session.deployed_diff_sha256, session.deployed_diff_sha256)
        else:
            locks = ws.get_gate_status(expected_diff_sha256=expected_diff_sha256)
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
            },
            "skill_profile": [rec.model_dump(mode="json") for rec in skills],
        }

    def load_eval_results(self) -> dict[str, Any]:
        candidate_files: list[Path] = []
        if self.eval_dir.is_dir():
            primary = self.eval_dir / "results.jsonl"
            if primary.is_file():
                candidate_files.append(primary)
            for extra in sorted(self.eval_dir.glob("results_*.jsonl")):
                if extra.is_file() and extra != primary:
                    candidate_files.append(extra)

        seen_keys: set[tuple[str, Topology, str, int]] = set()
        runs: list[EvalRunResult] = []
        used_sources: list[str] = []
        for path in candidate_files:
            added = 0
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parsed_run = EvalRunResult.model_validate_json(stripped)
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
                    if r.stop_reason in ("quiz_failed", "human_rejected", "refused")
                    or r.final_state.ship_log_status == "Blocked"
                ),
                None,
            ),
        )

        return {
            "available": True,
            "source_files": used_sources,
            "summary": {
                "total_runs": total_runs,
                "passed_runs": passed_runs,
                "pass_rate": round(passed_runs / total_runs, 4),
                "k_repeats": k_repeats,
                "pass_at_k": round(pass_at_k(primary_cohort_runs), 4),
                "pass_hat_k": round(pass_hat_k(primary_cohort_runs), 4),
                "by_outcome": by_outcome,
                "forbidden_calls_count": sum(1 for r in runs if r.forbidden_called),
                "hidden_failures_count": sum(1 for r in runs if r.hidden_failure),
                "avg_steps": round(sum(r.steps for r in runs) / total_runs, 2),
                "avg_latency_s": round(sum(r.latency_s for r in runs) / total_runs, 2),
                "avg_cost_usd": round(sum(r.cost_usd for r in runs) / total_runs, 6),
                "by_kind": by_kind,
                "cohorts": cohorts_payload,
                "showcase_passed_key": showcase_passed,
                "showcase_failed_or_blocked_key": showcase_failed,
            },
            "runs": [r.model_dump(mode="json") for r in runs],
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
) -> FastAPI:
    """Create the FastAPI Studio server with injectable workspace/workflow backends for testing."""
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
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_model=None)
    async def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

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
            return JSONResponse(
                status_code=415,
                content={"ok": False, "detail": "Content-Type must be application/json."},
            )
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        if origin and host and urlparse(origin).netloc.lower() != host.lower():
            return JSONResponse(
                status_code=403,
                content={"ok": False, "detail": "Cross-origin API key updates are forbidden."},
            )
        try:
            raw_body = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={"ok": False, "detail": "Invalid JSON request body."},
            )
        if not isinstance(raw_body, dict):
            return JSONResponse(
                status_code=422,
                content={"ok": False, "detail": "Expected JSON object payload."},
            )
        try:
            req = ApiKeyUpdateRequest.model_validate(raw_body)
        except ValidationError as exc:
            first_msg = exc.errors()[0].get("msg", "Invalid API key update request.") if exc.errors() else "Invalid."
            return JSONResponse(
                status_code=422,
                content={"ok": False, "detail": str(first_msg)},
            )

        providers = await _invoke_maybe_async(manager.update_api_key, req.provider, req.api_key)
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "provider": req.provider,
                "providers": providers,
                "notion_enabled": manager.settings.notion_enabled,
                "models": _build_models_catalogue(manager.settings),
            },
        )

    @app.post("/api/reset")
    async def post_reset(req: ResetRequest) -> dict[str, Any]:
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

    @app.post("/api/runs/stream")
    async def post_runs_stream(req: RunStreamRequest) -> StreamingResponse:
        catalogue = _build_models_catalogue(manager.settings)
        chosen_model = req.model or str(catalogue["default_model"])
        thread_id = req.thread_id or f"thr_{uuid.uuid4().hex[:8]}"

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

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()

            def on_event(ev: RunEvent) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            async def run_worker() -> tuple[WorkflowOutcomeLike | None, Exception | None]:
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
                    return outcome, None
                except Exception as exc:  # noqa: BLE001 — surfaced as structured SSE run_error frame
                    return None, exc
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            worker_task = asyncio.create_task(run_worker())
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
            if error is not None:
                session.stop_reason = "api_error"
                session.interrupt = None
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

            assert outcome is not None
            session.interrupt = outcome.interrupt
            session.stop_reason = outcome.stop_reason
            session.budget = outcome.budget
            session.final_answer = outcome.final_answer
            if outcome.interrupt and outcome.interrupt.quiz:
                session.expected_diff_sha256 = outcome.interrupt.quiz.diff_sha256
            elif outcome.interrupt and outcome.interrupt.gate:
                session.expected_diff_sha256 = outcome.interrupt.gate.diff_sha256

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
            elif outcome.stop_reason == "api_error":
                term_event = RunEvent(
                    type="run_error",
                    thread_id=thread_id,
                    seq=seq,
                    budget=outcome.budget,
                    stop_reason="api_error",
                    text=outcome.final_answer or "Upstream LLM provider error.",
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
        answer = QuizAnswer(selected_option=req.selected_option)
        session = manager.sessions.get(thread_id)
        wants_sse = stream or "text/event-stream" in request.headers.get("accept", "")

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

        try:
            wf = manager.get_workflow_for_session(session)
            grade, outcome = await _invoke_maybe_async(
                wf.submit_quiz,
                thread_id,
                answer,
                user_id=req.user_id,
                on_event=collect_event,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "code": "gate_locked",
                    "message": f"No resumable workflow quiz interrupt for thread '{thread_id}': {exc}",
                },
            )

        manager.sessions[thread_id] = session
        session.interrupt = outcome.interrupt
        session.stop_reason = outcome.stop_reason
        session.budget = outcome.budget
        session.final_answer = outcome.final_answer
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
        pre_diff = manager.workspace.get_diff_report()
        expected_sha = req.expected_diff_sha256 or pre_diff.diff_sha256

        # Validate expected_diff_sha256 shape before constructing strictSha256Hex DeployDecision
        try:
            decision = DeployDecision(
                action=req.action,
                expected_diff_sha256=expected_sha,
                reason=req.reason,
            )
        except ValidationError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "deployed": False,
                    "code": "gate_locked",
                    "missing_locks": ["sha256_matches"],
                    "message": f"Invalid expected_diff_sha256 digest: {exc.errors()[0]['msg']}",
                },
            )

        pre_locks = manager.workspace.get_gate_status(expected_diff_sha256=decision.expected_diff_sha256)

        if decision.approved:
            pre_missing = [lock for lock in pre_locks.missing_locks if lock != "human_approved"]
            if pre_locks.blocked or pre_diff.trees_identical or pre_missing:
                reported_missing = (
                    pre_missing
                    if pre_missing
                    else (["tests_passed"] if pre_diff.trees_identical else list(pre_locks.missing_locks))
                )
                return JSONResponse(
                    status_code=409,
                    content={
                        "deployed": False,
                        "code": "gate_locked",
                        "missing_locks": reported_missing,
                        "locks": pre_locks.model_dump(mode="json"),
                        "message": (
                            f"Server-side production gate rejected deployment: "
                            f"missing_locks={reported_missing} for diff {pre_locks.diff_sha256[:8]}."
                        ),
                    },
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

        try:
            wf = manager.get_workflow_for_session(session)
            outcome = await _invoke_maybe_async(
                wf.submit_deploy_decision,
                thread_id,
                decision,
                user_id=req.user_id,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "deployed": False,
                    "code": "gate_locked",
                    "missing_locks": list(pre_locks.missing_locks),
                    "locks": pre_locks.model_dump(mode="json"),
                    "message": f"No resumable workflow deploy gate for thread '{thread_id}': {exc}",
                },
            )

        manager.sessions[thread_id] = session
        session.interrupt = outcome.interrupt
        session.stop_reason = outcome.stop_reason
        session.budget = outcome.budget
        session.final_answer = outcome.final_answer
        stop_reason: StopReason = outcome.stop_reason or ("completed" if decision.approved else "human_rejected")

        ledger_proved = manager.settings.gate_ledger_path.is_file() and (
            expected_sha in GateLedger.load(manager.settings.gate_ledger_path).deployed
        )
        deployed_ok = bool(outcome.final_state.deployed or ledger_proved)

        if decision.approved and not deployed_ok:
            post_locks = manager.workspace.get_gate_status(expected_diff_sha256=decision.expected_diff_sha256)
            return JSONResponse(
                status_code=409,
                content={
                    "deployed": False,
                    "code": "gate_locked",
                    "missing_locks": list(post_locks.missing_locks),
                    "locks": post_locks.model_dump(mode="json"),
                    "message": outcome.final_answer
                    or f"Server-side production gate rejected deployment for diff {post_locks.diff_sha256[:8]}.",
                },
            )
        if deployed_ok:
            session.deployed_diff_sha256 = expected_sha

        manager.latest_thread_id = thread_id
        post_diff = manager.workspace.get_diff_report()
        ship_log = manager.workspace.get_latest_ship_log(thread_id=thread_id)
        if ship_log is None:
            ship_log = manager.workspace.get_latest_ship_log(thread_id=None)

        if deployed_ok:
            if manager.settings.gate_ledger_path.is_file():
                final_locks = GateLedger.load(manager.settings.gate_ledger_path).status(expected_sha, expected_sha)
            else:
                final_locks = DeployGateStatus(
                    diff_sha256=expected_sha,
                    tests_passed=True,
                    sha256_matches=True,
                    quiz_passed=True,
                    human_approved=True,
                    blocked=False,
                )
            deployed_at_iso = (
                ship_log.deployed_at.isoformat() if ship_log and ship_log.deployed_at else datetime.now(UTC).isoformat()
            )
        else:
            final_locks = manager.workspace.get_gate_status(expected_diff_sha256=expected_sha)
            deployed_at_iso = None

        return JSONResponse(
            status_code=200,
            content={
                "deployed": deployed_ok,
                "stop_reason": stop_reason,
                "diff": post_diff.model_dump(mode="json"),
                "deployed_at": deployed_at_iso,
                "ship_log": ship_log.model_dump(mode="json") if ship_log else None,
                "locks": final_locks.model_dump(mode="json"),
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
        description="Launch the No-Quiz-No-Prod Conversational Web Studio & Oral Defence Explorer.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    args = parser.parse_args(argv)

    if args.reload:
        uvicorn.run(
            "no_quiz_no_prod.ui.server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


__all__ = [
    "StudioManager",
    "create_app",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
