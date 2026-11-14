"""The No-Quiz-No-Prod state machine (LangGraph) and its runner API (#14 §5).

    init -> agent <-> tools                       (ReAct loop of the active agent; budget router -> give_up)
            agent -> handoff                      (deterministic checks on a final message)
    dual:   coder -> handoff -> examiner (fresh context) -> handoff -> quiz_gate
    single: single -> handoff -> quiz_gate        (one shared conversation)
    quiz_gate -> deploy_gate | examiner/single (remediation v2) | finish (quiz_failed)
    deploy_gate -> finish (completed | human_rejected | refused)

Harness guarantees, independent of the model:
- Step budget: an LLM-initiated tool call is a step; once ``steps >= max_steps`` the router goes to ``give_up``
  (``stop_reason="max_steps"``) and no further call runs. Strict ReAct execution: one call per model turn
  (extra parallel calls are answered "not executed", never run).
- Unusable turns: a model turn with no tool call and no text (a thinking model that spent its whole output budget on
  hidden reasoning, ``finish_reason="length"``) is asked again with a harness note, at most
  ``MAX_UNUSABLE_TURN_RETRIES`` times, then stops as ``api_error``. It is never read as a final "done" message.
- Writes carry ``idem_key(thread_id, tool, args)`` (staging writes and test runs also bind the
  staging tree they act on, so a re-run after a change is a new logical call while a retry replays). A retryable
  ``ToolErrorPayload`` (``timeout`` with ``outcome_unknown``, ``rate_limited``) is retried at most
  ``MAX_TOOL_RETRIES`` times with backoff and the SAME key; ``attempts`` and ``replayed`` land in the trace.
- The quiz is graded server-side by the workspace; a wrong first answer updates Layer 4 memory and sends the
  Examiner back for doc v2 + a new question; the deploy gate asks the human, records the decision on the harness
  side (``apply_deploy_decision``) and only then calls ``code-mcp.deploy_to_production``. No agent can deploy.
- Every step, artifact and pause is streamed as a ``RunEvent`` to ``on_event`` (the SSE payloads of the Studio).
"""

import json
import logging
import operator
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, Final, Literal, NotRequired, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import Field, ValidationError

from no_quiz_no_prod.agents import (
    AGENTS,
    TOOLS,
    AgentSpec,
    ToolDef,
    bind_quiz_args,
    bind_ship_log_args,
    build_coder_messages,
    build_examiner_brief,
    build_examiner_messages,
    build_single_messages,
    build_single_remediation_message,
    developer_request,
    harness_message,
    missing_deliverables,
    parse_agent_status,
    render_applied_rules,
    render_tool_result,
    strip_status_line,
)
from no_quiz_no_prod.agents.tools import IDEMPOTENCY_ARG
from no_quiz_no_prod.contracts import (
    ALL_QUIZ_OPTION_KEYS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_USER_ID,
    MAX_QUIZ_ATTEMPTS,
    AgentRole,
    ContractModel,
    DeployDecision,
    DeployGateStatus,
    DiffReport,
    EvalRunResult,
    FaultKind,
    HumanInterrupt,
    HumanPersona,
    InjectionSpec,
    PublicQuizView,
    PytestReport,
    QuizAnswer,
    QuizGradeResult,
    QuizOptionKey,
    QuizQuestion,
    ReleaseCommit,
    RunBudget,
    RunEvent,
    RunEventType,
    RunFinalState,
    RuntimeSettings,
    ScenarioSpec,
    ShipLogEntry,
    SkillProfileRecord,
    StopReason,
    ToolCallTrace,
    ToolErrorPayload,
    Topology,
    idem_key,
)
from no_quiz_no_prod.graph.mcp_bridge import MCPToolBridge, McpTransport, ToolDispatcher, ToolOutcome, create_bridge
from no_quiz_no_prod.llm import ChatTurnResponse, LLMClient, LLMError, LLMUsage, NormalizedToolCall, tool_result_message
from no_quiz_no_prod.memory.skill_store import SkillMemory, build_forget_graph, is_forget_command

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME: Final = "checkpoints.sqlite"
MAX_TOOL_RETRIES: Final = 2  # retries after the first attempt, for retryable tool errors only
MAX_CALLS_PER_TURN: Final = 1  # ReAct: observe each result before the next action
RETRY_BACKOFF_S: Final = (0.25, 0.75)
MAX_HANDOFF_NUDGES: Final = 2  # harness reminders before a phase that keeps failing its checks is stopped
MAX_UNUSABLE_TURN_RETRIES: Final = 2  # re-asks after a model turn with no tool call and no text (truncated or empty)
EVAL_STEP_HEADROOM: Final = 4  # eval hard cap = scenario budget + headroom, so over_budget stays observable
MAX_HITL_ROUNDS: Final = MAX_QUIZ_ATTEMPTS + 1  # every quiz attempt, then one deploy decision
TREE_BOUND_TOOLS: Final = frozenset({"write_staging_file", "run_staging_tests"})
TREE_KEY_FIELD: Final = "staging_tree_sha256"  # mixed into the key only, never sent to the server
HARVESTED_KEYS: Final = ("quiz", "ship_log", "rules_read", "wrote_staging")
RELEASE_NOTE_CHARS: Final = 200
INJECTION_PROBE_CHARS: Final = 48
FORGET_THREAD_PREFIX: Final = "forget:"
ARTIFACTS_BY_TOOL: Final[Mapping[str, tuple[str, ...]]] = {
    "write_staging_file": ("diff", "locks"),
    "get_staging_diff": ("diff", "locks"),
    "run_staging_tests": ("diff", "pytest", "locks"),
    "upsert_ship_log_doc": ("ship_log",),
    "create_comprehension_quiz": ("quiz",),
}
NOTHING_STAGED: Final = "No change was staged, so there is nothing to ship."
DIFF_BLOCKED: Final = "This exact diff is blocked (quiz exhausted or rejected): it can never be deployed."
NUDGE_TESTS: Final = (
    "The staging tests have not passed on the current diff (Lock 1 is missing). Run `run_staging_tests` on the full "
    "suite (no pytest_k), fix any failure, and only then finish."
)
QUIZ_PROMPT: Final = (
    "Comprehension quiz, attempt {attempt} of {max_attempts}: answer it to unlock the production approval."
)
QUIZ_BLOCKED: Final = (
    "Quiz failed on the last attempt: diff {sha} is blocked for good and production was not touched. The correct "
    "answer was {correct}."
)
DEPLOY_READY: Final = "Locks 1-3 hold for diff {sha}: approve to deploy it to production, or reject it."
DEPLOY_NOT_READY: Final = "Diff {sha} is missing {missing}: an approval will be refused by the server-side gate."
DEPLOYED: Final = "Deployed to production: diff {sha} passed all four SHA-256 locks."
RELEASED: Final = " Release commit {commit} is on {branch} ({remote})."
REJECTED: Final = "Deployment rejected by the human reviewer: diff {sha} is blocked and production was not touched."
GATE_REFUSED: Final = "The server-side gate refused the deployment: {message}"
BUDGET_EXHAUSTED: Final = "Stopped: the step budget is exhausted ({steps}/{max_steps} tool calls)."
SKIPPED_CALL: Final = "Not executed: the step budget of this run is exhausted."
ONE_CALL_PER_TURN: Final = "Not executed: the harness runs ONE tool call per turn. Call it again if you still need it."
UNUSABLE_TURN: Final = (
    "Your last reply {problem} before any tool call or text, so nothing happened. Keep your reasoning short, then "
    "call ONE tool now, or write your final message starting with its STATUS line."
)
UNUSABLE_TURN_STOP: Final = (
    "Stopped: the model returned no usable turn {count} times in a row (its last reply {problem})."
)


class WorkspaceHarness(Protocol):
    """The ``SandboxWorkspace`` methods the harness calls in-process (#14 §4)."""

    def reset(
        self,
        seed: str = DEFAULT_SEED,
        *,
        fault: FaultKind | None = None,
        injection: InjectionSpec | None = None,
        preserve_skill_memory: bool = False,
    ) -> RunFinalState: ...

    def snapshot_final_state(self, thread_id: str | None = None) -> RunFinalState: ...

    def get_diff_report(self) -> DiffReport: ...

    def get_latest_pytest(self) -> PytestReport | None: ...

    def get_latest_ship_log(self, thread_id: str | None = None) -> ShipLogEntry | None: ...

    def get_latest_private_quiz(
        self, thread_id: str | None = None, diff_sha256: str | None = None
    ) -> QuizQuestion | None: ...

    def grade_quiz_answer(
        self, question_id: str, selected_option: QuizOptionKey, *, user_id: str = DEFAULT_USER_ID
    ) -> QuizGradeResult: ...

    def apply_deploy_decision(self, decision: DeployDecision, *, thread_id: str | None = None) -> DeployGateStatus: ...

    def get_gate_status(self, expected_diff_sha256: str | None = None) -> DeployGateStatus: ...

    def get_skill_profile(self, user_id: str = DEFAULT_USER_ID) -> list[SkillProfileRecord]: ...

    def upsert_skill_record(self, record: SkillProfileRecord) -> SkillProfileRecord: ...

    def forget_user_skills(self, user_id: str = DEFAULT_USER_ID) -> int: ...


class WorkflowOutcome(ContractModel):
    """Where a thread stands after one call: paused on a human gate (``interrupt``) or terminal (``stop_reason``)."""

    thread_id: str
    stop_reason: StopReason | None = None
    interrupt: HumanInterrupt | None = None
    final_answer: str = ""
    trace: list[ToolCallTrace] = Field(default_factory=list)
    budget: RunBudget
    final_state: RunFinalState


class WorkflowInput(TypedDict):
    task: str
    user_id: NotRequired[str]
    max_steps: NotRequired[int]
    injection_text: NotRequired[str | None]


class WorkflowState(TypedDict, total=False):
    task: str
    user_id: str
    max_steps: int  # per-turn allowance (input); the cumulative cap is budget.max_steps
    injection_text: str | None  # adversarial eval runs: flags tool results that carry the planted text
    topology: Topology  # per thread: a workflow rebuilt after a restart resumes it with the right topology
    model: str  # per turn: resuming after a restart keeps the model the turn started with
    phase: AgentRole  # the active LLM agent: coder | examiner | single
    messages: dict[str, list[dict[str, Any]]]  # one OpenAI-format history per agent role
    pending_calls: list[dict[str, Any]]  # tool calls of the last model turn, not executed yet
    last_text: str
    trace: Annotated[list[dict[str, Any]], operator.add]
    budget: dict[str, Any]
    attempt: int
    quiz: dict[str, Any] | None  # PublicQuizView of the current attempt
    ship_log: dict[str, Any] | None
    last_grade: dict[str, Any] | None
    prior_failed_grade: dict[str, Any] | None
    rules_read: dict[str, dict[str, str]]  # rule_code -> {title, body}, from read_notion_rule results
    wrote_staging: bool
    nudges: int
    budget_exhausted: bool
    stop_reason: StopReason | None
    final_answer: str


# ---- small pure helpers --------------------------------------------------------------------------------------------
def _config(thread_id: str) -> RunnableConfig:
    if not thread_id:
        raise ValueError("thread_id must be a non-empty string")
    return {"configurable": {"thread_id": thread_id}}


def _thread_id(config: RunnableConfig) -> str:
    return str(config["configurable"]["thread_id"])


def _budget(state: Mapping[str, Any]) -> RunBudget:
    raw = state.get("budget")
    return RunBudget.model_validate(raw) if raw else RunBudget()


def _with_usage(budget: RunBudget, usage: LLMUsage) -> RunBudget:
    return budget.model_copy(
        update={
            "tokens_in": budget.tokens_in + usage.tokens_in,
            "tokens_out": budget.tokens_out + usage.tokens_out,
            "cost_usd": budget.cost_usd + usage.cost_usd,
            "latency_s": budget.latency_s + usage.latency_s,
        }
    )


def _grade(raw: Mapping[str, Any] | None) -> QuizGradeResult | None:
    return QuizGradeResult.model_validate(raw) if raw else None


def _assistant_message(turn: ChatTurnResponse) -> dict[str, Any]:
    """OpenAI-format assistant turn (used when a client returns none)."""
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.args)}}
            for call in turn.tool_calls
        ]
    return message


def _unusable_turn(turn: ChatTurnResponse, max_tokens: int) -> str | None:
    """Why a model turn carries no decision at all (no tool call, no text), or None when it is usable.

    Thinking models can spend the whole output budget on hidden reasoning (``finish_reason="length"``): such a turn
    must be asked again, never read as a final message (an empty text parses as "done": a silent no-op run).
    """
    if turn.tool_calls or turn.content.strip():
        return None
    return f"hit the {max_tokens}-token output limit" if turn.finish_reason == "length" else "was empty"


def _refusal(code: Literal["invalid_argument", "forbidden"], message: str) -> ToolOutcome:
    return ToolOutcome(error=ToolErrorPayload(code=code, message=message))


def _answer_skipped(
    history: list[dict[str, Any]], calls: list[dict[str, Any]], reason: str = SKIPPED_CALL
) -> list[dict[str, Any]]:
    """Answer calls that will not run (not executed, not traced): every tool call keeps its tool message."""
    skipped = _refusal("forbidden", reason).error
    return [*history, *(tool_result_message(c["id"], render_tool_result(c["name"], error=skipped)) for c in calls)]


def _public_quiz(outcome: ToolOutcome) -> ToolOutcome:
    """Defense in depth: whatever the server returned, only the public view of a quiz goes further."""
    if not outcome.ok or not isinstance(outcome.result, Mapping):
        return outcome
    public = {key: value for key, value in outcome.result.items() if key in PublicQuizView.model_fields}
    try:
        return ToolOutcome(result=PublicQuizView.model_validate(public).model_dump(mode="json"))
    except ValidationError as exc:
        return ToolOutcome(error=ToolErrorPayload(code="internal_error", message=f"invalid quiz view: {exc}"[:300]))


def _carries(value: Any, needle: str | None) -> bool:
    """True when a tool result contains the first line of the planted injection text (whitespace-insensitive)."""
    first_line = (needle or "").strip().splitlines()[0] if (needle or "").strip() else ""
    probe = " ".join(first_line.split())[:INJECTION_PROBE_CHARS].lower()
    if not probe:
        return False
    if isinstance(value, str):
        return probe in " ".join(value.split()).lower()
    if isinstance(value, Mapping):
        return any(_carries(item, probe) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_carries(item, probe) for item in value)
    return False


def _parse_quiz_answer(value: Any) -> QuizAnswer:
    """Resume value of a quiz interrupt: ``{"selected_option": "B"}`` or a bare letter (Studio)."""
    option = value.get("selected_option") if isinstance(value, Mapping) else value
    if not isinstance(option, str):
        raise ValueError("a quiz answer needs selected_option")
    return QuizAnswer(selected_option=option.strip().upper())  # type: ignore[arg-type]


def _release_suffix(result: Any) -> str:
    """The ``RELEASED`` sentence of a deploy result carrying a ``ReleaseCommit``; "" for any other result."""
    release = result.get("release") if isinstance(result, Mapping) else None
    if release is None:
        return ""
    try:
        commit = ReleaseCommit.model_validate(release)
    except ValidationError:
        return ""
    return RELEASED.format(commit=commit.commit_sha[:12], branch=commit.branch, remote=commit.remote)


def _parse_deploy_decision(value: Any) -> DeployDecision:
    """Resume value of a deploy interrupt: a ``DeployDecision`` dict, "approve"/"reject", or a bool (Studio)."""
    if isinstance(value, bool):
        value = {"action": "approve" if value else "reject"}
    elif isinstance(value, str):
        value = {"action": value.strip().lower()}
    elif isinstance(value, Mapping):
        value = {key: value[key] for key in DeployDecision.model_fields if key in value}
    return DeployDecision.model_validate(value)


def _ask_human[T](request: HumanInterrupt, parse: Callable[[Any], T]) -> T:
    """``interrupt()`` until the resume value parses (an invalid answer asks again instead of crashing)."""
    payload = request.model_dump(mode="json")
    while True:
        try:
            return parse(interrupt(payload))
        except (ValidationError, ValueError, TypeError):
            continue


def _pending(snapshot: Any) -> HumanInterrupt | None:
    for item in getattr(snapshot, "interrupts", ()) or ():
        try:
            return HumanInterrupt.model_validate(item.value)
        except ValidationError:
            continue
    return None


def _finish(stop_reason: StopReason, answer: str, **update: Any) -> Command[Literal["finish"]]:
    return Command(goto="finish", update={**update, "stop_reason": stop_reason, "final_answer": answer})


class NoQuizWorkflow:
    """Compiles the graph once (``SqliteSaver`` at ``<workspace_dir>/checkpoints.sqlite``) and drives threads.

    Collaborators are lazy, so constructing a workflow never spawns a server or opens a file. Keyword-only extras
    make every dependency injectable: ``workspace`` (the ``SandboxWorkspace`` harness), ``tool_dispatcher`` (the
    ``"inprocess"`` transport without MCP servers), ``bridge``, ``memory``, ``checkpointer`` and ``sleep``.
    """

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        llm_client: LLMClient | None = None,
        topology: Topology = "dual",
        model: str | None = None,
        mcp_transport: McpTransport = "stdio",
        *,
        workspace: WorkspaceHarness | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        bridge: MCPToolBridge | None = None,
        memory: SkillMemory | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if topology not in ("dual", "single"):
            raise ValueError(f"unknown topology {topology!r}: expected 'dual' or 'single'")
        self.settings = settings or RuntimeSettings.from_env()
        self.topology: Topology = topology
        self.mcp_transport: McpTransport = mcp_transport
        self.model: str = model or (getattr(llm_client, "model", None) or self.settings.openrouter_model)
        self._llm = llm_client
        self._workspace = workspace
        self._dispatcher = tool_dispatcher
        self._bridge = bridge
        self._memory = memory
        self._checkpointer = checkpointer
        self._sleep = sleep
        self._conn: sqlite3.Connection | None = None
        self._app: CompiledStateGraph | None = None
        self._forget_app: CompiledStateGraph | None = None
        self._lock = threading.RLock()
        self._sinks: dict[str, Callable[[RunEvent], None]] = {}
        self._event_seq: dict[str, int] = {}

    # ---- lazy collaborators ----------------------------------------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        with self._lock:
            if self._llm is None:
                self._llm = LLMClient(model=self.model, settings=self.settings)
            return self._llm

    @property
    def workspace(self) -> WorkspaceHarness:
        with self._lock:
            if self._workspace is None:
                from no_quiz_no_prod.backend.workspace import SandboxWorkspace  # lazy: backend-mcp's domain

                self._workspace = SandboxWorkspace(self.settings)
            return self._workspace

    @property
    def bridge(self) -> MCPToolBridge:
        with self._lock:
            if self._bridge is None:
                self._bridge = create_bridge(self.mcp_transport, self.settings, dispatcher=self._dispatcher)
            return self._bridge

    @property
    def memory(self) -> SkillMemory:
        with self._lock:
            if self._memory is None:
                self._memory = SkillMemory(source=self.workspace, settings=self.settings)
            return self._memory

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        with self._lock:
            if self._checkpointer is None:
                self.settings.workspace_dir.mkdir(parents=True, exist_ok=True)
                path = self.settings.workspace_dir / CHECKPOINT_FILENAME
                self._conn = sqlite3.connect(path, check_same_thread=False)
                self._checkpointer = SqliteSaver(self._conn)
            return self._checkpointer

    @property
    def app(self) -> CompiledStateGraph:
        with self._lock:
            if self._app is None:
                self._app = self.build_graph(checkpointer=self.checkpointer, store=self.memory.store)
            return self._app

    @property
    def forget_app(self) -> CompiledStateGraph:
        with self._lock:
            if self._forget_app is None:
                self._forget_app = build_forget_graph(self.memory, checkpointer=self.checkpointer)
            return self._forget_app

    def close(self) -> None:
        """Stop the MCP servers and close the SQLite files (idempotent)."""
        with self._lock:
            if self._bridge is not None:
                self._bridge.close()
            if self._memory is not None:
                self._memory.close()
            if self._conn is not None:
                self._conn.close()
                self._conn, self._checkpointer, self._app, self._forget_app = None, None, None, None

    def __enter__(self) -> "NoQuizWorkflow":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- the graph -------------------------------------------------------------------------------------------------
    def build_graph(
        self, *, checkpointer: BaseCheckpointSaver | None = None, store: BaseStore | None = None
    ) -> CompiledStateGraph:
        """The state machine (no checkpointer for LangGraph Studio, which brings its own)."""
        graph = StateGraph(WorkflowState, input_schema=WorkflowInput)
        graph.add_node("init", self._node_init)
        graph.add_node("agent", self._node_agent)
        graph.add_node("tools", self._node_tools)
        graph.add_node("handoff", self._node_handoff, destinations=("agent", "quiz_gate", "finish"))
        graph.add_node("quiz_gate", self._node_quiz_gate, destinations=("deploy_gate", "agent", "finish"))
        graph.add_node("deploy_gate", self._node_deploy_gate)
        graph.add_node("give_up", self._node_give_up)
        graph.add_node("finish", self._node_finish)
        graph.add_edge(START, "init")
        graph.add_edge("init", "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_agent,
            {"tools": "tools", "give_up": "give_up", "handoff": "handoff", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "tools", lambda state: "give_up" if state.get("budget_exhausted") else "agent", ["give_up", "agent"]
        )
        graph.add_edge("deploy_gate", "finish")
        graph.add_edge("give_up", END)
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer, store=store, name="no_quiz_no_prod")

    def _node_init(self, state: WorkflowState) -> dict[str, Any]:
        user_id = state.get("user_id") or DEFAULT_USER_ID
        topology: Topology = state.get("topology") or self.topology
        lead: AgentRole = "single" if topology == "single" else "coder"
        task = state["task"]
        previous = (state.get("messages") or {}).get(lead)
        turn_steps = state.get("max_steps") or DEFAULT_MAX_STEPS
        if previous:  # a follow-up in the same thread (e.g. the answer to a clarifying question) keeps the history
            history = [*previous, developer_request(task, follow_up=True)]
            prior = _budget(state)
            budget = prior.model_copy(update={"max_steps": prior.steps + turn_steps})
        else:
            if lead == "single":
                history = build_single_messages(task, self.memory.format_memory_prompt_block(user_id))
            else:
                history = build_coder_messages(task)
            budget = RunBudget(max_steps=turn_steps)
        return {
            "user_id": user_id,
            "topology": topology,
            "model": self.model,
            "phase": lead,
            "messages": {lead: history},
            "pending_calls": [],
            "last_text": "",
            "budget": budget.model_dump(mode="json"),
            "attempt": 1,
            "quiz": None,
            "ship_log": None,
            "last_grade": None,
            "prior_failed_grade": None,
            "rules_read": dict(state.get("rules_read") or {}),
            "wrote_staging": False,
            "nudges": 0,
            "budget_exhausted": False,
            "stop_reason": None,
            "final_answer": "",
        }

    def _node_agent(self, state: WorkflowState) -> dict[str, Any]:
        """One model turn of the active agent; a turn with no tool call and no text is asked again (bounded)."""
        spec = AGENTS[state["phase"]]
        history = state["messages"][spec.role]
        model = state.get("model") or self.model
        prompt, budget, problem = history, _budget(state), ""
        for _attempt in range(MAX_UNUSABLE_TURN_RETRIES + 1):
            try:
                turn = self.llm.chat_with_tools(prompt, spec.tool_specs(), model=model, max_tokens=spec.max_tokens)
            except LLMError as exc:
                return {
                    "stop_reason": "api_error",
                    "final_answer": str(exc) or "the LLM provider failed",
                    "pending_calls": [],
                    "budget": budget.model_dump(mode="json"),
                }
            budget = _with_usage(budget, turn.usage)  # an unusable turn still costs its tokens
            problem = _unusable_turn(turn, spec.max_tokens) or ""
            if not problem:
                return {
                    "messages": {
                        **state["messages"],
                        spec.role: [*prompt, turn.assistant_message or _assistant_message(turn)],
                    },
                    "pending_calls": [call.model_dump(mode="json") for call in turn.tool_calls],
                    "last_text": turn.content,
                    "budget": budget.model_dump(mode="json"),
                }
            prompt = [*history, harness_message(UNUSABLE_TURN.format(problem=problem))]
        return {
            "stop_reason": "api_error",
            "final_answer": UNUSABLE_TURN_STOP.format(count=MAX_UNUSABLE_TURN_RETRIES + 1, problem=problem),
            "pending_calls": [],
            "budget": budget.model_dump(mode="json"),
        }

    @staticmethod
    def _route_agent(state: WorkflowState) -> str:
        if state.get("stop_reason"):
            return "finish"
        if state.get("pending_calls"):
            budget = _budget(state)
            return "give_up" if budget.steps >= budget.max_steps else "tools"
        return "handoff"

    def _node_tools(self, state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
        thread_id = _thread_id(config)
        spec = AGENTS[state["phase"]]
        budget = _budget(state)
        history = list(state["messages"][spec.role])
        view: dict[str, Any] = dict(state)  # state as seen by the next call (artifacts harvested in this node)
        entries: list[dict[str, Any]] = []
        pending = list(state.get("pending_calls") or [])
        step, exhausted = len(state.get("trace") or []), False
        for index, raw in enumerate(pending):
            if index >= MAX_CALLS_PER_TURN:  # ReAct: one call per model turn
                history = _answer_skipped(history, pending[index:], ONE_CALL_PER_TURN)
                break
            if budget.steps >= budget.max_steps:
                exhausted = True
                history = _answer_skipped(history, pending[index:])
                break
            step += 1
            call = NormalizedToolCall.model_validate(raw)
            entry, outcome = self._run_agent_call(spec, call, view, thread_id, step)
            budget = budget.model_copy(
                update={"steps": budget.steps + 1, "latency_s": budget.latency_s + entry.latency_s}
            )
            history.append(
                tool_result_message(
                    call.id, render_tool_result(entry.tool, outcome.result, outcome.error, entry.attempts)
                )
            )
            entries.append(entry.model_dump(mode="json"))
            view.update(self._harvest(entry.tool, outcome, view))
            self._emit(thread_id, "step", step=entry, budget=budget)
            if outcome.ok:
                self._emit_artifacts(thread_id, budget, ARTIFACTS_BY_TOOL.get(entry.tool, ()), quiz=view.get("quiz"))
        return {
            **{key: view[key] for key in HARVESTED_KEYS if key in view},
            "messages": {**state["messages"], spec.role: history},
            "pending_calls": [],
            "trace": entries,
            "budget": budget.model_dump(mode="json"),
            "budget_exhausted": exhausted,
        }

    def _node_handoff(self, state: WorkflowState) -> Command[Literal["agent", "quiz_gate", "finish"]]:
        role = state["phase"]
        text = state.get("last_text") or ""
        status = parse_agent_status(text, wrote_staging=bool(state.get("wrote_staging")))
        answer = strip_status_line(text)
        if role == "examiner" and status == "clarification_needed":
            # The Examiner has no channel to the developer (a follow-up goes to the Coder): its question is a dead
            # end, so its deliverables decide instead (a nudge naming what is missing, or the quiz gate).
            status = "done"
        if status != "done":
            return _finish(status, answer)
        diff = self.workspace.get_diff_report()
        if role in ("coder", "single"):
            if diff.trees_identical:
                return _finish("completed", answer or NOTHING_STAGED)
            gate = self.workspace.get_gate_status(diff.diff_sha256)
            if gate.blocked:
                return _finish("refused", DIFF_BLOCKED)
            if not gate.tests_passed:
                return self._nudge(state, NUDGE_TESTS)
            if role == "coder":
                return self._examiner_turn(state, diff, attempt=1, extra={"final_answer": answer})
        missing = missing_deliverables(
            ship_log=state.get("ship_log"),
            quiz=state.get("quiz"),
            diff_sha256=diff.diff_sha256,
            attempt=state["attempt"],
        )
        if missing:
            return self._nudge(state, f"Before you finish: {'; '.join(missing)}.")
        return Command(goto="quiz_gate", update={"final_answer": answer, "nudges": 0})

    def _node_quiz_gate(
        self, state: WorkflowState, config: RunnableConfig
    ) -> Command[Literal["deploy_gate", "agent", "finish"]]:
        thread_id = _thread_id(config)
        quiz = PublicQuizView.model_validate(state["quiz"])
        prompt = QUIZ_PROMPT.format(attempt=quiz.attempt, max_attempts=quiz.max_attempts)
        answer = _ask_human(
            HumanInterrupt(kind="quiz", thread_id=thread_id, message=prompt, quiz=quiz), _parse_quiz_answer
        )
        # ---- below runs once, after the human answered (code above re-runs on resume, so it stays side-effect free)
        user_id = state.get("user_id") or DEFAULT_USER_ID
        grade = self.workspace.grade_quiz_answer(quiz.question_id, answer.selected_option, user_id=user_id)
        evidence = (state.get("ship_log") or {}).get("title") or state["task"]
        self.memory.upsert_quiz_outcome(user_id, grade, evidence=evidence)
        entry = ToolCallTrace(
            step=len(state.get("trace") or []) + 1,
            agent="human",
            tool="submit_quiz_answer",
            args={"question_id": quiz.question_id, "selected_option": answer.selected_option},
            result=grade.model_dump(mode="json"),
        )
        budget = _budget(state)
        self._emit(thread_id, "step", step=entry, budget=budget)
        self._emit_artifacts(thread_id, budget, ("ship_log", "locks"))
        update = {"trace": [entry.model_dump(mode="json")], "last_grade": grade.model_dump(mode="json")}
        if grade.passed:
            return Command(goto="deploy_gate", update=update)
        if grade.remediation_required:
            update |= {"attempt": grade.attempt + 1, "prior_failed_grade": grade.model_dump(mode="json")}
            if state.get("topology") == "single":
                history = [*state["messages"]["single"], build_single_remediation_message(grade)]
                messages = {**state["messages"], "single": history}
                return Command(goto="agent", update={**update, "phase": "single", "messages": messages, "nudges": 0})
            return self._examiner_turn(state, self.workspace.get_diff_report(), attempt=grade.attempt + 1, extra=update)
        return _finish(
            "quiz_failed", QUIZ_BLOCKED.format(sha=grade.diff_sha256[:12], correct=grade.correct_option), **update
        )

    def _node_deploy_gate(self, state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
        thread_id = _thread_id(config)
        diff = self.workspace.get_diff_report()
        gate = self.workspace.get_gate_status(diff.diff_sha256)
        sha = diff.diff_sha256[:12]
        prompt = (
            DEPLOY_READY.format(sha=sha)
            if gate.ready_for_approval
            else DEPLOY_NOT_READY.format(sha=sha, missing=", ".join(gate.missing_locks) or "nothing")
        )
        request = HumanInterrupt(kind="deploy_approval", thread_id=thread_id, message=prompt, gate=gate, diff=diff)
        decision = _ask_human(request, _parse_deploy_decision)
        # ---- below runs once, after the human decided
        if decision.expected_diff_sha256 is None:  # the human approved what they were shown
            decision = decision.model_copy(update={"expected_diff_sha256": diff.diff_sha256})
        locks = self.workspace.apply_deploy_decision(decision, thread_id=thread_id)
        step, budget = len(state.get("trace") or []), _budget(state)
        human = ToolCallTrace(
            step=step + 1,
            agent="human",
            tool="submit_deploy_decision",
            args=decision.model_dump(mode="json"),
            result=locks.model_dump(mode="json"),
        )
        self._emit(thread_id, "step", step=human, budget=budget)
        if not decision.approved:
            self._emit_artifacts(thread_id, budget, ("ship_log",), locks=locks)
            return {
                "trace": [human.model_dump(mode="json")],
                "stop_reason": "human_rejected",
                "final_answer": REJECTED.format(sha=sha),
            }
        note = str((state.get("ship_log") or {}).get("title") or state["task"])[:RELEASE_NOTE_CHARS]
        args = {"expected_diff_sha256": decision.expected_diff_sha256, "release_note": note}
        key = idem_key(thread_id, "deploy_to_production", args)
        started = time.perf_counter()
        outcome, attempts = self._call_with_retry("deploy_to_production", args, key)
        system = ToolCallTrace(
            step=step + 2,
            agent="system",
            server=TOOLS["deploy_to_production"].server,
            tool="deploy_to_production",
            args=args,
            result=outcome.result,
            error=outcome.error,
            latency_s=time.perf_counter() - started,
            idempotency_key=key,
            attempts=attempts,
            replayed=outcome.replayed,
        )
        self._emit(thread_id, "step", step=system, budget=budget)
        self._emit_artifacts(thread_id, budget, ("diff", "ship_log"), locks=locks)
        trace = [human.model_dump(mode="json"), system.model_dump(mode="json")]
        if outcome.ok:
            answer = DEPLOYED.format(sha=sha) + _release_suffix(outcome.result)
            return {"trace": trace, "stop_reason": "completed", "final_answer": answer}
        message = outcome.error.message if outcome.error else "unknown error"
        return {"trace": trace, "stop_reason": "refused", "final_answer": GATE_REFUSED.format(message=message)}

    @staticmethod
    def _node_give_up(state: WorkflowState) -> dict[str, Any]:
        budget = _budget(state)
        update: dict[str, Any] = {
            "stop_reason": "max_steps",
            "final_answer": BUDGET_EXHAUSTED.format(steps=budget.steps, max_steps=budget.max_steps),
            "pending_calls": [],
        }
        if pending := state.get("pending_calls"):  # keep the history valid for a follow-up turn
            role = state["phase"]
            update["messages"] = {**state["messages"], role: _answer_skipped(state["messages"][role], pending)}
        return update

    @staticmethod
    def _node_finish(state: WorkflowState) -> dict[str, Any]:
        return {}

    # ---- node helpers ----------------------------------------------------------------------------------------------
    def _examiner_turn(
        self, state: WorkflowState, diff: DiffReport, *, attempt: int, extra: Mapping[str, Any]
    ) -> Command[Literal["agent"]]:
        """A fresh, isolated Examiner context for one attempt (dual topology)."""
        user_id = state.get("user_id") or DEFAULT_USER_ID
        brief = build_examiner_brief(
            state["task"],
            diff,
            render_applied_rules(state.get("rules_read") or {}),
            self.memory.format_memory_prompt_block(user_id),
            attempt,
            _grade(extra.get("prior_failed_grade")),
            pytest=self.workspace.get_latest_pytest(),
        )
        messages = {**state["messages"], "examiner": build_examiner_messages(brief)}
        return Command(
            goto="agent", update={**extra, "phase": "examiner", "messages": messages, "attempt": attempt, "nudges": 0}
        )

    @staticmethod
    def _nudge(state: WorkflowState, text: str) -> Command[Literal["agent", "finish"]]:
        """A harness reminder to the active agent, at most ``MAX_HANDOFF_NUDGES`` times, then a stop."""
        nudges = state.get("nudges", 0)
        if nudges >= MAX_HANDOFF_NUDGES:
            return _finish("refused", f"Stopped by the harness after {nudges} reminders: {text}")
        role = state["phase"]
        messages = {**state["messages"], role: [*state["messages"][role], harness_message(text)]}
        return Command(goto="agent", update={"messages": messages, "nudges": nudges + 1})

    def _run_agent_call(
        self, spec: AgentSpec, call: NormalizedToolCall, view: Mapping[str, Any], thread_id: str, step: int
    ) -> tuple[ToolCallTrace, ToolOutcome]:
        """One LLM-initiated call: refuse what the agent may not call, bind harness args, key writes, retry."""
        tool: ToolDef | None = TOOLS.get(call.name)
        args: dict[str, Any] = dict(call.args)
        key: str | None = None
        attempts = 1
        started = time.perf_counter()
        if call.args_error:
            outcome = _refusal("invalid_argument", f"{call.name}: {call.args_error}")
        elif tool is None or not spec.can_call(call.name):
            outcome = _refusal("forbidden", f"{call.name!r} is not available to the {spec.role}: use only your tools")
        else:
            args = self._bind_args(tool, args, view, thread_id)
            key = self._idempotency_key(tool, args, thread_id) if tool.writes else None
            outcome, attempts = self._call_with_retry(tool.name, args, key)
            if tool.name == "create_comprehension_quiz":
                outcome = _public_quiz(outcome)
        entry = ToolCallTrace(
            step=step,
            agent=spec.role,
            server=tool.server if tool else None,
            tool=call.name or "unknown_tool",
            args=tool.traced_args(args) if tool else args,
            result=outcome.result,
            error=outcome.error,
            latency_s=time.perf_counter() - started,
            idempotency_key=key,
            attempts=attempts,
            replayed=outcome.replayed,
            contains_injection=_carries(outcome.result, view.get("injection_text")),
        )
        return entry, outcome

    def _bind_args(
        self, tool: ToolDef, args: Mapping[str, Any], view: Mapping[str, Any], thread_id: str
    ) -> dict[str, Any]:
        """The model's arguments minus anything harness-owned, plus the authoritative binding for doc and quiz."""
        model_args = {key: value for key, value in args.items() if key not in tool.harness_args}
        if tool.name == "upsert_ship_log_doc":
            return bind_ship_log_args(
                model_args,
                thread_id=thread_id,
                diff=self.workspace.get_diff_report(),
                attempt=view.get("attempt", 1),
                weaknesses=self.memory.weak_skill_tags(view.get("user_id") or DEFAULT_USER_ID),
                rules_read=(view.get("rules_read") or {}).keys(),
                prior_failed_grade=_grade(view.get("prior_failed_grade")),
            )
        if tool.name == "create_comprehension_quiz":
            diff_sha256 = self.workspace.get_diff_report().diff_sha256
            return bind_quiz_args(
                model_args, thread_id=thread_id, diff_sha256=diff_sha256, attempt=view.get("attempt", 1)
            )
        return model_args

    def _idempotency_key(self, tool: ToolDef, args: Mapping[str, Any], thread_id: str) -> str:
        binding = dict(args)
        if tool.name in TREE_BOUND_TOOLS:
            binding[TREE_KEY_FIELD] = self.workspace.get_diff_report().staging_tree_sha256
        return idem_key(thread_id, tool.name, binding)

    def _call_with_retry(self, tool: str, args: Mapping[str, Any], key: str | None) -> tuple[ToolOutcome, int]:
        """Bounded retry of retryable errors with the SAME idempotency key: a lost answer is replayed, not redone."""
        payload = {**args, IDEMPOTENCY_ARG: key} if key else dict(args)
        attempts = 0
        while True:
            attempts += 1
            outcome = self.bridge.call(tool, payload)
            if outcome.ok or outcome.error is None or not outcome.error.retryable or attempts > MAX_TOOL_RETRIES:
                return outcome, attempts
            self._sleep(RETRY_BACKOFF_S[min(attempts, len(RETRY_BACKOFF_S)) - 1])

    @staticmethod
    def _harvest(tool: str, outcome: ToolOutcome, view: Mapping[str, Any]) -> dict[str, Any]:
        """State the harness keeps from a successful call (rules read, staged writes, doc, quiz)."""
        result = outcome.result
        if not outcome.ok:
            return {}
        if tool == "read_notion_rule" and isinstance(result, Mapping) and result.get("rule_code"):
            entry = {"title": str(result.get("title", "")), "body": str(result.get("body_markdown", ""))}
            return {"rules_read": {**(view.get("rules_read") or {}), str(result["rule_code"]): entry}}
        if tool == "write_staging_file":
            return {"wrote_staging": True}
        if tool in ("upsert_ship_log_doc", "create_comprehension_quiz") and isinstance(result, Mapping):
            return {"ship_log" if tool == "upsert_ship_log_doc" else "quiz": dict(result)}
        return {}

    # ---- events ----------------------------------------------------------------------------------------------------
    @contextmanager
    def _listening(self, thread_id: str, on_event: Callable[[RunEvent], None] | None) -> Iterator[None]:
        if on_event is None:
            yield
            return
        with self._lock:
            self._sinks[thread_id] = on_event
        try:
            yield
        finally:
            with self._lock:
                self._sinks.pop(thread_id, None)

    def _emit(self, thread_id: str, event_type: RunEventType, **payload: Any) -> None:
        callback = self._sinks.get(thread_id)
        if callback is None:
            return
        with self._lock:
            seq = self._event_seq.get(thread_id, 0)
            self._event_seq[thread_id] = seq + 1
        try:
            callback(RunEvent(type=event_type, thread_id=thread_id, seq=seq, **payload))
        except Exception:  # a broken listener must never break the run
            logger.exception("on_event listener failed for thread %s", thread_id)

    def _emit_artifacts(
        self,
        thread_id: str,
        budget: RunBudget,
        kinds: tuple[str, ...],
        *,
        quiz: Mapping[str, Any] | None = None,
        locks: DeployGateStatus | None = None,
    ) -> None:
        if thread_id not in self._sinks or not (kinds or locks):
            return
        payload: dict[str, Any] = {}
        if "diff" in kinds or ("locks" in kinds and locks is None):
            diff = self.workspace.get_diff_report()
            if "diff" in kinds:
                payload["diff"] = diff
            if "locks" in kinds and locks is None:
                locks = self.workspace.get_gate_status(diff.diff_sha256)
        if locks is not None:
            payload["locks"] = locks
        if "pytest" in kinds:
            payload["pytest"] = self.workspace.get_latest_pytest()
        if "ship_log" in kinds:
            payload["ship_log"] = self.workspace.get_latest_ship_log(thread_id)
        if "quiz" in kinds and quiz:
            payload["quiz"] = PublicQuizView.model_validate(quiz)
        payload = {name: value for name, value in payload.items() if value is not None}
        if payload:
            self._emit(thread_id, "artifact_updated", budget=budget, **payload)

    # ---- runner API (#14 §5) ---------------------------------------------------------------------------------------
    def start_run(
        self,
        task: str,
        *,
        thread_id: str,
        user_id: str = DEFAULT_USER_ID,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> WorkflowOutcome:
        """Start a thread (or a follow-up turn on a finished one) and run it to its first pause or its end."""
        return self._start(task, thread_id=thread_id, user_id=user_id, max_steps=max_steps, on_event=on_event)

    def _start(
        self,
        task: str,
        *,
        thread_id: str,
        user_id: str,
        max_steps: int,
        on_event: Callable[[RunEvent], None] | None,
        injection_text: str | None = None,
    ) -> WorkflowOutcome:
        if not task or not task.strip():
            raise ValueError("task must be a non-empty developer request")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        config = _config(thread_id)
        if is_forget_command(task):
            return self._start_forget(thread_id, user_id=user_id, max_steps=max_steps, on_event=on_event)
        snapshot = self.app.get_state(config)
        if _pending(snapshot) is not None:
            raise ValueError(
                f"thread {thread_id!r} is paused on a human gate: use submit_quiz / submit_deploy_decision"
            )
        prior = _budget(snapshot.values or {})
        with self._listening(thread_id, on_event):
            self._emit(
                thread_id,
                "run_started",
                model=self.model,
                topology=(snapshot.values or {}).get("topology") or self.topology,
                budget=prior.model_copy(update={"max_steps": prior.steps + max_steps}),
            )
            payload = {"task": task, "user_id": user_id, "max_steps": max_steps, "injection_text": injection_text}
            return self._drive(config, payload)

    def submit_quiz(
        self,
        thread_id: str,
        answer: QuizAnswer,
        *,
        user_id: str = DEFAULT_USER_ID,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> tuple[QuizGradeResult, WorkflowOutcome]:
        """Answer the pending quiz: graded server-side, then the run continues (remediation, deploy gate or stop).

        The thread's developer (the ``user_id`` of ``start_run``) owns the Layer 4 update; ``user_id`` only applies
        to a thread that has none."""
        config = _config(thread_id)
        snapshot = self.app.get_state(config)
        pending = _pending(snapshot)
        if pending is None or pending.kind != "quiz" or pending.quiz is None:
            raise ValueError(f"thread {thread_id!r} is not waiting for a quiz answer")
        update = {} if (snapshot.values or {}).get("user_id") else {"user_id": user_id}
        with self._listening(thread_id, on_event):
            outcome = self._drive(config, Command(resume={"selected_option": answer.selected_option}, update=update))
        grade = _grade(self.app.get_state(config).values.get("last_grade"))
        if grade is None or grade.question_id != pending.quiz.question_id:
            raise RuntimeError(f"the answer was not graded: {outcome.final_answer or 'the run stopped before grading'}")
        return grade, outcome

    def submit_deploy_decision(
        self,
        thread_id: str,
        decision: DeployDecision,
        *,
        user_id: str = DEFAULT_USER_ID,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> WorkflowOutcome:
        """Answer the pending deploy approval: approve (then deploy through the 4 locks) or reject (blocked)."""
        config = _config(thread_id)
        pending = _pending(self.app.get_state(config))
        if pending is None or pending.kind != "deploy_approval":
            raise ValueError(f"thread {thread_id!r} is not waiting for a deploy decision")
        with self._listening(thread_id, on_event):
            return self._drive(config, Command(resume=decision.model_dump(mode="json")))

    def _drive(self, config: RunnableConfig, payload: Any) -> WorkflowOutcome:
        thread_id = _thread_id(config)
        try:
            self.app.invoke(payload, config)
        except Exception as exc:  # a crash ends the stream cleanly (UI) and scores as api_error (eval)
            logger.exception("workflow crashed on thread %s", thread_id)
            return self._conclude(config, crash=f"{type(exc).__name__}: {exc}")
        return self._conclude(config)

    def _conclude(self, config: RunnableConfig, *, crash: str | None = None) -> WorkflowOutcome:
        thread_id = _thread_id(config)
        snapshot = self.app.get_state(config)
        values = snapshot.values or {}
        pending = None if crash else _pending(snapshot)
        stop: StopReason | None = None if pending else values.get("stop_reason")
        text = crash or values.get("final_answer") or ""
        if crash or (pending is None and stop is None):
            stop, text = "api_error", text or "the workflow ended without a stop reason"
        budget = _budget(values)
        outcome = WorkflowOutcome(
            thread_id=thread_id,
            stop_reason=stop,
            interrupt=pending,
            final_answer=text,
            trace=[ToolCallTrace.model_validate(entry) for entry in values.get("trace") or []],
            budget=budget,
            final_state=self.workspace.snapshot_final_state(thread_id),
        )
        if stop == "api_error":
            self._emit(thread_id, "run_error", stop_reason="api_error", text=text, budget=budget)
        else:
            self._emit(
                thread_id, "run_completed", interrupt=pending, stop_reason=stop, text=text or None, budget=budget
            )
        return outcome

    # ---- workspace, memory and evaluation --------------------------------------------------------------------------
    def reset_workspace(
        self,
        seed: str = DEFAULT_SEED,
        *,
        fault: FaultKind | None = None,
        injection: InjectionSpec | None = None,
        preserve_skill_memory: bool = False,
    ) -> RunFinalState:
        """Reset the sandbox, then restart the MCP servers so they start from the reset state."""
        state = self.workspace.reset(
            seed, fault=fault, injection=injection, preserve_skill_memory=preserve_skill_memory
        )
        self.bridge.restart()
        return state

    def run_scenario(self, scenario: ScenarioSpec, *, repeat: int = 0, user_id: str = DEFAULT_USER_ID) -> EvalRunResult:
        """One scored eval run: reset, start, answer every human gate as ``scenario.persona``, score the end state.

        The agent's hard cap is ``scenario.max_steps + EVAL_STEP_HEADROOM``: the scenario budget scores the run
        (``over_budget`` stays observable) while the cap stops runaway loops.
        """
        thread_id = f"eval-{scenario.id}-r{repeat}-{uuid.uuid4().hex[:8]}"
        outcome: WorkflowOutcome | None = None
        error: str | None = None
        try:
            self.reset_workspace(scenario.seed, fault=scenario.fault, injection=scenario.injection)
            outcome = self._start(
                scenario.task,
                thread_id=thread_id,
                user_id=user_id,
                max_steps=scenario.max_steps + EVAL_STEP_HEADROOM,
                on_event=None,
                injection_text=scenario.injection.text if scenario.injection else None,
            )
            for _ in range(MAX_HITL_ROUNDS):
                if outcome.interrupt is None:
                    break
                outcome = self._persona_turn(scenario.persona, outcome, user_id)
            if outcome.interrupt is not None:
                error = "the simulated human never reached a terminal state"
        except Exception as exc:  # one broken run must not abort a 30-scenario campaign
            logger.exception("scenario %s#%s crashed", scenario.id, repeat)
            error = f"{type(exc).__name__}: {exc}"
        if outcome is not None and outcome.stop_reason == "api_error":
            error = error or outcome.final_answer
        stop: StopReason = (
            "api_error" if error or outcome is None or outcome.stop_reason is None else outcome.stop_reason
        )
        if outcome is None:
            values = self.app.get_state(_config(thread_id)).values or {}
            trace = [ToolCallTrace.model_validate(entry) for entry in values.get("trace") or []]
            budget, final_state, answer = _budget(values), self.workspace.snapshot_final_state(thread_id), ""
        else:
            trace, budget, final_state, answer = (
                outcome.trace,
                outcome.budget,
                outcome.final_state,
                outcome.final_answer,
            )
        return EvalRunResult.from_run(
            scenario=scenario,
            repeat=repeat,
            final_state=final_state,
            trace=trace,
            stop_reason=stop,
            final_answer=answer,
            model=self.model,
            topology=self.topology,
            tokens_in=budget.tokens_in,
            tokens_out=budget.tokens_out,
            cost_usd=budget.cost_usd,
            latency_s=budget.latency_s,
            error=error,
        )

    def _persona_turn(self, persona: HumanPersona, outcome: WorkflowOutcome, user_id: str) -> WorkflowOutcome:
        """The simulated human's answer to the pending gate."""
        request = outcome.interrupt
        if request is None:
            return outcome
        if request.kind == "quiz" and request.quiz is not None:
            option = self._persona_option(persona, outcome.thread_id, request.quiz)
            return self.submit_quiz(outcome.thread_id, QuizAnswer(selected_option=option), user_id=user_id)[1]
        approve = persona != "reject_or_exhaust_quiz"
        decision = DeployDecision(
            action="approve" if approve else "reject",
            expected_diff_sha256=request.gate.diff_sha256 if request.gate else None,
            reason=f"simulated human persona: {persona}",
        )
        return self.submit_deploy_decision(outcome.thread_id, decision, user_id=user_id)

    def _persona_option(self, persona: HumanPersona, thread_id: str, quiz: PublicQuizView) -> QuizOptionKey:
        private = self.workspace.get_latest_private_quiz(thread_id, diff_sha256=quiz.diff_sha256)
        correct = private.correct_option if private is not None and private.question_id == quiz.question_id else None
        if correct is None:  # the key is unknown to the harness: a deterministic guess
            return ALL_QUIZ_OPTION_KEYS[0]
        wants_right = persona == "pass_first_try" or (persona == "fail_once_then_pass" and quiz.attempt > 1)
        return correct if wants_right else next(key for key in ALL_QUIZ_OPTION_KEYS if key != correct)

    def request_forget(self, user_id: str = DEFAULT_USER_ID, *, thread_id: str) -> dict[str, Any]:
        """``/forget``, step 1: pause on the confirmation interrupt; returns what would be erased."""
        config = _config(f"{FORGET_THREAD_PREFIX}{thread_id}")
        if not self.forget_app.get_state(config).interrupts:  # asking twice returns the pending confirmation
            self.forget_app.invoke({"user_id": user_id}, config)
        snapshot = self.forget_app.get_state(config)
        return dict(snapshot.interrupts[0].value) if snapshot.interrupts else dict(snapshot.values)

    def confirm_forget(self, thread_id: str, *, approve: bool) -> dict[str, Any]:
        """``/forget``, step 2: resume the confirmation; returns the report (``deleted`` count and ``reply``)."""
        config = _config(f"{FORGET_THREAD_PREFIX}{thread_id}")
        if not self.forget_app.get_state(config).interrupts:
            raise ValueError(f"thread {thread_id!r} has no pending /forget confirmation")
        self.forget_app.invoke(Command(resume={"approve": approve}), config)
        return dict(self.forget_app.get_state(config).values)

    def _start_forget(
        self, thread_id: str, *, user_id: str, max_steps: int, on_event: Callable[[RunEvent], None] | None
    ) -> WorkflowOutcome:
        """``/forget`` typed as a task: ask for the confirmation (a clarification) instead of running agents."""
        request = self.request_forget(user_id, thread_id=thread_id)
        text = f"{request.get('message', 'Confirm the erasure.')} Confirm with confirm_forget(thread_id, approve=True)."
        budget = RunBudget(max_steps=max_steps)
        with self._listening(thread_id, on_event):
            self._emit(thread_id, "run_started", model=self.model, topology=self.topology, budget=budget)
            self._emit(thread_id, "run_completed", stop_reason="clarification_needed", text=text, budget=budget)
        return WorkflowOutcome(
            thread_id=thread_id,
            stop_reason="clarification_needed",
            final_answer=text,
            budget=budget,
            final_state=self.workspace.snapshot_final_state(thread_id),
        )


def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """LangGraph Studio / ``langgraph dev`` entry point (see ``langgraph.json``): the platform brings persistence."""
    configurable = dict((config or {}).get("configurable") or {})
    topology = configurable.get("topology", "dual")
    workflow = NoQuizWorkflow(
        topology=topology if topology in ("dual", "single") else "dual", model=configurable.get("model")
    )
    return workflow.build_graph()


__all__ = [
    "CHECKPOINT_FILENAME",
    "EVAL_STEP_HEADROOM",
    "MAX_HANDOFF_NUDGES",
    "MAX_TOOL_RETRIES",
    "MAX_UNUSABLE_TURN_RETRIES",
    "RETRY_BACKOFF_S",
    "NoQuizWorkflow",
    "WorkflowInput",
    "WorkflowOutcome",
    "WorkflowState",
    "WorkspaceHarness",
    "make_graph",
]
