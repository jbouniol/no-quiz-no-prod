"""Shared deterministic test kit (single source for every test module; pytest puts ``tests/`` on ``sys.path``).

- ``ScriptedLLM``: an ``LLMClient`` stand-in that plays each agent (Coder, Examiner, Single) from its own queue of
  scripted turns, recognised by the role header of the system prompt, and records every call it receives.
- ``FIXED_FINTECH_ORDERS``: a RULE-001 compliant ``orders.py`` that keeps the seeded ``test_fintech.py`` suite green.
- ``parse_sse_events``: decodes a Studio SSE body and validates every frame against the frozen ``RunEvent`` contract.
- ``HERMETIC_RELEASE_ENV``: the release-remote variables every test process and subprocess gets, so a GitHub remote
  or token in the developer's ``.env`` can never be read, let alone pushed to (``conftest.py`` applies it).
"""

import copy
import json
from collections import deque
from types import SimpleNamespace
from typing import Any, Final

from no_quiz_no_prod.agents.prompts import ROLE_HEADER_CODER, ROLE_HEADER_EXAMINER, ROLE_HEADER_SINGLE
from no_quiz_no_prod.contracts import DEFAULT_GIT_BRANCH, EnvVar, RunEvent
from no_quiz_no_prod.llm import ChatTurnResponse, LLMUsage, NormalizedToolCall

SCRIPTED_MODEL: Final = "fake/scripted"
# Set (empty) wins over .env: load_project_env never replaces a variable that is already set, even an empty one.
HERMETIC_RELEASE_ENV: Final[dict[str, str]] = {
    EnvVar.NO_QUIZ_GIT_REMOTE_URL.value: "",
    EnvVar.NO_QUIZ_GITHUB_TOKEN.value: "",
    EnvVar.NO_QUIZ_GIT_BRANCH.value: DEFAULT_GIT_BRANCH,
}
USAGE: Final = LLMUsage(tokens_in=100, tokens_out=20, cost_usd=0.0001, latency_s=0.0, model=SCRIPTED_MODEL)
ROLE_BY_HEADER: Final = {ROLE_HEADER_CODER: "coder", ROLE_HEADER_EXAMINER: "examiner", ROLE_HEADER_SINGLE: "single"}

FIXED_FINTECH_ORDERS: Final = '''"""Checkout totals and installment plans (RULE-001: Decimal money, cent quantization, ROUND_HALF_UP)."""

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_money(value: Decimal | str | int) -> Decimal:
    """Parse a money value and quantize it to the cent with ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def calculate_order_total(
    items: list[dict], vat_rate: str | float = "0.20", discount_rate: str | float = "0.00"
) -> Decimal | float:
    """Return the amount to charge for ``items``: the discounted subtotal plus VAT, exact to the cent."""
    subtotal = Decimal("0.00")
    for item in items:
        price, quantity = Decimal(str(item["price"])), int(item["quantity"])
        if price < 0 or quantity < 0:
            raise ValueError("price and quantity must be non-negative")
        subtotal += to_money(price * quantity)
    net = subtotal - to_money(subtotal * Decimal(str(discount_rate)))
    return net + to_money(net * Decimal(str(vat_rate)))


def split_into_installments(total: str | Decimal, count: int) -> list[Decimal]:
    """Split ``total`` into ``count`` installments summing exactly to it (early ones take the extra cents)."""
    if count <= 0:
        raise ValueError("count must be positive")
    base, remainder = divmod(int(to_money(total) / CENT), count)
    return [(base + (1 if index < remainder else 0)) * CENT for index in range(count)]
'''


# ---- the scripted LLM ----------------------------------------------------------------------------------------------
def call(name: str, **args: Any) -> tuple[str, dict[str, Any]]:
    """One scripted tool call: ``(tool name, arguments)``."""
    return name, args


def turn(*calls: tuple[str, dict[str, Any]], text: str = "", finish_reason: str | None = None) -> ChatTurnResponse:
    """One scripted model turn: tool calls (OpenAI format) and/or a final text (``finish_reason`` overridable,
    e.g. ``"length"`` for a thinking model that spent its whole output budget on hidden reasoning)."""
    tool_calls = [
        NormalizedToolCall(id=f"call_{index}_{name}", name=name, args=args) for index, (name, args) in enumerate(calls)
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
            for c in tool_calls
        ]
    return ChatTurnResponse(
        content=text,
        tool_calls=tool_calls,
        usage=USAGE,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        assistant_message=message,
    )


class ScriptedLLM:
    """Plays each agent from a queue of turns; records every call (role, messages, visible tool names)."""

    model = SCRIPTED_MODEL

    def __init__(self, **scripts: list[ChatTurnResponse]) -> None:
        self.scripts = {role: deque(turns) for role, turns in scripts.items()}
        self.calls: list[SimpleNamespace] = []

    def chat_with_tools(self, messages, tools=None, model=None, temperature=0.0, max_tokens=2000) -> ChatTurnResponse:
        system = messages[0]["content"]
        role = next(role for header, role in ROLE_BY_HEADER.items() if system.startswith(header))
        tool_names = [spec["function"]["name"] for spec in tools or []]
        self.calls.append(SimpleNamespace(role=role, messages=copy.deepcopy(list(messages)), tools=tool_names))
        queue = self.scripts.get(role)
        if not queue:
            raise AssertionError(f"the {role} has no scripted turn left")
        return queue.popleft()

    def briefs(self, role: str = "examiner") -> list[str]:
        """The first user message of every fresh context of ``role``."""
        return [c.messages[1]["content"] for c in self.calls if c.role == role and len(c.messages) == 2]

    def remaining(self) -> dict[str, int]:
        """Scripted turns not consumed yet, per role (all zero once a scripted flow ran exactly as planned)."""
        return {role: len(queue) for role, queue in self.scripts.items()}


# ---- Studio SSE ----------------------------------------------------------------------------------------------------
def parse_sse_events(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    """``[(event type, payload)]`` of an SSE body; every frame must validate as a ``RunEvent``."""
    frames: list[tuple[str, dict[str, Any]]] = []
    for chunk in raw_body.replace("\r\n", "\n").split("\n\n"):
        if not chunk.strip():
            continue
        ev_type = ""
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                ev_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            RunEvent.model_validate(payload)  # every streamed frame honours the frozen RunEvent contract
            frames.append((ev_type, payload))
    return frames


__all__ = [
    "FIXED_FINTECH_ORDERS",
    "ROLE_BY_HEADER",
    "SCRIPTED_MODEL",
    "USAGE",
    "ScriptedLLM",
    "call",
    "parse_sse_events",
    "turn",
]
