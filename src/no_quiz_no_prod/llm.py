"""Unified LLM client: OpenRouter (OpenAI-compatible) and native Anthropic behind one normalized interface.

Callers speak the OpenAI chat format everywhere: ``messages`` with ``role`` / ``content`` / ``tool_calls`` /
``tool_call_id``, and OpenAI function ``tools``. This module translates to the Anthropic Messages API when the route is
native Anthropic, and normalizes every answer into a ``ChatTurnResponse`` (text, tool calls, usage with tokens, USD
cost and latency). Failures of either SDK surface as one ``LLMError``, which the graph maps to
``stop_reason="api_error"``.

Routing: ``provider="anthropic"``, or a model id starting with ``claude`` and without ``/`` (``claude-sonnet-4-5``),
goes to the native Anthropic API with ``ANTHROPIC_API_KEY``. Anything else goes to OpenRouter with
``OPENROUTER_API_KEY`` (``~deepseek/deepseek-v4-flash-latest``, ``anthropic/claude-sonnet-4.5``...).

Cost: OpenRouter reports what it charged in ``usage.cost``; when it does not (and always for native Anthropic), the
cost is estimated from ``PRICE_PER_MTOK``. Note that the installed Anthropic SDK exposes no sampling parameters, so
``temperature`` only applies to the OpenRouter route.
"""

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import anthropic
import openai
from pydantic import BaseModel, Field, ValidationError

from no_quiz_no_prod.contracts import ENV_FILE, ContractModel, EnvVar, LLMProvider, RuntimeSettings

DEFAULT_TEMPERATURE: Final = 0.0
DEFAULT_MAX_TOKENS: Final = 2000
REQUEST_TIMEOUT_S: Final = 90.0  # per HTTP request timeout
MAX_SDK_RETRIES: Final = 3  # transport-level retries done by the SDKs themselves
STRUCTURED_MAX_ATTEMPTS: Final = 2  # extra chance for a reply that does not validate against the schema
STRUCTURED_TOOL_NAME: Final = "emit_structured_output"  # forced tool used for structured output on Anthropic

# USD per million tokens (input, output). Only a fallback: OpenRouter's own `usage.cost` always wins.
PRICE_PER_MTOK: Final[Mapping[str, tuple[float, float]]] = {
    "~deepseek/deepseek-v4-flash-latest": (0.05, 0.16),
    "deepseek/deepseek-v4-flash": (0.05, 0.16),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-5": (5.0, 25.0),
}
# Substring fallbacks, checked in order, for model ids missing from the table above.
PRICE_FAMILY_FALLBACKS: Final[tuple[tuple[str, tuple[float, float]], ...]] = (
    ("claude-opus", (5.0, 25.0)),
    ("claude-sonnet", (3.0, 15.0)),
    ("claude-haiku", (1.0, 5.0)),
    ("deepseek", (0.05, 0.16)),
)
# Anthropic stop reasons, in the OpenAI finish_reason vocabulary the rest of the project uses.
_FINISH_REASONS: Final[Mapping[str, str]] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}
_UNSAFE_ANTHROPIC_ID: Final = re.compile(r"[^a-zA-Z0-9_-]")


# ---- normalized results --------------------------------------------------------------------------------------------
class LLMUsage(ContractModel):
    """Usage of one call, or of several summed with ``+`` (``sum(usages, LLMUsage())``)."""

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    model: str = ""  # requested model id; "a+b" once usages of different models are summed

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        models = dict.fromkeys(name for part in (self.model, other.model) for name in part.split("+") if name)
        return LLMUsage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
            model="+".join(models),
        )


class NormalizedToolCall(ContractModel):
    """One tool call requested by the model, whatever the provider."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    args_error: str | None = None  # set when the provider sent arguments that are not a JSON object (args is then {})


class ChatTurnResponse(ContractModel):
    """One model turn: append ``assistant_message`` to the history, run ``tool_calls``, answer each with
    ``tool_result_message``."""

    content: str = ""
    tool_calls: list[NormalizedToolCall] = Field(default_factory=list)
    usage: LLMUsage
    finish_reason: str | None = None  # OpenAI vocabulary: stop | tool_calls | length | content_filter
    assistant_message: dict[str, Any] = Field(default_factory=dict)  # OpenAI format, provider extras preserved


class LLMError(RuntimeError):
    """No usable answer from a provider: missing key, network, auth, quota, empty or unparsable reply."""

    def __init__(self, message: str, *, provider: LLMProvider | None = None, model: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


# ---- pure helpers --------------------------------------------------------------------------------------------------
def resolve_provider(model: str | None, provider: LLMProvider | None = None) -> LLMProvider:
    """Explicit ``provider`` wins; a ``claude*`` id without ``/`` is native Anthropic; anything else is OpenRouter."""
    if provider is not None:
        return provider
    if model and "/" not in model and model.lstrip("~").startswith("claude"):
        return "anthropic"
    return "openrouter"


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost from ``PRICE_PER_MTOK`` (exact id first, then family substring); 0.0 for an unknown model."""
    price = PRICE_PER_MTOK.get(model) or next((p for family, p in PRICE_FAMILY_FALLBACKS if family in model), None)
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000 if price else 0.0


def available_providers() -> dict[str, bool]:
    """Which providers have an API key configured (``.env`` included). Never returns or logs a key."""
    return dict(RuntimeSettings.from_env().providers)


def function_tool(name: str, description: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """An OpenAI function-tool spec (e.g. from an MCP tool's name, description and input schema)."""
    schema = dict(parameters) if parameters else {"type": "object", "properties": {}}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


def tool_result_message(tool_call_id: str, content: Any) -> dict[str, Any]:
    """OpenAI ``tool`` message answering one tool call (text as is, models and objects as JSON)."""
    if isinstance(content, BaseModel):
        text = content.model_dump_json()
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    if not raw or not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"arguments are not valid JSON: {exc.msg}"
    return (value, None) if isinstance(value, dict) else ({}, "arguments must be a JSON object")


def _text_of(content: Any) -> str:
    """Text of an OpenAI message ``content``: a string, None, or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(content)


def _openai_assistant_message(content: str, calls: Sequence[NormalizedToolCall]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)},
            }
            for call in calls
        ]
    return message


def _anthropic_id(raw: Any) -> str:
    return _UNSAFE_ANTHROPIC_ID.sub("_", str(raw or "")) or "call"


def _to_anthropic_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI history -> (system prompt, Anthropic turns).

    Assistant tool calls become ``tool_use`` blocks, tool messages become ``tool_result`` blocks of a user turn, and
    consecutive turns of the same role are merged, because the Messages API requires alternating roles.
    """
    system: list[str] = []
    turns: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        role, text = message.get("role"), _text_of(message.get("content"))
        text_blocks = [{"type": "text", "text": text}] if text.strip() else []
        if role in ("system", "developer"):
            system.extend(block["text"] for block in text_blocks)
        elif role == "assistant":
            tool_uses = [
                {
                    "type": "tool_use",
                    "id": _anthropic_id(call.get("id")),
                    "name": call.get("function", {}).get("name", ""),
                    "input": _parse_arguments(call.get("function", {}).get("arguments"))[0],
                }
                for call in message.get("tool_calls") or []
            ]
            push("assistant", text_blocks + tool_uses)
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": _anthropic_id(message.get("tool_call_id")), "content": text}
            push("user", [result if text else {**result, "content": "(empty result)"}])
        else:  # "user", or any unknown role, is user input
            push("user", text_blocks)
    return "\n\n".join(system), turns


def _to_anthropic_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI function tools (or bare / MCP-style specs) -> Anthropic tool definitions."""
    converted = []
    for tool in tools:
        spec = tool.get("function", tool)
        schema = spec.get("parameters") or spec.get("input_schema") or spec.get("inputSchema")
        converted.append(
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": schema or {"type": "object", "properties": {}},
            }
        )
    return converted


def _validate_text[T: BaseModel](schema: type[T], text: str | None) -> T:
    """Validate a JSON reply, tolerating Markdown code fences and prose around the object."""
    cleaned = (text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return schema.model_validate_json(cleaned[start : end + 1] if 0 <= start < end else cleaned)


# ---- the client ----------------------------------------------------------------------------------------------------
class LLMClient:
    """One client, two providers. Thread-safe; async code uses the ``a``-prefixed wrappers (worker thread).

    ``model`` / ``provider`` set the defaults; every call may override ``model``, which re-routes that call only
    (e.g. a dual topology running the Coder on OpenRouter and the Examiner on native Claude).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: LLMProvider | None = None,
        settings: RuntimeSettings | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
        max_retries: int = MAX_SDK_RETRIES,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.provider: LLMProvider = resolve_provider(model, provider)
        self.model = model or (
            self.settings.anthropic_model if self.provider == "anthropic" else self.settings.openrouter_model
        )
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._openrouter_client: openai.OpenAI | None = None
        self._anthropic_client: anthropic.Anthropic | None = None

    def available_providers(self) -> dict[str, bool]:
        return dict(self.settings.providers)

    # -- public API --------------------------------------------------------------------------------------------------
    def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ChatTurnResponse:
        """One model turn with optional tools. Raises ``LLMError`` on any provider failure."""
        provider, model_id = self._route(model)
        try:
            if provider == "anthropic":
                return self._anthropic_chat(messages, tools, model_id, max_tokens)
            return self._openrouter_chat(messages, tools, model_id, temperature, max_tokens)
        except (openai.OpenAIError, anthropic.AnthropicError) as exc:
            raise self._provider_error(provider, model_id, exc) from exc

    def parse_structured[T: BaseModel](
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[T, LLMUsage]:
        """A reply validated against ``schema`` plus the usage of every attempt.

        OpenRouter uses JSON-schema structured outputs; native Anthropic uses one forced tool whose input schema is
        ``schema``. A reply that does not validate gets ``STRUCTURED_MAX_ATTEMPTS`` tries in total, then ``LLMError``.
        """
        provider, model_id = self._route(model)
        total, problem = LLMUsage(model=model_id), "no attempt"
        for _ in range(STRUCTURED_MAX_ATTEMPTS):
            try:
                if provider == "anthropic":
                    parsed, usage, problem = self._anthropic_structured(messages, schema, model_id, max_tokens)
                else:
                    parsed, usage, problem = self._openrouter_structured(
                        messages, schema, model_id, temperature, max_tokens
                    )
            except (openai.APIError, anthropic.AnthropicError) as exc:  # transport already retried by the SDK
                raise self._provider_error(provider, model_id, exc) from exc
            except (ValueError, openai.OpenAIError) as exc:  # unusable reply (ValidationError included): one more try
                parsed, usage, problem = None, LLMUsage(model=model_id), f"{type(exc).__name__}: {exc}"
            total += usage
            if parsed is not None:
                return parsed, total
        raise LLMError(
            f"no valid {schema.__name__} after {STRUCTURED_MAX_ATTEMPTS} attempts: {problem}"[:600],
            provider=provider,
            model=model_id,
        )

    async def achat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ChatTurnResponse:
        """``chat_with_tools`` in a worker thread, so an event loop (FastAPI SSE, MCP sessions) never blocks."""
        return await asyncio.to_thread(self.chat_with_tools, messages, tools, model, temperature, max_tokens)

    async def aparse_structured[T: BaseModel](
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[T, LLMUsage]:
        """``parse_structured`` in a worker thread."""
        return await asyncio.to_thread(self.parse_structured, messages, schema, model, temperature, max_tokens)

    # -- routing & SDK clients ---------------------------------------------------------------------------------------
    def _route(self, model: str | None) -> tuple[LLMProvider, str]:
        if model is None or model == self.model:
            return self.provider, self.model
        return resolve_provider(model), model

    def _api_key(self, provider: LLMProvider) -> str:
        secret = self.settings.openrouter_api_key if provider == "openrouter" else self.settings.anthropic_api_key
        if secret is None:
            var = EnvVar.OPENROUTER_API_KEY if provider == "openrouter" else EnvVar.ANTHROPIC_API_KEY
            raise LLMError(f"{var.value} is not set: add it to {ENV_FILE.name} (see .env.example)", provider=provider)
        return secret.get_secret_value()

    def _openrouter(self) -> openai.OpenAI:
        if self._openrouter_client is None:
            self._openrouter_client = openai.OpenAI(
                base_url=self.settings.openrouter_base_url,
                api_key=self._api_key("openrouter"),
                max_retries=self._max_retries,
                timeout=self._timeout_s,
            )
        return self._openrouter_client

    def _anthropic(self) -> anthropic.Anthropic:
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._api_key("anthropic"), max_retries=self._max_retries, timeout=self._timeout_s
            )
        return self._anthropic_client

    @staticmethod
    def _provider_error(provider: LLMProvider, model: str, exc: Exception) -> LLMError:
        return LLMError(f"{provider} call failed ({type(exc).__name__}): {exc}"[:600], provider=provider, model=model)

    # -- OpenRouter --------------------------------------------------------------------------------------------------
    @staticmethod
    def _openrouter_usage(response: Any, model: str, latency_s: float) -> LLMUsage:
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        reported = getattr(usage, "cost", None)  # OpenRouter extension of the OpenAI usage object
        cost = float(reported) if isinstance(reported, int | float) else estimate_cost_usd(model, tokens_in, tokens_out)
        return LLMUsage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, latency_s=latency_s, model=model)

    def _openrouter_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatTurnResponse:
        params: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = list(tools)
        started = time.perf_counter()
        response = self._openrouter().chat.completions.create(**params)
        usage = self._openrouter_usage(response, model, time.perf_counter() - started)
        if not response.choices:
            raise LLMError("OpenRouter returned no choices (empty reply)", provider="openrouter", model=model)
        choice = response.choices[0]
        calls = []
        for call in choice.message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is not None:
                args, args_error = _parse_arguments(function.arguments)
                calls.append(NormalizedToolCall(id=call.id, name=function.name, args=args, args_error=args_error))
        return ChatTurnResponse(
            content=choice.message.content or "",
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            assistant_message=choice.message.model_dump(exclude_none=True),  # keeps reasoning details, if any
        )

    def _openrouter_structured[T: BaseModel](
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[T | None, LLMUsage, str]:
        started = time.perf_counter()
        response = self._openrouter().chat.completions.parse(
            model=model,
            messages=list(messages),
            response_format=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = self._openrouter_usage(response, model, time.perf_counter() - started)
        if not response.choices:
            return None, usage, "empty reply (no choices)"
        message = response.choices[0].message
        if message.parsed is not None:
            return message.parsed, usage, ""
        if message.refusal:
            return None, usage, f"refused: {message.refusal}"
        return _validate_text(schema, message.content), usage, ""

    # -- native Anthropic --------------------------------------------------------------------------------------------
    def _anthropic_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        model: str,
        max_tokens: int,
        tool_choice: Mapping[str, Any] | None = None,
    ) -> ChatTurnResponse:
        system, turns = _to_anthropic_messages(messages)
        params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": turns}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = _to_anthropic_tools(tools)
        if tool_choice:
            params["tool_choice"] = dict(tool_choice)
        started = time.perf_counter()
        response = self._anthropic().messages.create(**params)
        latency_s = time.perf_counter() - started
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            NormalizedToolCall(
                id=block.id,
                name=block.name,
                args=dict(block.input) if isinstance(block.input, Mapping) else {},
                args_error=None if isinstance(block.input, Mapping) else "tool input is not a JSON object",
            )
            for block in response.content
            if block.type == "tool_use"
        ]
        raw = response.usage
        tokens_in = raw.input_tokens + (raw.cache_creation_input_tokens or 0) + (raw.cache_read_input_tokens or 0)
        usage = LLMUsage(
            tokens_in=tokens_in,
            tokens_out=raw.output_tokens,
            cost_usd=estimate_cost_usd(model, tokens_in, raw.output_tokens),
            latency_s=latency_s,
            model=model,
        )
        stop = response.stop_reason
        return ChatTurnResponse(
            content=text,
            tool_calls=calls,
            usage=usage,
            finish_reason=_FINISH_REASONS.get(stop, stop) if stop else None,
            assistant_message=_openai_assistant_message(text, calls),
        )

    def _anthropic_structured[T: BaseModel](
        self, messages: Sequence[Mapping[str, Any]], schema: type[T], model: str, max_tokens: int
    ) -> tuple[T | None, LLMUsage, str]:
        tool = function_tool(
            STRUCTURED_TOOL_NAME, f"Return the answer as one {schema.__name__} object.", schema.model_json_schema()
        )
        choice = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        turn = self._anthropic_chat(messages, [tool], model, max_tokens, tool_choice=choice)
        call = next((c for c in turn.tool_calls if c.name == STRUCTURED_TOOL_NAME), None)
        if call is None:
            return None, turn.usage, "the model did not call the structured-output tool"
        try:
            return schema.model_validate(call.args), turn.usage, ""
        except ValidationError as exc:
            return None, turn.usage, f"invalid {schema.__name__}: {exc.error_count()} validation error(s)"


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "PRICE_FAMILY_FALLBACKS",
    "PRICE_PER_MTOK",
    "ChatTurnResponse",
    "LLMClient",
    "LLMError",
    "LLMUsage",
    "NormalizedToolCall",
    "available_providers",
    "estimate_cost_usd",
    "function_tool",
    "resolve_provider",
    "tool_result_message",
]
