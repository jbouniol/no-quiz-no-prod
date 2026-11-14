"""Offline tests of the unified LLM client: provider routing, retries, cost accounting, format translation.

No network and no key: the SDK clients are either replaced by small fakes, or the real OpenAI SDK is driven through an
``httpx.MockTransport`` so its own retry loop runs against scripted HTTP answers.
"""

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest
from pydantic import BaseModel

from no_quiz_no_prod import llm as llm_module
from no_quiz_no_prod.contracts import RuntimeSettings
from no_quiz_no_prod.llm import (
    MAX_SDK_RETRIES,
    REQUEST_TIMEOUT_S,
    STRUCTURED_MAX_ATTEMPTS,
    LLMClient,
    LLMError,
    LLMUsage,
    _parse_arguments,
    _to_anthropic_messages,
    _to_anthropic_tools,
    estimate_cost_usd,
    function_tool,
    resolve_provider,
    tool_result_message,
)

SETTINGS = RuntimeSettings(openrouter_api_key="sk-or-test", anthropic_api_key="sk-ant-test")  # type: ignore[arg-type]
MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}]
TOOL = function_tool("read_code_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}})


class Verdict(BaseModel):
    accepted: bool
    reason: str


# ---- fakes ---------------------------------------------------------------------------------------------------------
def completion(
    content: str | None = "ok",
    *,
    tool_calls: list[tuple[str, str, str | None]] | None = None,
    usage: dict[str, Any] | None = None,
    finish_reason: str = "stop",
    parsed: Any = None,
    refusal: str | None = None,
) -> SimpleNamespace:
    """An OpenAI-SDK-shaped chat completion (only what the client reads)."""
    calls = [
        SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        for call_id, name, arguments in tool_calls or []
    ]
    message = SimpleNamespace(
        content=content,
        tool_calls=calls or None,
        parsed=parsed,
        refusal=refusal,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content, "reasoning": "kept"},
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(**(usage or {"prompt_tokens": 1000, "completion_tokens": 500})),
    )


class FakeCompletions:
    """Records every request and answers from a queue (an exception in the queue is raised)."""

    def __init__(self, *answers: Any) -> None:
        self.answers, self.requests = list(answers), []

    def _next(self, **params: Any) -> Any:
        self.requests.append(params)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    create = parse = _next


def openrouter_client(*answers: Any, model: str | None = None) -> tuple[LLMClient, FakeCompletions]:
    client, completions = LLMClient(model=model, settings=SETTINGS), FakeCompletions(*answers)
    client._openrouter_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return client, completions


def anthropic_response(*blocks: SimpleNamespace, stop_reason: str = "end_turn", **usage: int) -> SimpleNamespace:
    counts = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=SimpleNamespace(**(counts | usage)))


def anthropic_client(*answers: Any, model: str = "claude-sonnet-4-5") -> tuple[LLMClient, FakeCompletions]:
    client, messages = LLMClient(model=model, settings=SETTINGS), FakeCompletions(*answers)
    client._anthropic_client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    return client, messages


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(block_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


# ---- provider routing ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "explicit", "expected"),
    [
        ("claude-sonnet-4-5", None, "anthropic"),
        ("~claude-haiku-4-5", None, "anthropic"),
        ("anthropic/claude-sonnet-4.5", None, "openrouter"),  # a slash is an OpenRouter id, even for Claude
        ("~deepseek/deepseek-v4-flash-latest", None, "openrouter"),
        ("openai/gpt-4o-mini", None, "openrouter"),
        (None, None, "openrouter"),
        ("openai/gpt-4o-mini", "anthropic", "anthropic"),  # an explicit provider always wins
    ],
)
def test_provider_is_resolved_from_the_model_id(model: str | None, explicit: Any, expected: str) -> None:
    assert resolve_provider(model, explicit) == expected


def test_default_models_follow_the_provider_and_the_settings() -> None:
    settings = SETTINGS.model_copy(update={"openrouter_model": "x/y", "anthropic_model": "claude-z"})
    assert LLMClient(settings=settings).model == "x/y"
    assert LLMClient(provider="anthropic", settings=settings).model == "claude-z"
    assert LLMClient(settings=settings).available_providers() == {"openrouter": True, "anthropic": True}


def test_a_per_call_model_reroutes_that_call_only() -> None:
    client, completions = openrouter_client(completion("from openrouter"), completion("again"))
    client._anthropic_client = SimpleNamespace(messages=FakeCompletions(anthropic_response(text_block("from claude"))))  # type: ignore[assignment]

    assert client.chat_with_tools(MESSAGES).content == "from openrouter"
    rerouted = client.chat_with_tools(MESSAGES, model="claude-haiku-4-5")
    assert rerouted.content == "from claude" and rerouted.usage.model == "claude-haiku-4-5"
    assert client.chat_with_tools(MESSAGES).content == "again"  # the default route is untouched
    assert [request["model"] for request in completions.requests] == [client.model, client.model]
    assert client._anthropic_client.messages.requests[0]["model"] == "claude-haiku-4-5"  # type: ignore[union-attr]


@pytest.mark.parametrize(("model", "variable"), [("x/y", "OPENROUTER_API_KEY"), ("claude-z", "ANTHROPIC_API_KEY")])
def test_a_missing_key_is_an_llm_error_that_names_the_variable_and_never_calls_out(model: str, variable: str) -> None:
    client = LLMClient(model=model, settings=RuntimeSettings())
    with pytest.raises(LLMError, match=variable) as raised:
        client.chat_with_tools(MESSAGES)
    assert raised.value.provider == resolve_provider(model)


# ---- retries and timeouts ------------------------------------------------------------------------------------------
def test_sdk_clients_are_built_with_the_retry_and_timeout_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: built.setdefault("openrouter", kwargs))
    monkeypatch.setattr(llm_module.anthropic, "Anthropic", lambda **kwargs: built.setdefault("anthropic", kwargs))
    client = LLMClient(settings=SETTINGS)
    assert client._openrouter() is client._openrouter()  # built once, reused
    client._anthropic()
    for provider, key in (("openrouter", "sk-or-test"), ("anthropic", "sk-ant-test")):
        assert built[provider]["max_retries"] == MAX_SDK_RETRIES and built[provider]["timeout"] == REQUEST_TIMEOUT_S
        assert built[provider]["api_key"] == key
    assert built["openrouter"]["base_url"] == SETTINGS.openrouter_base_url
    custom = LLMClient(settings=SETTINGS, timeout_s=5.0, max_retries=0)
    monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: kwargs)
    assert custom._openrouter()["max_retries"] == 0 and custom._openrouter()["timeout"] == 5.0  # type: ignore[index]


def scripted_transport(*statuses: int) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """HTTP answers for the real OpenAI SDK: the given statuses in order, then 200 with a completion."""
    queue, seen = list(statuses), []
    body = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "x/y",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "recovered"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00042},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if queue:
            return httpx.Response(queue.pop(0), headers={"retry-after-ms": "1"}, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), seen


def real_sdk_client(transport: httpx.MockTransport, max_retries: int) -> LLMClient:
    client = LLMClient(model="x/y", settings=SETTINGS, max_retries=max_retries)
    client._openrouter_client = openai.OpenAI(
        base_url="http://openrouter.invalid/api/v1",
        api_key="sk-or-test",
        max_retries=max_retries,
        http_client=httpx.Client(transport=transport),
    )
    return client


def test_transient_provider_failures_are_retried_by_the_sdk_then_succeed() -> None:
    transport, seen = scripted_transport(429, 503)
    turn = real_sdk_client(transport, max_retries=2).chat_with_tools(MESSAGES, [TOOL], max_tokens=64)
    assert turn.content == "recovered" and len(seen) == 3  # two retries, same request
    assert turn.usage.cost_usd == pytest.approx(0.00042) and turn.usage.tokens_in == 10
    sent = [json.loads(request.content) for request in seen]
    assert sent[0] == sent[2] and sent[0]["max_tokens"] == 64 and sent[0]["tools"] == [TOOL]


def test_exhausted_retries_and_non_retryable_failures_become_one_llm_error() -> None:
    transport, seen = scripted_transport(429, 429, 429, 429)
    with pytest.raises(LLMError, match="openrouter call failed") as raised:
        real_sdk_client(transport, max_retries=1).chat_with_tools(MESSAGES)
    assert len(seen) == 2 and raised.value.provider == "openrouter" and raised.value.model == "x/y"
    assert isinstance(raised.value.__cause__, openai.RateLimitError)

    transport, seen = scripted_transport(401)
    with pytest.raises(LLMError, match="AuthenticationError"):
        real_sdk_client(transport, max_retries=3).chat_with_tools(MESSAGES)
    assert len(seen) == 1  # an auth failure is never retried


def test_anthropic_sdk_errors_and_empty_replies_are_llm_errors() -> None:
    request = httpx.Request("POST", "http://anthropic.invalid/v1/messages")
    client, _ = anthropic_client(anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError, match="anthropic call failed") as raised:
        client.chat_with_tools(MESSAGES)
    assert raised.value.provider == "anthropic"
    empty, _ = openrouter_client(SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMError, match="no choices"):
        empty.chat_with_tools(MESSAGES)


# ---- cost and token accounting -------------------------------------------------------------------------------------
def test_cost_is_estimated_from_the_price_table_with_family_fallbacks() -> None:
    assert estimate_cost_usd("claude-sonnet-4-5", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert estimate_cost_usd("claude-opus-9-preview", 2_000_000, 0) == pytest.approx(10.0)  # family fallback
    assert estimate_cost_usd("~deepseek/deepseek-v4-flash-latest", 1_000_000, 1_000_000) == pytest.approx(0.21)
    assert estimate_cost_usd("vendor/unknown-model", 10**9, 10**9) == 0.0


def test_openrouter_reported_cost_wins_and_the_estimate_is_only_a_fallback() -> None:
    reported = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.0123}
    client, _ = openrouter_client(completion(usage=reported), completion(), model="~deepseek/deepseek-v4-flash-latest")
    charged, estimated = client.chat_with_tools(MESSAGES).usage, client.chat_with_tools(MESSAGES).usage
    assert charged.cost_usd == pytest.approx(0.0123) and (charged.tokens_in, charged.tokens_out) == (1000, 500)
    assert estimated.cost_usd == pytest.approx((1000 * 0.05 + 500 * 0.16) / 1_000_000)
    assert charged.latency_s >= 0 and charged.model == "~deepseek/deepseek-v4-flash-latest"
    zero, _ = openrouter_client(completion(usage={"prompt_tokens": None, "completion_tokens": None}))
    assert zero.chat_with_tools(MESSAGES).usage.tokens_in == 0  # a provider that omits usage does not crash


def test_anthropic_usage_counts_cache_tokens_as_input() -> None:
    client, _ = anthropic_client(
        anthropic_response(text_block("hi"), cache_creation_input_tokens=300, cache_read_input_tokens=600)
    )
    usage = client.chat_with_tools(MESSAGES).usage
    assert (usage.tokens_in, usage.tokens_out) == (1000, 50)
    assert usage.cost_usd == pytest.approx((1000 * 3.0 + 50 * 15.0) / 1_000_000)


def test_usages_add_up_across_calls_and_models() -> None:
    first = LLMUsage(tokens_in=10, tokens_out=1, cost_usd=0.1, latency_s=1.0, model="a")
    second = LLMUsage(tokens_in=5, tokens_out=2, cost_usd=0.2, latency_s=0.5, model="b")
    total = sum([first, second, first], LLMUsage())
    assert (total.tokens_in, total.tokens_out, total.model) == (25, 4, "a+b")
    assert total.cost_usd == pytest.approx(0.4) and total.latency_s == pytest.approx(2.5)


# ---- tool calls and format translation -----------------------------------------------------------------------------
def test_openrouter_tool_calls_are_normalized_and_bad_arguments_are_reported_not_raised() -> None:
    calls = [
        ("c1", "read_code_file", '{"path": "orders.py"}'),
        ("c2", "read_code_file", "{not json"),
        ("c3", "x", "[1]"),
    ]
    client, completions = openrouter_client(completion(None, tool_calls=calls, finish_reason="tool_calls"))
    turn = client.chat_with_tools(MESSAGES, [TOOL], temperature=0.3, max_tokens=77)
    assert turn.content == "" and turn.finish_reason == "tool_calls"
    good, broken, not_object = turn.tool_calls
    assert good.args == {"path": "orders.py"} and good.args_error is None
    assert broken.args == {} and "not valid JSON" in (broken.args_error or "")
    assert not_object.args == {} and not_object.args_error == "arguments must be a JSON object"
    assert turn.assistant_message["reasoning"] == "kept"  # provider extras are preserved for the next turn
    request = completions.requests[0]
    assert request["temperature"] == 0.3 and request["max_tokens"] == 77 and request["tools"] == [TOOL]
    assert (
        "tools"
        not in openrouter_client(completion())[0]._openrouter_chat(MESSAGES, None, "m", 0.0, 1).assistant_message
    )
    assert _parse_arguments("  ") == ({}, None)


def test_openai_history_is_translated_to_alternating_anthropic_turns() -> None:
    history = [
        {"role": "system", "content": "Rules."},
        {"role": "developer", "content": "More rules."},
        {"role": "user", "content": "Fix it"},
        {"role": "user", "content": [{"type": "text", "text": "please"}, {"type": "image_url", "image_url": {}}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call.1:a",
                    "type": "function",
                    "function": {"name": "read_code_file", "arguments": '{"path": "a"}'},
                },
                {"id": "call 2", "type": "function", "function": {"name": "read_code_file", "arguments": "oops"}},
            ],
        },
        tool_result_message("call.1:a", {"content": "x = 1"}),
        tool_result_message("call 2", ""),
        {"role": "user", "content": "thanks"},
    ]
    system, turns = _to_anthropic_messages(history)
    assert system == "Rules.\n\nMore rules." and [turn["role"] for turn in turns] == ["user", "assistant", "user"]
    assert [block["text"] for block in turns[0]["content"]] == ["Fix it", "please"]  # consecutive users merged
    uses = turns[1]["content"]
    assert [use["id"] for use in uses] == ["call_1_a", "call_2"] and uses[0]["input"] == {"path": "a"}
    assert uses[1]["input"] == {}  # unparsable arguments never reach the API as a string
    results = turns[2]["content"]
    assert [block.get("tool_use_id") for block in results] == ["call_1_a", "call_2", None]
    assert results[0]["content"] == '{"content": "x = 1"}' and results[1]["content"] == "(empty result)"
    assert _to_anthropic_tools([TOOL, {"name": "bare"}]) == [
        {"name": "read_code_file", "description": "Read a file.", "input_schema": TOOL["function"]["parameters"]},
        {"name": "bare", "description": "", "input_schema": {"type": "object", "properties": {}}},
    ]


def test_anthropic_turn_is_normalized_to_the_openai_vocabulary() -> None:
    blocks = (
        text_block("Reading."),
        tool_use("toolu_1", "read_code_file", {"path": "auth.py"}),
        tool_use("toolu_2", "x", "bad"),
    )
    client, messages = anthropic_client(anthropic_response(*blocks, stop_reason="tool_use"))
    turn = client.chat_with_tools(MESSAGES, [TOOL], max_tokens=123)
    assert turn.content == "Reading." and turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].args == {"path": "auth.py"} and turn.tool_calls[1].args_error is not None
    replayed = turn.assistant_message["tool_calls"][0]
    assert replayed["id"] == "toolu_1" and json.loads(replayed["function"]["arguments"]) == {"path": "auth.py"}
    request = messages.requests[0]
    assert request["system"] == "You are terse." and request["max_tokens"] == 123 and "temperature" not in request
    assert request["tools"][0]["name"] == "read_code_file" and "tool_choice" not in request
    truncated, _ = anthropic_client(anthropic_response(stop_reason="max_tokens"))
    assert truncated.chat_with_tools(MESSAGES).finish_reason == "length"


# ---- structured output ---------------------------------------------------------------------------------------------
def test_structured_output_tolerates_fenced_json_and_sums_the_usage_of_every_attempt() -> None:
    fenced = 'Sure:\n```json\n{"accepted": true, "reason": "clear"}\n```'
    client, completions = openrouter_client(completion("not json at all"), completion(fenced), model="x/y")
    verdict, usage = client.parse_structured(MESSAGES, Verdict)
    assert verdict == Verdict(accepted=True, reason="clear")
    assert (
        len(completions.requests) == 2 == STRUCTURED_MAX_ATTEMPTS
        and completions.requests[0]["response_format"] is Verdict
    )
    assert usage.tokens_in == 1000  # the failed attempt raised before its usage was read: only the good one counts
    parsed, _ = openrouter_client(completion(None, parsed=Verdict(accepted=False, reason="sdk parsed")))
    assert parsed.parse_structured(MESSAGES, Verdict)[0].reason == "sdk parsed"


def test_structured_output_gives_up_with_an_llm_error_after_the_allowed_attempts() -> None:
    client, completions = openrouter_client(completion(None, refusal="cannot"), completion("{}"), completion("unused"))
    with pytest.raises(LLMError, match=f"no valid Verdict after {STRUCTURED_MAX_ATTEMPTS} attempts"):
        client.parse_structured(MESSAGES, Verdict)
    assert len(completions.requests) == STRUCTURED_MAX_ATTEMPTS  # bounded: the third answer is never asked for


def test_structured_output_on_anthropic_forces_one_tool_and_retries_an_invalid_input() -> None:
    bad = anthropic_response(tool_use("t1", llm_module.STRUCTURED_TOOL_NAME, {"accepted": "maybe"}))
    good = anthropic_response(tool_use("t2", llm_module.STRUCTURED_TOOL_NAME, {"accepted": True, "reason": "ok"}))
    client, messages = anthropic_client(bad, good)
    verdict, usage = client.parse_structured(MESSAGES, Verdict, max_tokens=50)
    assert verdict.accepted and usage.tokens_in == 200 and usage.tokens_out == 100  # both attempts are paid for
    request = messages.requests[0]
    assert request["tool_choice"] == {"type": "tool", "name": llm_module.STRUCTURED_TOOL_NAME}
    assert request["tools"][0]["input_schema"] == Verdict.model_json_schema()
    silent, _ = anthropic_client(
        anthropic_response(text_block("no tool")), anthropic_response(text_block("still none"))
    )
    with pytest.raises(LLMError, match="did not call the structured-output tool"):
        silent.parse_structured(MESSAGES, Verdict)
