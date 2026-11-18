"""LLM client tests using ``httpx.MockTransport`` — no network, no API key."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from llm.base import CompletionRequest, Message, Role, TokenLedger
from llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from llm.openai_compatible import OpenAICompatibleClient

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def make_client(
    handler: Handler,
    *,
    ledger: TokenLedger | None = None,
    max_retries: int = 2,
    backoff_base_s: float = 0.01,
) -> OpenAICompatibleClient:
    """Client with a fake transport and an instant, recorded sleep."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = OpenAICompatibleClient(
        base_url="https://fake.local/v1",
        api_key="test-key",
        default_model="test-model",
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        ledger=ledger,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
    )
    # Expose the recorded sleeps for assertions without widening the public API.
    client.sleeps = sleeps  # type: ignore[attr-defined]
    return client


def ok_response(text: str = "hello", *, usage: bool = True) -> httpx.Response:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return httpx.Response(200, json=body)


def request() -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.user, content="hi")], max_tokens=64, temperature=0.0
    )


async def test_complete_parses_text_and_usage() -> None:
    seen: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return ok_response("cited answer")

    client = make_client(handler)
    try:
        result = await client.complete(request())
    finally:
        await client.aclose()

    assert result.text == "cited answer"
    assert result.usage.total_tokens == 18
    assert result.finish_reason == "stop"
    assert result.provider_request_id == "chatcmpl-1"
    assert result.latency_ms >= 0

    sent = seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://fake.local/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(sent.content)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 64


async def test_token_ledger_accumulates_across_calls() -> None:
    ledger = TokenLedger()

    async def handler(_req: httpx.Request) -> httpx.Response:
        return ok_response()

    client = make_client(handler, ledger=ledger)
    try:
        await client.complete(request())
        await client.complete(request())
    finally:
        await client.aclose()

    assert ledger.calls == 2
    assert ledger.prompt_tokens == 22
    assert ledger.completion_tokens == 14
    assert ledger.total_tokens == 36
    assert ledger.as_payload()["per_model"] == {"test-model": 36}


async def test_missing_usage_block_defaults_to_zero() -> None:
    async def handler(_req: httpx.Request) -> httpx.Response:
        return ok_response(usage=False)

    client = make_client(handler)
    try:
        result = await client.complete(request())
    finally:
        await client.aclose()

    assert result.usage.total_tokens == 0
    assert client.ledger.total_tokens == 0


async def test_retries_server_error_then_succeeds() -> None:
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="overloaded")
        return ok_response("second try")

    client = make_client(handler)
    try:
        result = await client.complete(request())
    finally:
        await client.aclose()

    assert result.text == "second try"
    assert calls == 2


async def test_gives_up_after_max_retries() -> None:
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    client = make_client(handler, max_retries=2)
    try:
        with pytest.raises(LLMServerError):
            await client.complete(request())
    finally:
        await client.aclose()

    assert calls == 3  # first attempt + 2 retries


async def test_rate_limit_honours_retry_after() -> None:
    async def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "0.25"})

    client = make_client(handler, max_retries=1)
    try:
        with pytest.raises(LLMRateLimitError) as excinfo:
            await client.complete(request())
    finally:
        await client.aclose()

    assert excinfo.value.retry_after_s == 0.25
    assert client.sleeps == [0.25]  # type: ignore[attr-defined]


async def test_timeout_is_retried_then_raised() -> None:
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=_req)

    client = make_client(handler, max_retries=1)
    try:
        with pytest.raises(LLMTimeoutError):
            await client.complete(request())
    finally:
        await client.aclose()

    assert calls == 2
    assert len(client.sleeps) == 1  # type: ignore[attr-defined]


async def test_auth_error_is_not_retried() -> None:
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    client = make_client(handler, max_retries=3)
    try:
        with pytest.raises(LLMAuthError):
            await client.complete(request())
    finally:
        await client.aclose()

    assert calls == 1


async def test_bad_request_is_not_retried() -> None:
    calls = 0

    async def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad shape")

    client = make_client(handler, max_retries=3)
    try:
        with pytest.raises(LLMBadRequestError):
            await client.complete(request())
    finally:
        await client.aclose()

    assert calls == 1


async def test_malformed_response_raises_response_error() -> None:
    async def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "choices": []})

    client = make_client(handler)
    try:
        with pytest.raises(LLMResponseError):
            await client.complete(request())
    finally:
        await client.aclose()


async def test_invalid_json_raises_response_error() -> None:
    async def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = make_client(handler)
    try:
        with pytest.raises(LLMResponseError):
            await client.complete(request())
    finally:
        await client.aclose()


async def test_per_request_model_override_wins() -> None:
    captured: dict[str, object] = {}

    async def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return ok_response()

    client = make_client(handler)
    try:
        await client.complete(
            CompletionRequest(
                messages=[Message(role=Role.system, content="be terse")],
                model="override-model",
                extra={"response_format": {"type": "json_object"}},
            )
        )
    finally:
        await client.aclose()

    assert captured["model"] == "override-model"
    # ``extra`` is merged flat into the JSON body (not nested), so the provider
    # sees ``response_format`` exactly as given.
    assert captured["response_format"] == {"type": "json_object"}
    assert "extra" not in captured
