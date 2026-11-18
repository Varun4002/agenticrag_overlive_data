"""Embedder tests: fake determinism, batching, retries, token accounting.

Everything here runs offline against ``httpx.MockTransport``. The one live
provider check is manual: ``tools/smoke_embeddings.py`` (see the indexing
HANDOFF).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from config import Settings
from indexing.embedder import (
    MAX_BATCH_SIZE,
    DeterministicFakeEmbedder,
    Embedder,
    EmbeddingResult,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from indexing.errors import (
    EmbeddingAuthError,
    EmbeddingBadRequestError,
    EmbeddingDimensionError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingServerError,
    EmbeddingTimeoutError,
    IndexingConfigError,
)

DIMENSIONS = 8


def vector(seed: float) -> list[float]:
    return [seed + index / 100 for index in range(DIMENSIONS)]


def ok_response(inputs: list[str], tokens: int = 7) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "data": [
                {"index": index, "embedding": vector(index)} for index in range(len(inputs))
            ],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        },
    )


class Recorder:
    """Captured requests, batches, and backoff sleeps."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def make_embedder(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    recorder: Recorder | None = None,
    **kwargs: Any,
) -> OpenAICompatibleEmbedder:
    recorder = recorder or Recorder()
    return OpenAICompatibleEmbedder(
        base_url="http://test.local/v1",
        api_key="not-a-real-key",
        model="test-model",
        dimensions=DIMENSIONS,
        max_batch_size=kwargs.pop("max_batch_size", MAX_BATCH_SIZE),
        transport=httpx.MockTransport(handler),
        sleep=recorder.sleep,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Deterministic fake
# ---------------------------------------------------------------------------


async def test_fake_embedder_is_deterministic_and_unit_length() -> None:
    first = DeterministicFakeEmbedder(dimensions=DIMENSIONS)
    second = DeterministicFakeEmbedder(dimensions=DIMENSIONS)
    result = await first.embed(["alpha", "beta"])
    again = await second.embed(["alpha", "beta"])

    assert result.vectors == again.vectors
    assert result.vectors[0] != result.vectors[1]
    for embedding in result.vectors:
        assert len(embedding) == DIMENSIONS
        assert abs(sum(value * value for value in embedding) - 1.0) < 1e-9


async def test_fake_embedder_batches_and_counts_tokens() -> None:
    embedder = DeterministicFakeEmbedder(dimensions=DIMENSIONS, max_batch_size=2)
    result = await embedder.embed(["a", "b", "c", "d", "e"])

    assert embedder.calls == [2, 2, 1]
    assert result.batches == 3
    assert len(result.vectors) == 5
    assert result.tokens > 0


async def test_fake_embedder_default_dimension_matches_schema() -> None:
    from db.models import EMBEDDING_DIMENSIONS

    assert DeterministicFakeEmbedder().dimensions == EMBEDDING_DIMENSIONS


async def test_fake_embedder_satisfies_protocol() -> None:
    embedder = DeterministicFakeEmbedder(dimensions=DIMENSIONS)
    assert isinstance(embedder, Embedder)
    assert await embedder.embed([]) == EmbeddingResult(vectors=[], tokens=0, batches=0)


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


async def test_request_shape_and_response_parsing() -> None:
    recorder = Recorder()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer not-a-real-key"
        payload = json.loads(request.content)
        seen.append(payload)
        return ok_response(payload["input"], tokens=11)

    embedder = make_embedder(handler, recorder=recorder)
    result = await embedder.embed(["one", "two"])

    assert seen == [{"model": "test-model", "input": ["one", "two"]}]
    assert "dimensions" not in seen[0], "dimensions is opt-in for gateway compatibility"
    assert result.vectors == [vector(0), vector(1)]
    assert result.tokens == 11
    assert result.batches == 1
    await embedder.aclose()


async def test_send_dimensions_flag_includes_dimensions() -> None:
    seen: list[dict[str, Any]] = []
    embedder = make_embedder(
        lambda request: (
            seen.append(json.loads(request.content)) or ok_response(["x"])
        ),
        send_dimensions=True,
    )
    await embedder.embed(["x"])
    assert seen[0]["dimensions"] == DIMENSIONS


async def test_results_are_ordered_by_index_not_response_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert len(payload["input"]) == 3
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": vector(2)},
                    {"index": 0, "embedding": vector(0)},
                    {"index": 1, "embedding": vector(1)},
                ],
                "usage": {"prompt_tokens": 3},
            },
        )

    embedder = make_embedder(handler)
    result = await embedder.embed(["a", "b", "c"])
    assert result.vectors == [vector(0), vector(1), vector(2)]


async def test_large_input_is_split_into_batches_of_max_batch_size() -> None:
    recorder = Recorder()
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sizes.append(len(payload["input"]))
        return ok_response(payload["input"], tokens=5)

    embedder = make_embedder(handler, recorder=recorder, max_batch_size=4)
    result = await embedder.embed([f"text-{index}" for index in range(10)])

    assert sizes == [4, 4, 2]
    assert result.batches == 3
    assert result.tokens == 15
    assert len(result.vectors) == 10


async def test_usage_falls_back_to_estimate_when_provider_omits_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": vector(index)}
                    for index in range(len(payload["input"]))
                ]
            },
        )

    embedder = make_embedder(handler)
    result = await embedder.embed(["one two three", "four five"])
    assert result.tokens == 5


async def test_retries_server_errors_then_succeeds() -> None:
    recorder = Recorder()
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503, text="upstream unavailable")
        return ok_response(["x"], tokens=2)

    embedder = make_embedder(handler, recorder=recorder)
    result = await embedder.embed(["x"])

    assert attempts["count"] == 3
    assert result.tokens == 2
    assert len(recorder.sleeps) == 2
    assert all(0.0 <= delay <= 20.0 for delay in recorder.sleeps)


async def test_exhausted_retries_raise_server_error() -> None:
    embedder = make_embedder(lambda request: httpx.Response(500, text="boom"), max_retries=1)
    with pytest.raises(EmbeddingServerError, match="HTTP 500"):
        await embedder.embed(["x"])


async def test_auth_errors_are_terminal() -> None:
    recorder = Recorder()
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, text="bad key")

    embedder = make_embedder(handler, recorder=recorder)
    with pytest.raises(EmbeddingAuthError, match="credentials"):
        await embedder.embed(["x"])
    assert calls["count"] == 1
    assert recorder.sleeps == []


async def test_bad_request_is_terminal() -> None:
    embedder = make_embedder(lambda request: httpx.Response(400, text="nope"))
    with pytest.raises(EmbeddingBadRequestError, match="HTTP 400"):
        await embedder.embed(["x"])


async def test_rate_limit_honours_retry_after() -> None:
    recorder = Recorder()
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, text="slow down")
        return ok_response(["x"])

    embedder = make_embedder(handler, recorder=recorder)
    await embedder.embed(["x"])
    assert recorder.sleeps == [0.25]


async def test_rate_limit_error_exposes_retry_after_when_exhausted() -> None:
    embedder = make_embedder(
        lambda request: httpx.Response(429, headers={"Retry-After": "3"}, text="slow down"),
        max_retries=0,
    )
    with pytest.raises(EmbeddingRateLimitError) as excinfo:
        await embedder.embed(["x"])
    assert excinfo.value.retry_after_s == 3.0
    assert excinfo.value.retryable is True


async def test_timeouts_are_retried() -> None:
    recorder = Recorder()
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return ok_response(["x"])

    embedder = make_embedder(handler, recorder=recorder)
    result = await embedder.embed(["x"])
    assert calls["count"] == 2
    assert len(result.vectors) == 1
    assert len(recorder.sleeps) == 1


async def test_timeout_exhaustion_raises_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("nope", request=request)

    embedder = make_embedder(handler, max_retries=1)
    with pytest.raises(EmbeddingTimeoutError):
        await embedder.embed(["x"])


async def test_dimension_mismatch_raises_configuration_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 1},
            },
        )

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingDimensionError, match="3-dimension vectors"):
        await embedder.embed(["x"])


async def test_malformed_payloads_raise_response_errors() -> None:
    cases: list[Any] = [
        {"data": [], "usage": {}},
        {"data": [{"index": 0}], "usage": {}},
        {"data": [{"index": 0, "embedding": ["x"] * DIMENSIONS}], "usage": {}},
        {"data": [{"index": 0, "embedding": vector(0)}, {"index": 1, "embedding": vector(1)}]},
        [1, 2, 3],
    ]
    for payload in cases:
        embedder = make_embedder(lambda request, payload=payload: httpx.Response(200, json=payload))
        with pytest.raises(EmbeddingResponseError):
            await embedder.embed(["only-one-input"])


async def test_invalid_json_raises_response_error_without_retry() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, content=b"not json")

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingResponseError, match="invalid JSON"):
        await embedder.embed(["x"])
    assert calls["count"] == 1


async def test_empty_input_makes_no_request() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        calls["count"] += 1
        return ok_response(["x"])

    embedder = make_embedder(handler)
    assert await embedder.embed([]) == EmbeddingResult(vectors=[], tokens=0, batches=0)
    assert calls["count"] == 0


async def test_lifetime_token_counter_accumulates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_response(json.loads(request.content)["input"], tokens=4)

    embedder = make_embedder(handler)
    await embedder.embed(["a"])
    await embedder.embed(["b", "c"])
    # The fake provider reports 4 tokens per request, so two calls accumulate 8.
    assert embedder.tokens_used == 8


def test_max_batch_size_is_capped_at_128() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        make_embedder(lambda request: ok_response(["x"]), max_batch_size=MAX_BATCH_SIZE + 1)
    with pytest.raises(ValueError, match="max_batch_size"):
        make_embedder(lambda request: ok_response(["x"]), max_batch_size=0)


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------


def test_build_embedder_fake_uses_configured_dimensions() -> None:
    embedder = build_embedder(Settings(embedding_dimensions=DIMENSIONS), kind="fake")
    assert isinstance(embedder, DeterministicFakeEmbedder)
    assert embedder.dimensions == DIMENSIONS


def test_build_embedder_auto_works_without_a_key_for_local_providers() -> None:
    """Ollama needs no API key; the local-first default must not demand one."""
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key=None,
        embedding_model="nomic-embed-text",
        embedding_dimensions=768,
    )
    embedder = build_embedder(settings, kind="auto")
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.dimensions == 768


def test_build_embedder_rejects_empty_base_url() -> None:
    with pytest.raises(IndexingConfigError, match="LLM_BASE_URL"):
        build_embedder(Settings(llm_base_url=""), kind="auto")


def test_build_embedder_openai_uses_settings() -> None:
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3.5:4b",
        embedding_model="nomic-embed-text",
        embedding_dimensions=768,
        llm_api_key="local-placeholder",
    )
    embedder = build_embedder(settings, kind="openai")
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model == "nomic-embed-text"
    assert embedder.dimensions == 768


def test_build_embedder_rejects_unknown_kind() -> None:
    with pytest.raises(IndexingConfigError, match="unknown embedder kind"):
        build_embedder(Settings(), kind="cohere")
