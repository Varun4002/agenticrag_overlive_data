"""Embedding interface and implementations.

One interface (:class:`Embedder`), two implementations:

``DeterministicFakeEmbedder``
    Hash-based vectors. Stable across processes and runs, configurable
    dimension, and it records every batch it receives. Tests, fixture-corpus
    indexing, and offline eval runs use it so no API key is ever required.

``OpenAICompatibleEmbedder``
    Async ``httpx`` client for ``POST {base_url}/embeddings`` (OpenAI and any
    gateway that copies the shape). Adds request batching (<= ``max_batch_size``,
    128 by default, under OpenAI's limit of 2048 but conservative for gateways),
    connect/read timeouts, bounded exponential backoff with jitter,
    ``Retry-After`` support, response validation, and token accounting.

Callers depend only on the protocol, so the indexer is embedder-agnostic and
tests never touch the network.

Model and dimension come from settings (``EMBEDDING_MODEL`` /
``EMBEDDING_DIMENSIONS``); see :func:`build_embedder`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from config import Settings, get_settings
from indexing.errors import (
    EmbeddingAuthError,
    EmbeddingBadRequestError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingServerError,
    EmbeddingTimeoutError,
    IndexingConfigError,
)
from indexing.tokens import estimate_tokens

logger = logging.getLogger(__name__)

#: Hard ceiling enforced per HTTP request, independent of provider limits.
MAX_BATCH_SIZE = 128

# Statuses worth retrying: request timeout/conflict/too-early, rate limiting, 5xx.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors in input order plus what the call cost."""

    vectors: list[list[float]]
    tokens: int
    batches: int

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")


@runtime_checkable
class Embedder(Protocol):
    """The only embedding interface the indexer and retrieval agent may use."""

    @property
    def model(self) -> str:
        """Provider model identifier, recorded in traces."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector size; must match ``core.chunks.embedding``."""
        ...

    @property
    def max_batch_size(self) -> int:
        """Largest number of inputs one ``embed`` call may send per request."""
        ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed ``texts`` in order, splitting into provider-sized batches.

        Raises an :class:`~indexing.errors.EmbeddingError` subclass when a batch
        fails after retries. Callers that need partial progress should pass at
        most ``max_batch_size`` texts per call (the indexer does exactly that,
        so one failed batch never costs the whole document).
        """
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


# ---------------------------------------------------------------------------
# Deterministic fake (tests, offline runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeterministicFakeEmbedder:
    """Hash-derived, L2-normalized vectors. No network, fully reproducible.

    Determinism is by construction: the vector bytes come from
    ``sha256(text + counter)``, so the same text yields the same vector on any
    machine and in any process. Vectors are unit length, which keeps cosine
    distance and dot product consistent for retrieval tests.
    """

    # Matches core.chunks.embedding (migration 0002, 768 dims). The indexer
    # refuses any embedder whose dimension differs from the schema.
    dimensions: int = 768
    model: str = "deterministic-fake"
    max_batch_size: int = MAX_BATCH_SIZE
    #: Sizes of the batches received, in order. Assertions use this.
    calls: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        batches = 0
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self.max_batch_size):
            batch = list(texts[start : start + self.max_batch_size])
            self.calls.append(len(batch))
            vectors.extend(self._vector(text) for text in batch)
            tokens += sum(estimate_tokens(text) for text in batch)
            batches += 1
        return EmbeddingResult(vectors=vectors, tokens=tokens, batches=batches)

    async def aclose(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        needed = self.dimensions * 4
        buffer = bytearray()
        counter = 0
        while len(buffer) < needed:
            buffer.extend(hashlib.sha256(f"{text}\x00{counter}".encode()).digest())
            counter += 1
        raw = bytes(buffer[:needed])
        values = [
            int.from_bytes(raw[offset : offset + 4], "big") / 2**31 - 1.0
            for offset in range(0, needed, 4)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:  # pragma: no cover - astronomically unlikely
            values[0] = 1.0
            norm = 1.0
        return [value / norm for value in values]


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP embedder
# ---------------------------------------------------------------------------


class OpenAICompatibleEmbedder:
    """Async client for OpenAI-style embedding endpoints."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dimensions: int,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 20.0,
        max_batch_size: int = MAX_BATCH_SIZE,
        send_dimensions: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not 1 <= max_batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be in [1, {MAX_BATCH_SIZE}]")
        self.model = model
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        #: Only send ``dimensions`` when the provider supports it (OpenAI
        #: embedding v3 does; several gateways reject unknown fields).
        self._send_dimensions = send_dimensions
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._client: httpx.AsyncClient | None = None
        #: Lifetime token count, for traces and the backfill report.
        self.tokens_used = 0

    # --- transport lifecycle -------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s),
                headers=headers,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenAICompatibleEmbedder:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- request building ----------------------------------------------
    def _payload(self, batch: Sequence[str]) -> dict[str, object]:
        payload: dict[str, object] = {"model": self.model, "input": list(batch)}
        if self._send_dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def _backoff_delay(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff with full jitter, bounded by ``backoff_max_s``."""
        if retry_after_s is not None:
            return min(retry_after_s, self._backoff_max_s)
        ceiling = min(self._backoff_base_s * (2**attempt), self._backoff_max_s)
        return random.uniform(0.0, ceiling)

    # --- error mapping --------------------------------------------------
    def _error_for_status(self, response: httpx.Response) -> EmbeddingError:
        status = response.status_code
        detail = (response.text or "")[:500]
        if status in {401, 403}:
            return EmbeddingAuthError(
                f"provider rejected credentials (HTTP {status}): {detail}",
                provider=self.provider,
            )
        if status == 429:
            return EmbeddingRateLimitError(
                f"rate limited by provider: {detail}",
                retry_after_s=_retry_after_seconds(response),
                provider=self.provider,
            )
        if status in RETRYABLE_STATUS or status >= 500:
            return EmbeddingServerError(
                f"provider error (HTTP {status}): {detail}", provider=self.provider
            )
        return EmbeddingBadRequestError(
            f"invalid request (HTTP {status}): {detail}", provider=self.provider
        )

    # --- response parsing ----------------------------------------------
    def _parse(self, data: object, expected: int) -> tuple[list[list[float]], int]:
        if not isinstance(data, dict):
            raise EmbeddingResponseError(
                "provider response was not a JSON object", provider=self.provider
            )
        entries = data.get("data")
        if not isinstance(entries, list) or not entries:
            raise EmbeddingResponseError(
                "provider response contained no embeddings", provider=self.provider
            )
        if len(entries) != expected:
            raise EmbeddingResponseError(
                f"provider returned {len(entries)} embeddings for {expected} inputs",
                provider=self.provider,
            )

        ordered: list[tuple[int, list[float]]] = []
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise EmbeddingResponseError(
                    "provider embedding had an unexpected shape", provider=self.provider
                )
            raw_vector = entry.get("embedding")
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingResponseError(
                    "provider embedding was empty or malformed", provider=self.provider
                )
            vector: list[float] = []
            for value in raw_vector:
                if not isinstance(value, int | float):
                    raise EmbeddingResponseError(
                        "provider embedding contained a non-numeric value",
                        provider=self.provider,
                    )
                vector.append(float(value))
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"provider returned {len(vector)}-dimension vectors but "
                    f"{self.dimensions} are configured (model {self.model!r}); "
                    "changing dimensions requires a new migration",
                    provider=self.provider,
                )
            index = entry.get("index", position)
            ordered.append((index if isinstance(index, int) else position, vector))
        ordered.sort(key=lambda item: item[0])

        raw_usage = data.get("usage")
        tokens = 0
        if isinstance(raw_usage, dict):
            value = raw_usage.get("prompt_tokens", raw_usage.get("total_tokens", 0))
            tokens = value if isinstance(value, int) else 0
        return [vector for _, vector in ordered], tokens

    # --- public API -----------------------------------------------------
    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed ``texts`` in order, one request per batch of <= max_batch_size."""
        if not texts:
            return EmbeddingResult(vectors=[], tokens=0, batches=0)

        vectors: list[list[float]] = []
        tokens = 0
        batches = 0
        for start in range(0, len(texts), self.max_batch_size):
            batch = list(texts[start : start + self.max_batch_size])
            batch_vectors, batch_tokens = await self._embed_batch(batch)
            vectors.extend(batch_vectors)
            tokens += batch_tokens
            batches += 1
        self.tokens_used += tokens
        return EmbeddingResult(vectors=vectors, tokens=tokens, batches=batches)

    async def _embed_batch(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        """One request, retried on retryable failures. Terminal errors raise."""
        if not batch:
            return [], 0
        if len(batch) > self.max_batch_size:
            raise ValueError(f"batch of {len(batch)} exceeds max_batch_size {self.max_batch_size}")

        payload = self._payload(batch)
        last_error: EmbeddingError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http().post("/embeddings", json=payload)
            except httpx.TimeoutException as exc:
                last_error = EmbeddingTimeoutError(
                    f"provider timed out after {self._timeout_s}s", provider=self.provider
                )
                logger.warning(
                    "embedding timeout (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
            except httpx.TransportError as exc:
                last_error = EmbeddingServerError(
                    f"transport failure: {exc}", provider=self.provider
                )
                logger.warning("embedding transport error (attempt %d): %s", attempt + 1, exc)
            else:
                if response.status_code == 200:
                    try:
                        parsed = response.json()
                    except ValueError as exc:
                        raise EmbeddingResponseError(
                            f"provider returned invalid JSON: {exc}", provider=self.provider
                        ) from exc
                    vectors, tokens = self._parse(parsed, len(batch))
                    if tokens == 0:
                        # Providers that omit usage: fall back to an estimate so
                        # token accounting stays honest instead of reporting 0.
                        tokens = sum(estimate_tokens(text) for text in batch)
                    return vectors, tokens

                error = self._error_for_status(response)
                if not error.retryable:
                    raise error
                last_error = error
                logger.warning(
                    "embedding request failed (attempt %d, HTTP %d)",
                    attempt + 1,
                    response.status_code,
                )
                retry_after = (
                    error.retry_after_s if isinstance(error, EmbeddingRateLimitError) else None
                )
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_delay(attempt, retry_after))
                continue

            if attempt < self._max_retries:
                await self._sleep(self._backoff_delay(attempt, None))

        assert last_error is not None  # loop always sets it before exhausting retries
        raise last_error


# ---------------------------------------------------------------------------
# Construction from settings
# ---------------------------------------------------------------------------


def build_embedder(
    settings: Settings | None = None,
    *,
    kind: str = "auto",
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> Embedder:
    """Build the configured embedder.

    ``kind``: ``"fake"`` for the deterministic test embedder, ``"openai"`` for the
    HTTP client, ``"auto"`` (default) for the HTTP client against whatever
    ``LLM_BASE_URL`` points at.

    Embeddings reuse ``LLM_BASE_URL`` / ``LLM_API_KEY``: there is no separate
    embedding provider configuration yet. That is correct for the local-first
    setup (Ollama at ``http://localhost:11434/v1``, no key) and still works for a
    hosted provider. If embeddings move to a different vendor than completions,
    add ``EMBEDDING_BASE_URL`` / ``EMBEDDING_API_KEY`` to ``config.py`` rather
    than duplicating a URL in code (see the indexing HANDOFF).
    """
    resolved = settings or get_settings()
    if kind == "fake":
        return DeterministicFakeEmbedder(dimensions=resolved.embedding_dimensions)
    if kind not in {"auto", "openai"}:
        raise IndexingConfigError(f"unknown embedder kind {kind!r}")

    if not resolved.llm_base_url:
        raise IndexingConfigError("LLM_BASE_URL is empty: no embedding endpoint to call")

    # Local-first: Ollama's OpenAI-compatible endpoint needs no API key, so
    # "auto" does not require one. A remote provider without a key fails on the
    # first request with EmbeddingAuthError, which is the right place for it.
    api_key = resolved.llm_api_key.get_secret_value() if resolved.llm_api_key else None
    embedder = OpenAICompatibleEmbedder(
        base_url=resolved.llm_base_url,
        api_key=api_key,
        model=resolved.embedding_model,
        dimensions=resolved.embedding_dimensions,
        timeout_s=resolved.llm_timeout_s,
        max_retries=resolved.llm_max_retries,
        backoff_base_s=resolved.llm_backoff_base_s,
        transport=transport,
        sleep=sleep,
    )
    return embedder


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After`` (delta-seconds form) when the provider sends it."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


__all__ = [
    "MAX_BATCH_SIZE",
    "DeterministicFakeEmbedder",
    "Embedder",
    "EmbeddingResult",
    "OpenAICompatibleEmbedder",
    "build_embedder",
]
