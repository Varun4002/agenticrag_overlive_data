"""OpenAI-compatible async HTTP client.

Works against OpenAI itself and any provider exposing the same
``POST {base_url}/chat/completions`` shape (Azure-compatible gateways, Groq,
local vLLM/Ollama proxies). Adds what agents need and the raw API does not:
timeouts, bounded exponential backoff with jitter, error classification, and
token accounting.

The API key is read from settings (``LLM_API_KEY``) and never logged.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

import httpx

from llm.base import CompletionRequest, CompletionResponse, TokenLedger, Usage
from llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# Statuses worth retrying: request timeout/conflict/too-early, rate limiting, 5xx.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


class OpenAICompatibleClient:
    """Thin async client for OpenAI-style chat completions."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        default_model: str,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 20.0,
        ledger: TokenLedger | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self.ledger = ledger or TokenLedger()
        self._transport = transport
        # Injected so tests run without real waiting.
        self._sleep = sleep or asyncio.sleep
        self._client: httpx.AsyncClient | None = None

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

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- request building ----------------------------------------------
    def _payload(self, request: CompletionRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model or self._default_model,
            "messages": [m.model_dump(mode="json") for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        payload.update(request.extra)
        return payload

    def _backoff_delay(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff with full jitter, bounded by ``backoff_max_s``."""
        if retry_after_s is not None:
            return min(retry_after_s, self._backoff_max_s)
        ceiling = min(self._backoff_base_s * (2**attempt), self._backoff_max_s)
        return random.uniform(0.0, ceiling)

    # --- error mapping --------------------------------------------------
    def _error_for_status(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        detail = self._safe_detail(response)
        if status in {401, 403}:
            return LLMAuthError(f"provider rejected credentials (HTTP {status}): {detail}")
        if status == 429:
            return LLMRateLimitError(
                f"rate limited by provider: {detail}",
                retry_after_s=_retry_after_seconds(response),
            )
        if status in RETRYABLE_STATUS or status >= 500:
            return LLMServerError(f"provider error (HTTP {status}): {detail}")
        return LLMBadRequestError(f"invalid request (HTTP {status}): {detail}")

    @staticmethod
    def _safe_detail(response: httpx.Response) -> str:
        """Short error body; truncated so a malformed response can't flood logs."""
        text = response.text or ""
        return text[:500]

    # --- response parsing ----------------------------------------------
    def _parse(self, data: object, model: str, latency_ms: int) -> CompletionResponse:
        if not isinstance(data, dict):
            raise LLMResponseError("provider response was not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("provider response contained no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseError("provider choice had an unexpected shape")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("provider choice had no text content")

        raw_usage = data.get("usage")
        usage = Usage()
        if isinstance(raw_usage, dict):
            usage = Usage(
                prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
                completion_tokens=_as_int(raw_usage.get("completion_tokens")),
                total_tokens=_as_int(raw_usage.get("total_tokens")),
            )

        finish_reason = first.get("finish_reason")
        request_id = data.get("id")
        return CompletionResponse(
            text=content,
            model=str(data.get("model") or model),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            provider_request_id=request_id if isinstance(request_id, str) else None,
        )

    # --- public API -----------------------------------------------------
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """POST one completion, retrying retryable failures with backoff."""
        payload = self._payload(request)
        model = str(payload["model"])
        last_error: LLMError | None = None

        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self._http().post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(
                    f"provider timed out after {self._timeout_s}s", provider=self.provider
                )
                logger.warning(
                    "llm timeout (attempt %d/%d): %s", attempt + 1, self._max_retries + 1, exc
                )
            except httpx.TransportError as exc:
                last_error = LLMServerError(
                    f"transport failure: {exc}", provider=self.provider
                )
                logger.warning("llm transport error (attempt %d): %s", attempt + 1, exc)
            else:
                if response.status_code == 200:
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    try:
                        parsed = self._parse(response.json(), model, latency_ms)
                    except ValueError as exc:
                        # Invalid JSON is not worth retrying for a 200 response.
                        raise LLMResponseError(
                            f"provider returned invalid JSON: {exc}", provider=self.provider
                        ) from exc
                    self.ledger.record(parsed.model, parsed.usage)
                    return parsed

                error = self._error_for_status(response)
                if not error.retryable:
                    raise error
                last_error = error
                logger.warning(
                    "llm request failed (attempt %d, HTTP %d)", attempt + 1, response.status_code
                )
                retry_after = (
                    error.retry_after_s if isinstance(error, LLMRateLimitError) else None
                )
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_delay(attempt, retry_after))
                continue

            if attempt < self._max_retries:
                await self._sleep(self._backoff_delay(attempt, None))

        assert last_error is not None  # loop always sets it before exhausting retries
        raise last_error


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After`` (delta-seconds form) when the provider sends it."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def build_llm_client(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout_s: float = 30.0,
    max_retries: int = 3,
    backoff_base_s: float = 0.5,
    ledger: TokenLedger | None = None,
) -> OpenAICompatibleClient:
    """Construct the client from plain values (callers pass settings)."""
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        default_model=model,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        ledger=ledger,
    )
