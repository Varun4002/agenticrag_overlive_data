"""Shared async HTTP framework for source connectors.

Every connector (Jira today, Notion/Gmail later) inherits the same behaviour from
here so retry, rate-budget, auth and error semantics do not drift per vendor:

- credentials come from the environment via :class:`connectors.settings.ConnectorSettings`
  and are never logged;
- every request is metered against a per-run :class:`RateBudget`;
- 429/5xx/transport failures retry with bounded exponential backoff + full
  jitter, honouring ``Retry-After``;
- failures surface as the typed errors below, split into retryable vs terminal,
  the same shape as ``llm/errors.py`` so callers can map them consistently.

Read-only by construction: this class exposes ``GET`` only. No connector may
mutate a source system (docs/agents/01-connector-agent.md, scope rules).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import httpx

logger = logging.getLogger(__name__)

# Statuses worth retrying: timeout/conflict/too-early plus rate limiting and 5xx
# (handled separately so a 429 can honour Retry-After).
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


# --- typed errors -------------------------------------------------------
class ConnectorError(Exception):
    """Base class for every connector failure."""

    retryable: bool = False

    def __init__(
        self, message: str, *, source: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.source = source
        self.status_code = status_code


class ConnectorConfigError(ConnectorError):
    """Required configuration (env var) is missing or invalid — not retryable."""


class ConnectorAuthError(ConnectorError):
    """Missing/expired/invalid credentials (HTTP 401/403) — retrying cannot help."""


class ConnectorNotFoundError(ConnectorError):
    """The requested item does not exist (HTTP 404) — terminal."""


class ConnectorBadRequestError(ConnectorError):
    """The source rejected the request shape/query (4xx other than 401/403/404/429)."""


class ConnectorRateLimitError(ConnectorError):
    """The source rate limited us (HTTP 429)."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
        source: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, source=source, status_code=status_code)
        self.retry_after_s = retry_after_s


class ConnectorServerError(ConnectorError):
    """The source returned a 5xx or a transient transport failure."""

    retryable = True


class ConnectorTimeoutError(ConnectorServerError):
    """No answer within the configured timeout."""


class ConnectorResponseError(ConnectorError):
    """The source answered 2xx, but the payload did not have the expected shape."""


class ConnectorRateBudgetExhausted(ConnectorError):
    """This run used its whole request budget; stop fetching and report partial data.

    Deliberately *not* retryable: the brief's failure table says a run that runs
    out of budget returns a partial ``SyncResult`` instead of hammering the API.
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message, source=source)


# --- rate budget --------------------------------------------------------
@dataclass(slots=True)
class RateBudget:
    """Per-source request budget for one sync run.

    A fixed window counter, not a token bucket: sync jobs are short, bursty and
    idempotent, so the useful question is "may I spend another request at all?"
    rather than "at what rate?".

    The brief asks for the budget to be tracked in ``core.sync_state``. That
    table has no budget column (schema frozen), so the budget is per-process per
    run and the gap is reported in ``HANDOFF.md``.
    """

    max_calls: int
    window_s: float = 60.0
    _used: int = 0
    _window_started: float = 0.0
    _clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self._window_started == 0.0:
            self._window_started = self._clock()

    def _roll_window(self) -> None:
        now = self._clock()
        if now - self._window_started >= self.window_s:
            self._window_started = now
            self._used = 0

    @property
    def used(self) -> int:
        self._roll_window()
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def spend(self, *, source: str = "unknown") -> None:
        """Consume one request slot, or raise :class:`ConnectorRateBudgetExhausted`."""
        self._roll_window()
        if self._used >= self.max_calls:
            raise ConnectorRateBudgetExhausted(
                f"request budget exhausted ({self.max_calls} calls per "
                f"{self.window_s:g}s window)",
                source=source,
            )
        self._used += 1


@dataclass(slots=True)
class ConnectorStats:
    """Cheap per-run counters, surfaced in trace payloads and the hand-off logs."""

    requests: int = 0
    retries: int = 0
    retry_after_honoured: int = 0

    def as_payload(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "retry_after_honoured": self.retry_after_honoured,
        }


@dataclass(slots=True)
class Pagination:
    """Vendor paging knobs for :meth:`BaseConnector.paginate`.

    Defaults match Jira Cloud's offset paging. Cursor-based vendors (Notion,
    Gmail) get their own knobs when those adapters land in M2/M5 — no
    speculative abstraction here.
    """

    items_key: str = "values"
    start_key: str = "startAt"
    limit_key: str = "maxResults"
    total_key: str = "total"


# --- connector base -----------------------------------------------------
class BaseConnector:
    """Read-only, rate-budgeted, retrying JSON client shared by all connectors."""

    source: str = "unknown"

    def __init__(
        self,
        *,
        base_url: str,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 20.0,
        budget: RateBudget | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self.budget = budget
        self.stats = ConnectorStats()
        self._transport = transport
        # Injected so tests never really wait.
        self._sleep = sleep or asyncio.sleep
        self._client: httpx.AsyncClient | None = None

    # --- transport lifecycle -------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s),
                headers={"Accept": "application/json", **self._headers},
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- retry policy ---------------------------------------------------
    def _backoff_delay(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff with full jitter, bounded by ``backoff_max_s``."""
        if retry_after_s is not None:
            return min(retry_after_s, self._backoff_max_s)
        ceiling = min(self._backoff_base_s * (2**attempt), self._backoff_max_s)
        return random.uniform(0.0, ceiling)

    def _error_for_status(self, response: httpx.Response) -> ConnectorError:
        status = response.status_code
        detail = _safe_detail(response)
        if status in {401, 403}:
            return ConnectorAuthError(
                f"credentials rejected (HTTP {status}): {detail}",
                source=self.source,
                status_code=status,
            )
        if status == 404:
            return ConnectorNotFoundError(
                f"not found (HTTP 404): {detail}", source=self.source, status_code=status
            )
        if status == 429:
            return ConnectorRateLimitError(
                f"rate limited: {detail}",
                retry_after_s=_retry_after_seconds(response),
                source=self.source,
                status_code=status,
            )
        if status in RETRYABLE_STATUS or status >= 500:
            return ConnectorServerError(
                f"source error (HTTP {status}): {detail}", source=self.source, status_code=status
            )
        return ConnectorBadRequestError(
            f"invalid request (HTTP {status}): {detail}", source=self.source, status_code=status
        )

    # --- public request API ---------------------------------------------
    async def get_json(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET ``path`` and return the decoded JSON object.

        Raises a :class:`ConnectorError` subclass: nothing else escapes.
        """
        last_error: ConnectorError | None = None
        for attempt in range(self._max_retries + 1):
            if self.budget is not None:
                # Retries are real requests against the vendor quota, so each
                # attempt spends budget.
                self.budget.spend(source=self.source)
            try:
                response = await self._http().get(path, params=dict(params or {}))
            except httpx.TimeoutException:
                last_error = ConnectorTimeoutError(
                    f"source timed out after {self._timeout_s}s", source=self.source
                )
                logger.warning("%s timeout (attempt %d)", self.source, attempt + 1)
                self.stats.retries += 1
            except httpx.TransportError as exc:
                last_error = ConnectorServerError(
                    f"transport failure: {exc}", source=self.source
                )
                logger.warning("%s transport error (attempt %d): %s", self.source, attempt + 1, exc)
                self.stats.retries += 1
            else:
                self.stats.requests += 1
                if response.status_code < 300:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ConnectorResponseError(
                            f"source returned invalid JSON: {exc}", source=self.source
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ConnectorResponseError(
                            "source returned a non-object JSON payload", source=self.source
                        )
                    return payload

                error = self._error_for_status(response)
                if not error.retryable:
                    raise error

                last_error = error
                retry_after = (
                    error.retry_after_s if isinstance(error, ConnectorRateLimitError) else None
                )
                if retry_after is not None:
                    self.stats.retry_after_honoured += 1
                self.stats.retries += 1
                logger.warning(
                    "%s request failed (attempt %d, HTTP %d)",
                    self.source,
                    attempt + 1,
                    response.status_code,
                )
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_delay(attempt, retry_after))
                continue

            if attempt < self._max_retries:
                await self._sleep(self._backoff_delay(attempt, None))

        assert last_error is not None  # the loop always sets it before exhausting retries
        raise last_error

    async def paginate(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        pagination: Pagination | None = None,
        page_size: int = 50,
        max_items: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield items from an offset-paginated collection endpoint.

        Stops on: an empty page, a page shorter than ``page_size``, ``start_at``
        reaching the reported ``total``, or ``max_items``. Any per-item filtering
        is the caller's job — this helper never inspects items beyond their type.
        """
        paging = pagination or Pagination()
        start = 0
        emitted = 0
        while True:
            query: dict[str, object] = dict(params or {})
            query[paging.start_key] = start
            query[paging.limit_key] = page_size
            payload = await self.get_json(path, query)

            items = payload.get(paging.items_key)
            if items is None:
                raise ConnectorResponseError(
                    f"response had no {paging.items_key!r} list", source=self.source
                )
            if not isinstance(items, list):
                raise ConnectorResponseError(
                    f"expected a list at {paging.items_key!r}", source=self.source
                )
            if not items:
                return

            for item in items:
                if not isinstance(item, dict):
                    raise ConnectorResponseError(
                        f"{paging.items_key!r} contained a non-object item", source=self.source
                    )
                yield item
                emitted += 1
                if max_items is not None and emitted >= max_items:
                    return

            start += len(items)
            total = payload.get(paging.total_key)
            if isinstance(total, int) and start >= total:
                return
            if len(items) < page_size:
                return


def _safe_detail(response: httpx.Response) -> str:
    """Short error body — truncated so a huge HTML error page cannot flood logs."""
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive: undecodable body
        return "<undecodable body>"
    return (text or "")[:500]


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After`` (delta-seconds form, which Jira/Notion/Gmail send)."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # HTTP-date form is legal per RFC 9110 but none of our sources use it.
        return None
