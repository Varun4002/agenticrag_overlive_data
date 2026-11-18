"""Framework tests: retries, Retry-After, typed errors, budget, pagination."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from connectors.base import (
    BaseConnector,
    ConnectorAuthError,
    ConnectorBadRequestError,
    ConnectorNotFoundError,
    ConnectorRateBudgetExhausted,
    ConnectorRateLimitError,
    ConnectorResponseError,
    ConnectorServerError,
    ConnectorTimeoutError,
    Pagination,
    RateBudget,
)

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class RecordingConnector(BaseConnector):
    """Concrete connector with a fake transport and recorded sleeps."""

    source = "test"

    def __init__(
        self,
        handler: Handler,
        *,
        max_retries: int = 2,
        backoff_base_s: float = 0.01,
        budget: RateBudget | None = None,
    ) -> None:
        self.sleeps: list[float] = []
        super().__init__(
            base_url="https://fake.local",
            headers={"Authorization": "Basic not-a-real-token"},
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            budget=budget,
            transport=httpx.MockTransport(handler),
            sleep=self._record_sleep,
        )

    async def _record_sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def page(items: list[dict[str, Any]], *, start: int, total: int, size: int) -> httpx.Response:
    return httpx.Response(
        200, json={"startAt": start, "maxResults": size, "total": total, "values": items}
    )


async def test_retries_server_error_then_succeeds() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"ok": True})

    connector = RecordingConnector(handler)
    try:
        payload = await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert payload == {"ok": True}
    assert calls == 2
    assert connector.stats.retries == 1
    assert len(connector.sleeps) == 1
    assert 0.0 <= connector.sleeps[0] <= 0.01  # jitter never exceeds the ceiling


async def test_gives_up_after_max_retries() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    connector = RecordingConnector(handler, max_retries=2)
    try:
        with pytest.raises(ConnectorServerError) as excinfo:
            await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert calls == 3  # first attempt + 2 retries
    assert excinfo.value.retryable is True
    assert excinfo.value.source == "test"


async def test_rate_limit_honours_retry_after() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "0.25"})

    connector = RecordingConnector(handler, max_retries=1)
    try:
        with pytest.raises(ConnectorRateLimitError) as excinfo:
            await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert excinfo.value.retry_after_s == 0.25
    assert connector.sleeps == [0.25]
    # Both attempts answered 429 with Retry-After; the retry between them slept 0.25s.
    assert connector.stats.retry_after_honoured == 2


async def test_retry_after_is_capped_by_backoff_max() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="very slow", headers={"Retry-After": "3600"})

    connector = RecordingConnector(handler, max_retries=1)
    try:
        with pytest.raises(ConnectorRateLimitError):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert connector.sleeps == [20.0]  # backoff_max_s


async def test_auth_error_is_not_retried() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad token")

    connector = RecordingConnector(handler, max_retries=3)
    try:
        with pytest.raises(ConnectorAuthError):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert calls == 1
    assert connector.sleeps == []


async def test_not_found_and_bad_request_are_terminal() -> None:
    async def not_found(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    async def bad_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad jql")

    cases = ((not_found, ConnectorNotFoundError), (bad_request, ConnectorBadRequestError))
    for handler, expected in cases:
        connector = RecordingConnector(handler, max_retries=3)
        try:
            with pytest.raises(expected):
                await connector.get_json("/thing")
        finally:
            await connector.aclose()


async def test_timeout_retried_then_raised() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    connector = RecordingConnector(handler, max_retries=1)
    try:
        with pytest.raises(ConnectorTimeoutError):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()

    assert calls == 2
    assert len(connector.sleeps) == 1


async def test_invalid_json_raises_response_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    connector = RecordingConnector(handler)
    try:
        with pytest.raises(ConnectorResponseError):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()


async def test_non_object_json_raises_response_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    connector = RecordingConnector(handler)
    try:
        with pytest.raises(ConnectorResponseError):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()


async def test_budget_stops_further_requests() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    budget = RateBudget(max_calls=2)
    connector = RecordingConnector(handler, budget=budget)
    try:
        await connector.get_json("/a")
        await connector.get_json("/b")
        with pytest.raises(ConnectorRateBudgetExhausted):
            await connector.get_json("/c")
    finally:
        await connector.aclose()

    assert calls == 2  # the third request never reached the server
    assert budget.remaining == 0


async def test_budget_counts_retries() -> None:
    """A retry is a real request: it must spend budget too."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    connector = RecordingConnector(handler, max_retries=5, budget=RateBudget(max_calls=1))
    try:
        with pytest.raises(ConnectorRateBudgetExhausted):
            await connector.get_json("/thing")
    finally:
        await connector.aclose()


async def test_budget_window_rolls_over() -> None:
    now = [1000.0]
    budget = RateBudget(max_calls=1, window_s=60.0, _clock=lambda: now[0])
    budget.spend()
    with pytest.raises(ConnectorRateBudgetExhausted):
        budget.spend()
    now[0] += 61.0  # next window
    budget.spend()
    assert budget.remaining == 0


async def test_paginate_walks_offset_pages() -> None:
    seen_starts: list[int] = []
    first = [{"id": 1}, {"id": 2}]
    second = [{"id": 3}]

    async def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startAt"])
        seen_starts.append(start)
        if start == 0:
            return page(first, start=0, total=3, size=2)
        return page(second, start=2, total=3, size=2)

    connector = RecordingConnector(handler)
    try:
        items = [
            item
            async for item in connector.paginate("/values", page_size=2, pagination=Pagination())
        ]
    finally:
        await connector.aclose()

    assert [item["id"] for item in items] == [1, 2, 3]
    assert seen_starts == [0, 2]


async def test_paginate_stops_on_max_items() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startAt"])
        items = [{"id": start}, {"id": start + 1}]
        return page(items, start=start, total=100, size=2)

    connector = RecordingConnector(handler)
    try:
        items = [item async for item in connector.paginate("/values", page_size=2, max_items=3)]
    finally:
        await connector.aclose()

    assert len(items) == 3
    assert connector.stats.requests == 2


async def test_paginate_requires_items_key() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    connector = RecordingConnector(handler)
    try:
        with pytest.raises(ConnectorResponseError):
            [item async for item in connector.paginate("/values")]
    finally:
        await connector.aclose()


async def test_paginate_returns_nothing_on_empty_page() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"values": [], "total": 0})

    connector = RecordingConnector(handler)
    try:
        items = [item async for item in connector.paginate("/values")]
    finally:
        await connector.aclose()

    assert items == []
