"""Shared test fixtures.

API tests do not need a database: ``get_event_sink`` is overridden with an
in-memory sink, so the trace assertions check the same code path the database
sink uses without requiring Postgres.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_event_sink
from api.main import create_app
from db import session as db_session
from db.tracing import InMemoryEventSink


@pytest.fixture
def sink() -> InMemoryEventSink:
    return InMemoryEventSink()


@pytest.fixture
def client(sink: InMemoryEventSink, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with tracing captured in memory and no DB requirement."""
    monkeypatch.setenv("DATABASE_URL", "")
    from config import get_settings

    get_settings.cache_clear()
    # Drop any engine cached by an earlier test so /health really sees "no DB".
    db_session.reset_engine_state()

    app = create_app()
    app.dependency_overrides[get_event_sink] = lambda: sink
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    db_session.reset_engine_state()


@pytest.fixture
def database_url() -> str | None:
    """The configured DATABASE_URL, or None when the environment has none."""
    url = os.environ.get("DATABASE_URL")
    return url or None
