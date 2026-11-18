"""Audit-availability policy: no audit record, no answer.

Covers ``docs/06-guardrails-pii.md`` "Audit availability policy":
- strict mode (default) + trace write fails      -> 503, never an unaudited answer
- strict mode + prod + no DATABASE_URL configured -> 503
- strict mode off                                -> the answer is served anyway
- dev + no DB                                    -> allowed (explicit opt-out)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_event_sink
from api.main import create_app
from config import get_settings
from db import session as db_session
from db.tracing import EventRecord


class FailingSink:
    """A sink whose writes always fail (database down, permissions, ...)."""

    async def write(self, record: EventRecord) -> bool:
        return False


@contextmanager
def _make_client(sink: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", "")
    get_settings.cache_clear()
    db_session.reset_engine_state()
    app = create_app()
    app.dependency_overrides[get_event_sink] = lambda: sink
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    db_session.reset_engine_state()


def test_failing_audit_sink_refuses_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode: a failed trace write must not produce an answer."""
    body = {"question": "what is due?", "session_id": "s", "user_id": "u"}

    with _make_client(FailingSink(), monkeypatch) as client:
        response = client.post("/ask", json=body)

    assert response.status_code == 503
    assert "audit" in response.json()["detail"].lower()


def test_unreachable_database_refuses_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured-but-dead DB goes through the real DatabaseEventSink -> 503."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://rag@localhost:59999/nope")
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_S", "1")
    get_settings.cache_clear()
    db_session.reset_engine_state()

    app = create_app()  # real sink, no override
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "what is due?", "session_id": "s", "user_id": "u"}
        )

    app.dependency_overrides.clear()
    get_settings.cache_clear()
    db_session.reset_engine_state()

    assert response.status_code == 503
    assert "audit" in response.json()["detail"].lower()


def test_strict_mode_off_does_not_block_on_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit dev opt-out: a failed trace write is not itself fatal.

    The request still cannot be answered without a database, but the failure must
    be the missing search dependency — not the audit gate. That difference is the
    whole point of ``AUDIT_STRICT``.
    """
    monkeypatch.setenv("AUDIT_STRICT", "false")

    with _make_client(FailingSink(), monkeypatch) as client:
        response = client.post(
            "/ask", json={"question": "hi", "session_id": "s", "user_id": "u"}
        )

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]
    assert "audit" not in response.json()["detail"].lower()


def test_prod_without_database_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod cannot run unaudited: missing DATABASE_URL is a 503, not a warning."""
    monkeypatch.setenv("APP_ENV", "prod")

    with _make_client(FailingSink(), monkeypatch) as client:
        response = client.post(
            "/ask", json={"question": "hi", "session_id": "s", "user_id": "u"}
        )

    assert response.status_code == 503
    assert "database_url" in response.json()["detail"].lower()
