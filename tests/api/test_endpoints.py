"""Endpoint tests: ``/ask`` needs a database (and says so), plus ``/health``.

The pipeline itself is covered by ``tests/orchestrator``; here we only assert the
HTTP boundary: validation, honest failure when the deployment cannot search, and
that every request is traced either way.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from db.tracing import InMemoryEventSink

PAYLOAD = {
    "question": "What tasks are due this week in Project Phoenix?",
    "session_id": "sess-1",
    "user_id": "user-1",
}


def test_health_reports_ok_without_database(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == {"configured": False, "reachable": False}


def test_ask_without_database_is_a_503_not_a_refusal(client: TestClient) -> None:
    """No database means "cannot search", which is a deployment error."""
    response = client.post("/ask", json=PAYLOAD)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "DATABASE_URL" in detail
    # A refusal here would be indistinguishable from "no evidence found".
    assert response.json() != {"answer": "", "citations": []}


def test_ask_failure_is_fully_traced(
    client: TestClient, sink: InMemoryEventSink
) -> None:
    client.post("/ask", json=PAYLOAD)

    trace_ids = {record.trace_id for record in sink.records}
    assert len(trace_ids) == 1
    trace_id = trace_ids.pop()
    events = sink.events_for(trace_id)

    assert [event.event for event in events] == [
        "api.request.received",
        "orchestrator.intake",
        "orchestrator.plan",
        "orchestrator.failed",
        "api.request.failed",
    ]
    assert [event.agent for event in events] == [
        "api",
        "orchestrator",
        "orchestrator",
        "orchestrator",
        "api",
    ]
    assert events[0].payload["question"] == PAYLOAD["question"]
    assert events[0].payload["user_id"] == PAYLOAD["user_id"]
    # The plan states what is not implemented, so a trace explains absences.
    assert events[2].payload["skipped_stages"] == [
        "guard_input",
        "memory_check",
        "guard_output",
    ]
    assert events[-1].latency_ms is not None


def test_trace_ids_are_unique_per_request(
    client: TestClient, sink: InMemoryEventSink
) -> None:
    client.post("/ask", json={"question": "a", "session_id": "s", "user_id": "u"})
    client.post("/ask", json={"question": "a", "session_id": "s", "user_id": "u"})

    trace_ids = {record.trace_id for record in sink.records}
    assert len(trace_ids) == 2
    for trace_id in trace_ids:
        assert sink.events_for(trace_id)


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "", "session_id": "s", "user_id": "u"}
    )
    assert response.status_code == 422


def test_ask_rejects_missing_user_id(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "hello", "session_id": "s"})
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [{}, {"question": "hi"}])
def test_ask_unprocessable_payloads(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/ask", json=payload).status_code == 422
