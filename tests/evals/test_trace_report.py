"""Trace-report and aggregate-stats tests against the scratch database.

They exercise the real ``audit.events`` path (``read_trace`` + jsonb payloads),
using the scratch DB from ``tests/evals/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db import session as db_session
from db.tracing import DatabaseEventSink, EventRecord, new_trace_id
from evals.trace_report import (
    TraceReportError,
    aggregate_stats,
    render_stats_markdown,
    render_trace_markdown,
    trace_report,
)


async def _write(
    factory: async_sessionmaker[AsyncSession],
    trace_id: str,
    agent: str,
    event: str,
    *,
    payload: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> None:
    await DatabaseEventSink(factory).write(
        EventRecord(
            trace_id=trace_id,
            agent=agent,
            event=event,
            payload=payload or {},
            latency_ms=latency_ms,
        )
    )


async def test_trace_report_builds_a_per_agent_timeline(
    eval_session_factory: async_sessionmaker[AsyncSession],
    clean_events: None,
) -> None:
    trace_id = new_trace_id()
    await _write(
        eval_session_factory,
        trace_id,
        "api",
        "api.request.received",
        payload={"question": "What is overdue?", "session_id": "s1"},
        latency_ms=0,
    )
    await _write(
        eval_session_factory,
        trace_id,
        "retrieval",
        "retrieval.ran",
        payload={"candidates": 12, "kept": 5},
        latency_ms=40,
    )
    await _write(
        eval_session_factory,
        trace_id,
        "synthesis",
        "synthesis.refused",
        payload={"reason": "insufficient_evidence"},
        latency_ms=10,
    )
    await _write(
        eval_session_factory,
        trace_id,
        "api",
        "api.request.completed",
        payload={"latency_ms": 60},
        latency_ms=60,
    )

    summary = await trace_report(trace_id, session_factory=eval_session_factory)

    assert summary.found is True
    assert summary.event_count == 4
    assert [event.event for event in summary.events] == [
        "api.request.received",
        "retrieval.ran",
        "synthesis.refused",
        "api.request.completed",
    ]
    assert summary.recorded_latency_ms == 110
    api_stats = next(agent for agent in summary.agents if agent.agent == "api")
    assert api_stats.events == 2
    assert api_stats.total_latency_ms == 60
    assert summary.flags == []

    markdown = render_trace_markdown(summary)
    assert markdown.startswith(f"# Trace `{trace_id}`")
    assert "## By agent" in markdown
    assert "## Timeline" in markdown
    assert "retrieval.ran" in markdown
    assert "insufficient_evidence" in markdown


async def test_trace_report_flags_an_unfinished_trace(
    eval_session_factory: async_sessionmaker[AsyncSession],
    clean_events: None,
) -> None:
    trace_id = new_trace_id()
    await _write(eval_session_factory, trace_id, "api", "api.request.received", latency_ms=0)

    summary = await trace_report(trace_id, session_factory=eval_session_factory)

    assert summary.found is True
    assert "missing_completion_event" in summary.flags


async def test_missing_trace_is_reported_not_invented(
    eval_session_factory: async_sessionmaker[AsyncSession],
    clean_events: None,
) -> None:
    summary = await trace_report("does-not-exist", session_factory=eval_session_factory)

    assert summary.found is False
    assert summary.event_count == 0
    assert summary.flags == ["trace_not_found"]
    assert "not found" in render_trace_markdown(summary)


async def test_aggregate_stats_rates_and_latency(
    eval_session_factory: async_sessionmaker[AsyncSession],
    clean_events: None,
) -> None:
    refused_trace = new_trace_id()
    clean_trace = new_trace_id()

    for trace_id in (refused_trace, clean_trace):
        await _write(
            eval_session_factory,
            trace_id,
            "api",
            "api.request.received",
            latency_ms=5,
        )
        await _write(
            eval_session_factory,
            trace_id,
            "orchestrator",
            "orchestrator.planned",
            latency_ms=15,
        )
        await _write(
            eval_session_factory,
            trace_id,
            "api",
            "api.request.completed",
            latency_ms=30,
        )

    await _write(
        eval_session_factory,
        refused_trace,
        "synthesis",
        "synthesis.refused",
        latency_ms=10,
    )
    await _write(
        eval_session_factory, refused_trace, "memory", "memory.cache_hit", latency_ms=1
    )
    await _write(
        eval_session_factory,
        clean_trace,
        "guardrail",
        "guardrail.injection_quarantined",
        latency_ms=2,
    )

    stats = await aggregate_stats(
        session_factory=eval_session_factory,
        window_hours=1,
        now=datetime.now(UTC),
    )

    assert stats.event_count == 9
    assert stats.trace_count == 2
    assert stats.completed_traces == 2
    assert stats.refusal_rate == pytest.approx(0.5)
    assert stats.refusal_traces == 1
    assert stats.cache_hit_rate == pytest.approx(0.5)
    assert stats.cache_hit_traces == 1
    assert stats.injection_quarantine_rate == pytest.approx(0.5)
    assert stats.over_redaction_rate is None

    api = next(agent for agent in stats.latency_by_agent if agent.agent == "api")
    assert api.events == 4
    assert api.measured_events == 4
    assert api.total_latency_ms == 70
    assert api.p50_latency_ms == 5

    markdown = render_stats_markdown(stats)
    assert markdown.startswith("# Aggregate stats")
    assert "50.00%" in markdown
    assert "## Latency by agent" in markdown


async def test_aggregate_stats_without_completed_traces_reports_none(
    eval_session_factory: async_sessionmaker[AsyncSession],
    clean_events: None,
) -> None:
    stats = await aggregate_stats(
        session_factory=eval_session_factory,
        window_hours=1,
        now=datetime.now(UTC),
    )

    assert stats.event_count == 0
    assert stats.refusal_rate is None
    assert stats.cache_hit_rate is None
    markdown = render_stats_markdown(stats)
    assert "n/a" in markdown
    assert "no audit.events" in markdown


async def test_aggregate_stats_rejects_a_non_positive_window(
    eval_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        await aggregate_stats(session_factory=eval_session_factory, window_hours=0)


async def test_trace_report_without_database_url_explains_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config import get_settings

    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    db_session.reset_engine_state()
    try:
        with pytest.raises(TraceReportError, match="DATABASE_URL is not set"):
            await trace_report("whatever")
    finally:
        get_settings.cache_clear()
        db_session.reset_engine_state()
