"""Trace and aggregate observability reports over ``audit.events``.

``trace_report(trace_id)`` rebuilds one request's agent decision timeline from
``audit.events`` (via ``db.tracing.read_trace``), so a milestone review can see
*which agent decided what, in what order, at what latency*.

``aggregate_stats()`` rolls recent events into the dashboard metrics the brief
asks for: latency by agent, refusal rate, and cache-hit / injection-quarantine
rates. Cache-hit and over-redaction numbers are placeholders until the Memory
and Guardrail agents emit the conventional event names below — a rate with no
events is ``None`` (not ``0.0``), so "no data" never masquerades as "no
problems".

Usage::

    python -m evals.trace_report <trace_id>
    python -m evals.trace_report --stats --window-hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import AuditEvent
from db.session import get_session_factory
from db.tracing import read_trace
from evals.scoring import percentile

SessionFactory = async_sessionmaker[AsyncSession]

#: Conventional event names for aggregate metrics. Not part of the frozen
#: audit schema (event names are free-form), so these are documented here and
#: in the component HANDOFF until the emitting agents land.
REFUSAL_EVENTS = ("synthesis.refused", "evidence.insufficient", "guardrail.blocked")
CACHE_HIT_EVENTS = ("memory.cache_hit", "memory.answer_cache.hit")
INJECTION_QUARANTINE_EVENTS = (
    "guardrail.injection_quarantined",
    "guardrail.injection_blocked",
)
OVERREDACTION_EVENTS = ("guardrail.over_redacted",)
COMPLETION_EVENT = "api.request.completed"
PAYLOAD_PREVIEW_CHARS = 160


class TraceReportError(RuntimeError):
    """Tracing is unavailable (no ``DATABASE_URL``) or the query failed."""


def _preview(payload: dict[str, Any]) -> str:
    try:
        text = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > PAYLOAD_PREVIEW_CHARS:
        return text[: PAYLOAD_PREVIEW_CHARS - 1] + "\u2026"
    return text


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%H:%M:%S.%f")[:-3]


class TraceEvent(BaseModel):
    """One row of the timeline."""

    seq: int
    at: datetime
    agent: str
    event: str
    latency_ms: int | None = None
    payload_preview: str = ""


class AgentStats(BaseModel):
    """Per-agent roll-up for one trace."""

    agent: str
    events: int
    measured_events: int
    total_latency_ms: int
    p50_latency_ms: int
    p95_latency_ms: int


class TraceSummary(BaseModel):
    """A full per-agent timeline for one trace."""

    trace_id: str
    found: bool
    event_count: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: int = 0
    recorded_latency_ms: int = 0
    agents: list[AgentStats] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


def _summarise(trace_id: str, rows: Sequence[AuditEvent]) -> TraceSummary:
    if not rows:
        return TraceSummary(
            trace_id=trace_id, found=False, event_count=0, flags=["trace_not_found"]
        )

    events = [
        TraceEvent(
            seq=index,
            at=row.created_at,
            agent=row.agent,
            event=row.event,
            latency_ms=row.latency_ms,
            payload_preview=_preview(row.payload),
        )
        for index, row in enumerate(rows, start=1)
    ]

    by_agent: dict[str, list[int]] = {}
    for row in rows:
        by_agent.setdefault(row.agent, [])
        if row.latency_ms is not None:
            by_agent[row.agent].append(row.latency_ms)
    agents = [
        AgentStats(
            agent=agent,
            events=sum(1 for row in rows if row.agent == agent),
            measured_events=len(latencies),
            total_latency_ms=sum(latencies),
            p50_latency_ms=percentile(latencies, 0.50),
            p95_latency_ms=percentile(latencies, 0.95),
        )
        for agent, latencies in sorted(by_agent.items())
    ]

    started = events[0].at
    finished = events[-1].at
    elapsed_ms = int((finished - started).total_seconds() * 1000)
    recorded = sum(event.latency_ms or 0 for event in events)

    flags: list[str] = []
    if not any(event.event == COMPLETION_EVENT for event in events):
        flags.append("missing_completion_event")
    if not any(event.latency_ms is not None for event in events):
        flags.append("no_recorded_latencies")

    return TraceSummary(
        trace_id=trace_id,
        found=True,
        event_count=len(events),
        started_at=started,
        finished_at=finished,
        elapsed_ms=elapsed_ms,
        recorded_latency_ms=recorded,
        agents=agents,
        events=events,
        flags=flags,
    )


async def trace_report(
    trace_id: str,
    *,
    session_factory: SessionFactory | None = None,
) -> TraceSummary:
    """Load every ``audit.events`` row for ``trace_id`` into a timeline."""
    factory = session_factory or get_session_factory()
    if factory is None:
        raise TraceReportError(
            "DATABASE_URL is not set: trace reports need the audit.events table "
            "(see .env.example)"
        )
    rows = await read_trace(factory, trace_id)
    return _summarise(trace_id, rows)


class AgentLatency(BaseModel):
    """Latency roll-up for one agent across the window."""

    agent: str
    events: int
    measured_events: int
    total_latency_ms: int
    p50_latency_ms: int
    p95_latency_ms: int


class AggregateStats(BaseModel):
    """Dashboard-as-a-script metrics over a time window.

    Rates are ``None`` when the denominator is zero. ``over_redaction_rate`` is
    always ``None`` until an over-redaction signal is defined (see notes).
    """

    window_hours: int
    window_start: datetime
    event_count: int = 0
    trace_count: int = 0
    completed_traces: int = 0
    latency_by_agent: list[AgentLatency] = Field(default_factory=list)
    refusal_rate: float | None = None
    refusal_traces: int = 0
    cache_hit_rate: float | None = None
    cache_hit_traces: int = 0
    injection_quarantine_rate: float | None = None
    injection_quarantine_traces: int = 0
    over_redaction_rate: float | None = None
    notes: list[str] = Field(default_factory=list)


async def aggregate_stats(
    *,
    session_factory: SessionFactory | None = None,
    window_hours: int = 24,
    now: datetime | None = None,
) -> AggregateStats:
    """Aggregate recent ``audit.events`` rows into dashboard metrics."""
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    factory = session_factory or get_session_factory()
    if factory is None:
        raise TraceReportError(
            "DATABASE_URL is not set: aggregate stats need the audit.events table "
            "(see .env.example)"
        )
    window_start = (now or datetime.now(UTC)) - timedelta(hours=window_hours)
    async with factory() as session:
        rows = list(
            await session.scalars(
                select(AuditEvent).where(AuditEvent.created_at >= window_start)
            )
        )

    traces = {row.trace_id for row in rows}
    completed = {
        row.trace_id for row in rows if row.event == COMPLETION_EVENT
    }
    latencies_by_agent: dict[str, list[int]] = {}
    for row in rows:
        latencies_by_agent.setdefault(row.agent, [])
        if row.latency_ms is not None:
            latencies_by_agent[row.agent].append(row.latency_ms)

    def rate(matched: set[str]) -> float | None:
        if not completed:
            return None
        return len(matched & completed) / len(completed)

    refusal_traces = {row.trace_id for row in rows if row.event in REFUSAL_EVENTS}
    cache_hit_traces = {row.trace_id for row in rows if row.event in CACHE_HIT_EVENTS}
    injection_traces = {
        row.trace_id for row in rows if row.event in INJECTION_QUARANTINE_EVENTS
    }

    notes: list[str] = []
    if not rows:
        notes.append("no audit.events in this window")
    elif not completed:
        notes.append(
            f"no {COMPLETION_EVENT} events: rates are undefined, not zero "
            "(run a request through /ask first)"
        )
    notes.append(
        "cache-hit and refusal rates depend on the conventional event names documented "
        "in src/evals/trace_report.py; they will read as no-data until those agents emit them"
    )
    notes.append("over-redaction rate has no emitting event yet — reported as None")

    return AggregateStats(
        window_hours=window_hours,
        window_start=window_start,
        event_count=len(rows),
        trace_count=len(traces),
        completed_traces=len(completed),
        latency_by_agent=[
            AgentLatency(
                agent=agent,
                events=sum(1 for row in rows if row.agent == agent),
                measured_events=len(latencies),
                total_latency_ms=sum(latencies),
                p50_latency_ms=percentile(latencies, 0.50),
                p95_latency_ms=percentile(latencies, 0.95),
            )
            for agent, latencies in sorted(latencies_by_agent.items())
        ],
        refusal_rate=rate(refusal_traces),
        refusal_traces=len(refusal_traces & completed),
        cache_hit_rate=rate(cache_hit_traces),
        cache_hit_traces=len(cache_hit_traces & completed),
        injection_quarantine_rate=rate(injection_traces),
        injection_quarantine_traces=len(injection_traces & completed),
        over_redaction_rate=None,
        notes=notes,
    )


def render_trace_markdown(summary: TraceSummary) -> str:
    """Render a trace summary as a per-agent timeline."""
    header = f"# Trace `{summary.trace_id}`"
    lines = [header, ""]
    if not summary.found:
        lines.append("_Trace not found in `audit.events`._")
        lines.append("")
        return "\n".join(lines)

    started = _format_time(summary.started_at) if summary.started_at else "?"
    finished = _format_time(summary.finished_at) if summary.finished_at else "?"
    lines.append(f"- events: {summary.event_count}")
    lines.append(
        f"- elapsed: {summary.elapsed_ms} ms "
        f"({started} -> {finished} UTC)"
    )
    lines.append(f"- recorded agent latency: {summary.recorded_latency_ms} ms")
    if summary.flags:
        lines.append(f"- flags: `{', '.join(summary.flags)}`")
    lines.append("")

    lines.append("## By agent")
    lines.append("")
    lines.append("| Agent | Events | Measured | Total ms | p50 ms | p95 ms |")
    lines.append("|---|---|---|---|---|---|")
    for agent in summary.agents:
        lines.append(
            f"| {agent.agent} | {agent.events} | {agent.measured_events} "
            f"| {agent.total_latency_ms} | {agent.p50_latency_ms} | {agent.p95_latency_ms} |"
        )
    lines.append("")

    lines.append("## Timeline")
    lines.append("")
    lines.append("| # | Time (UTC) | Agent | Event | Latency ms | Payload |")
    lines.append("|---|---|---|---|---|---|")
    for event in summary.events:
        latency = "-" if event.latency_ms is None else str(event.latency_ms)
        payload = event.payload_preview.replace("|", "\\|")
        lines.append(
            f"| {event.seq} | {_format_time(event.at)} | {event.agent} | {event.event} "
            f"| {latency} | {payload} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_stats_markdown(stats: AggregateStats) -> str:
    """Render aggregate dashboard metrics as markdown."""

    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = [
        f"# Aggregate stats — last {stats.window_hours}h",
        "",
        f"- window start: `{stats.window_start.isoformat(timespec='seconds')}`",
        f"- events: {stats.event_count} across {stats.trace_count} traces "
        f"({stats.completed_traces} completed)",
        "",
        "## Latency by agent",
        "",
        "| Agent | Events | Measured | Total ms | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for agent in stats.latency_by_agent:
        lines.append(
            f"| {agent.agent} | {agent.events} | {agent.measured_events} "
            f"| {agent.total_latency_ms} | {agent.p50_latency_ms} | {agent.p95_latency_ms} |"
        )
    lines.extend(
        [
            "",
            "## Rates",
            "",
            "| Metric | Value | Traces |",
            "|---|---|---|",
            f"| refusal rate | {rate(stats.refusal_rate)} | {stats.refusal_traces} |",
            f"| cache hit rate | {rate(stats.cache_hit_rate)} | {stats.cache_hit_traces} |",
            f"| injection quarantine rate | {rate(stats.injection_quarantine_rate)} "
            f"| {stats.injection_quarantine_traces} |",
            f"| over-redaction rate | {rate(stats.over_redaction_rate)} | - |",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in stats.notes)
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.trace_report",
        description="Per-trace timeline and aggregate stats from audit.events.",
    )
    parser.add_argument("trace_id", nargs="?", help="trace id to report on")
    parser.add_argument("--stats", action="store_true", help="print aggregate stats instead")
    parser.add_argument(
        "--window-hours", type=int, default=24, help="aggregate window in hours (default: 24)"
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    if args.stats:
        stats = await aggregate_stats(window_hours=args.window_hours)
        print(render_stats_markdown(stats), end="")
        return 0
    if not args.trace_id:
        print("error: provide a trace id or --stats", file=sys.stderr)
        return 2
    summary = await trace_report(args.trace_id)
    print(render_trace_markdown(summary), end="")
    return 0 if summary.found else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint; returns the process exit code."""
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except TraceReportError as exc:
        print(f"trace report error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
