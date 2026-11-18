"""Structured tracing into ``audit.events``.

Every agent emits rows shaped ``(trace_id, agent, event, payload, latency_ms)``
so a single query reconstructs a request's full decision path. Writing a trace
never raises: sinks report success instead, and the caller decides whether a
failure is fatal (``AUDIT_STRICT`` — see ``api.routes.ask``).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import AuditEvent

logger = logging.getLogger(__name__)


def new_trace_id() -> str:
    """Generate a trace id for one request."""
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One structured trace event."""

    trace_id: str
    agent: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None


class EventSink(Protocol):
    """Destination for trace events."""

    async def write(self, record: EventRecord) -> bool:
        """Persist one event; never raises.

        Returns ``True`` when the event was persisted **or** intentionally
        dropped (dev no-op sink), ``False`` when persistence was attempted and
        failed. Callers use this to fail closed under ``AUDIT_STRICT``.
        """
        ...


class NullEventSink:
    """Sink that drops events (no database configured)."""

    async def write(self, record: EventRecord) -> bool:
        logger.debug("trace(dropped) %s %s.%s", record.trace_id, record.agent, record.event)
        return True


class InMemoryEventSink:
    """Sink that keeps events in a list — used by API tests without a database."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    async def write(self, record: EventRecord) -> bool:
        self.records.append(record)
        return True

    def events_for(self, trace_id: str) -> list[EventRecord]:
        return [r for r in self.records if r.trace_id == trace_id]


class DatabaseEventSink:
    """Writes events to ``audit.events`` in their own short-lived transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.failed_writes = 0

    async def write(self, record: EventRecord) -> bool:
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(
                    insert(AuditEvent).values(
                        trace_id=record.trace_id,
                        agent=record.agent,
                        event=record.event,
                        payload=record.payload,
                        latency_ms=record.latency_ms,
                    )
                )
        except Exception as exc:
            self.failed_writes += 1
            logger.warning(
                "failed to persist trace event %s.%s for %s: %s",
                record.agent,
                record.event,
                record.trace_id,
                exc,
            )
            return False
        return True


class Tracer:
    """Emits trace events for one request.

    ``agent`` is the emitting component (``api``, ``orchestrator``, ...) and
    ``event`` is a dotted action name (``retrieval.ran``, ``guardrail.blocked``).
    """

    def __init__(self, sink: EventSink, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_trace_id()
        self._sink = sink

    async def emit(
        self,
        agent: str,
        event: str,
        payload: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> bool:
        """Write one event; never raises. Returns False if the write failed."""
        return await self._sink.write(
            EventRecord(
                trace_id=self.trace_id,
                agent=agent,
                event=event,
                payload=payload or {},
                latency_ms=latency_ms,
            )
        )

    @asynccontextmanager
    async def span(
        self,
        agent: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Emit ``<event>.started``/``<event>.finished`` around a block.

        Yields a mutable dict merged into the finished payload, so callers can
        attach outcomes measured inside the block (scores, counts, decisions).
        """
        extra = payload or {}
        started = time.perf_counter()
        await self.emit(agent, f"{event}.started", dict(extra))
        try:
            yield extra
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            await self.emit(agent, f"{event}.finished", dict(extra), latency_ms=latency_ms)


def make_sink(
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> EventSink:
    """Pick a sink: the database if configured, otherwise a no-op sink."""
    return DatabaseEventSink(session_factory) if session_factory is not None else NullEventSink()


async def read_trace(
    session_factory: async_sessionmaker[AsyncSession],
    trace_id: str,
) -> Sequence[AuditEvent]:
    """Load every event for a trace, in write order (debugging/eval helper)."""
    async with session_factory() as session:
        result = await session.scalars(
            select(AuditEvent)
            .where(AuditEvent.trace_id == trace_id)
            .order_by(AuditEvent.id.asc())
        )
        return list(result)


async def with_timed_event(
    tracer: Tracer,
    agent: str,
    event: str,
    payload: dict[str, Any],
    fn: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run ``fn`` and emit one event carrying both its result and its latency."""
    started = time.perf_counter()
    outcome = await fn()
    latency_ms = int((time.perf_counter() - started) * 1000)
    merged: dict[str, Any] = {**payload, **outcome}
    await tracer.emit(agent, event, merged, latency_ms=latency_ms)
    return merged
