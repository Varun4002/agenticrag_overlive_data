"""FastAPI dependency wiring.

Everything the routes need is provided through ``Depends`` so tests can swap
the tracer's sink for an in-memory one without a database (see
``tests/conftest.py``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from db.session import get_session_factory
from db.tracing import EventSink, Tracer, make_sink


def get_event_sink() -> EventSink:
    """Trace sink: database-backed when configured, otherwise a no-op."""
    return make_sink(get_session_factory())


EventSinkDep = Annotated[EventSink, Depends(get_event_sink)]


def get_tracer(sink: EventSinkDep) -> Tracer:
    """Tracer for one request; ``/ask`` assigns the trace id it reports."""
    return Tracer(sink)


TracerDep = Annotated[Tracer, Depends(get_tracer)]
