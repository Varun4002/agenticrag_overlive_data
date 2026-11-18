"""HTTP routes: ``POST /ask`` and ``GET /health``.

``/ask`` runs the orchestrator (retrieve → scrub → verify → synthesize) and
returns the frozen :class:`~orchestrator.contracts.AskResponse`. It never
invents an answer: when the evidence is insufficient it returns a typed
refusal, and when the deployment cannot search at all it returns 503 instead of
a refusal (a refusal would look like "no evidence found").

Audit policy (``AUDIT_STRICT``, default on): a request is served only when its
audit events are actually written. If the trace sink fails, the request is
refused with 503 rather than answered unaudited.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from api.dependencies import EventSinkDep, TracerDep
from config import Settings, get_settings
from db import session as db_session
from orchestrator.contracts import AskRequest, AskResponse
from orchestrator.run import Orchestrator
from retrieval.errors import RetrievalConfigError

logger = logging.getLogger(__name__)

router = APIRouter()

AUDIT_UNAVAILABLE = (
    "audit trail unavailable: refusing to answer without an audit record"
)


def _require_auditable_settings(settings: Settings) -> None:
    """Fail closed when prod demands an audit trail it cannot write."""
    if settings.audit_strict and settings.app_env == "prod" and not settings.database_configured:
        raise HTTPException(
            status_code=503,
            detail="audit trail required in prod: DATABASE_URL is not configured",
        )


@router.post("/ask", response_model=AskResponse, summary="Ask a question over live work data")
async def ask(req: AskRequest, tracer: TracerDep, sink: EventSinkDep) -> AskResponse:
    """Answer one question from retrieved, scrubbed, verified evidence."""
    settings = get_settings()
    _require_auditable_settings(settings)

    started = time.perf_counter()
    received_ok = await tracer.emit(
        "api",
        "api.request.received",
        {
            "question": req.question,
            "question_chars": len(req.question),
            "session_id": req.session_id,
            "user_id": req.user_id,
        },
    )
    if not received_ok and settings.audit_strict:
        raise HTTPException(status_code=503, detail=AUDIT_UNAVAILABLE)

    orchestrator = Orchestrator(settings=settings, sink=sink)
    try:
        response = await orchestrator.ask(req, trace_id=tracer.trace_id)
    except RetrievalConfigError as exc:
        # A misconfigured deployment is not "no evidence found".
        await tracer.emit(
            "api",
            "api.request.failed",
            {"reason": "configuration", "detail": str(exc)[:300]},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("unhandled failure for trace %s", tracer.trace_id)
        await tracer.emit(
            "api",
            "api.request.failed",
            {"reason": "internal_error"},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(
            status_code=500, detail="internal error while answering; see the trace id"
        ) from None

    completed_ok = await tracer.emit(
        "api",
        "api.request.completed",
        {
            "status": "refused" if response.refusal else "answered",
            "answer_chars": len(response.answer),
            "citation_count": len(response.citations),
            "refusal_reason": response.refusal.reason if response.refusal else None,
            "degraded": response.degraded,
        },
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    if not completed_ok and settings.audit_strict:
        raise HTTPException(status_code=503, detail=AUDIT_UNAVAILABLE)

    return response


@router.get("/health", summary="Liveness plus dependency status")
async def health() -> dict[str, object]:
    """Report liveness and whether the database is configured/reachable."""
    settings = get_settings()
    reachable = await db_session.database_available()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "database": {
            "configured": settings.database_configured,
            "reachable": reachable,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }
