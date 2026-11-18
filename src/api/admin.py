"""Operator dashboard API: read the corpus, inspect traces, trigger work.

Everything here exists so a human can see what the pipeline is doing without
writing SQL. Two classes of endpoint:

* **read-only** — overview counts, documents, chunks, traces, retrieval preview,
  PII preview. Safe to call repeatedly; the UI polls the overview.
* **mutating** — connector sync (Jira, Gmail), embedding backfill, fixture
  seeding. Each is serialized by a process-wide lock, capped by a wall-clock
  timeout, and refused when the dashboard is not enabled for this environment.

The dashboard is **not** part of the frozen request path: ``/ask`` does not call
into this module and this module does not change any pipeline contract. It calls
the same public entry points the CLIs use (``connectors.service.sync``,
``indexing.indexer``, ``retrieval.search.retrieve``, ``pii.scrubber.scrub``), so
what the dashboard shows is what the pipeline actually does.

Security: ``ADMIN_ENABLED`` (default: on in dev/test, off otherwise) gates every
route here. Mutating routes can spend money (embedding tokens) and hit external
APIs (Jira), so the dashboard must bind to localhost or sit behind a proxy in any
shared deployment.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import get_settings
from db import session as db_session
from db.tracing import read_trace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

#: Upper bound of a browser-initiated job. Beyond this the caller gets 504 and
#: the work is left to finish (or fail) on its own; re-triggering is safe because
#: sync and indexing are both idempotent.
SYNC_TIMEOUT_S = 180.0
BACKFILL_TIMEOUT_S = 240.0

#: Documents per ``Indexer.run`` call inside one backfill request.
BACKFILL_PAGE_SIZE = 25

MAX_DOCUMENT_CHARS = 200_000
MAX_TEXT_PREVIEW = 2_000

SourceName = Literal["jira", "notion", "gmail"]

_sync_lock = asyncio.Lock()
_backfill_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Dependencies and helpers
# ---------------------------------------------------------------------------


def require_admin() -> None:
    """Refuse the dashboard when it is not enabled for this environment."""
    if not get_settings().admin_available:
        raise HTTPException(
            status_code=403,
            detail=(
                "operator dashboard is disabled for this environment; "
                "set ADMIN_ENABLED=true to expose /ui and /admin/*"
            ),
        )


AdminDep = Annotated[None, Depends(require_admin)]


def _factory() -> async_sessionmaker[AsyncSession]:
    factory = db_session.get_session_factory()
    if factory is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not set: the operator dashboard needs a database",
        )
    return factory


def _iso(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


async def _scalar(session: AsyncSession, sql: str, **params: object) -> int:
    value = await session.scalar(text(sql), params or None)
    return int(value or 0)


def _rows(result: Any) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in result]


# ---------------------------------------------------------------------------
# Read-only: overview
# ---------------------------------------------------------------------------


@router.get("/overview", summary="Counts, health and non-secret configuration")
async def overview(_: AdminDep) -> dict[str, Any]:
    """One payload with everything the dashboard header needs."""
    settings = get_settings()
    factory = db_session.get_session_factory()
    reachable = await db_session.database_available()

    payload: dict[str, Any] = {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
        },
        "database": {
            "configured": settings.database_configured,
            "reachable": reachable,
        },
        "answer_mode": "pipeline",
        "counts": None,
        "by_source": [],
        "sync_state": [],
        "acl_users": [],
        "recent_events": [],
        "config": {
            "llm_model": settings.llm_model,
            "llm_json_model": settings.llm_json_model or settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "evidence_sufficiency_threshold": settings.evidence_sufficiency_threshold,
            "pii_hash_salt_is_default": settings.pii_hash_salt == "dev-only-pii-salt-change-me",
        },
    }
    payload["connectors"] = _connector_settings()
    if factory is None or not reachable:
        return payload

    async with factory() as session:
        payload["counts"] = {
            "documents_current": await _scalar(
                session, "SELECT count(*) FROM core.documents WHERE is_current"
            ),
            "documents_total": await _scalar(session, "SELECT count(*) FROM core.documents"),
            "chunks": await _scalar(session, "SELECT count(*) FROM core.chunks"),
            "chunks_embedded": await _scalar(
                session, "SELECT count(*) FROM core.chunks WHERE embedding IS NOT NULL"
            ),
            "redactions": await _scalar(session, "SELECT count(*) FROM audit.redactions"),
            "events": await _scalar(session, "SELECT count(*) FROM audit.events"),
            "traces": await _scalar(
                session, "SELECT count(DISTINCT trace_id) FROM audit.events"
            ),
            "answer_cache": await _scalar(session, "SELECT count(*) FROM mem.answer_cache"),
            "entities": await _scalar(session, "SELECT count(*) FROM mem.entity_registry"),
        }
        payload["by_source"] = _rows(
            await session.execute(
                text(
                    "SELECT d.source::text AS source, count(*) AS n,"
                    " sum(CASE WHEN c.n IS NULL THEN 0 ELSE 1 END) AS indexed"
                    " FROM core.documents d"
                    " LEFT JOIN (SELECT document_id, count(*) AS n FROM core.chunks"
                    "            GROUP BY document_id) c ON c.document_id = d.id"
                    " WHERE d.is_current GROUP BY d.source ORDER BY d.source"
                )
            )
        )
        payload["sync_state"] = [
            {
                "source": row["source"],
                "watermark": _iso(row["watermark"]),
                "last_run_at": _iso(row["last_run_at"]),
                "last_status": row["last_status"],
                "error": row["error"],
            }
            for row in _rows(
                await session.execute(
                    text(
                        "SELECT source, watermark, last_run_at, last_status, error"
                        " FROM core.sync_state ORDER BY source"
                    )
                )
            )
        ]
        payload["acl_users"] = _rows(
            await session.execute(
                text(
                    "SELECT u.user_id, count(*) AS documents"
                    " FROM core.documents d,"
                    " jsonb_array_elements_text(d.acl->'users') AS u(user_id)"
                    " WHERE d.is_current AND u.user_id <> '*'"
                    " GROUP BY u.user_id ORDER BY documents DESC, u.user_id LIMIT 8"
                )
            )
        )
        payload["recent_events"] = _rows(
            await session.execute(
                text(
                    "SELECT agent, event, created_at FROM audit.events"
                    " ORDER BY id DESC LIMIT 12"
                )
            )
        )
        for row in payload["recent_events"]:
            row["created_at"] = _iso(row["created_at"])
    return payload


def _connector_settings() -> dict[str, Any]:
    """Non-secret connector state; tolerates a broken/partial config."""
    from connectors import service as connector_service

    try:
        return connector_service.describe_settings()
    except Exception as exc:  # a malformed env var must not break the dashboard
        logger.warning("could not describe connector settings: %s", exc)
        return {"error": type(exc).__name__, "message": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Read-only: documents and chunks
# ---------------------------------------------------------------------------


@router.get("/documents", summary="Browse ingested documents")
async def documents(
    _: AdminDep,
    source: SourceName | None = None,
    q: str | None = Query(default=None, max_length=200),
    include_history: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    factory = _factory()
    clauses = ["TRUE"]
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source:
        clauses.append("d.source::text = :source")
        params["source"] = source
    if q:
        clauses.append(
            "(d.title ILIKE :q OR d.source_id ILIKE :q OR d.content ILIKE :q)"
        )
        params["q"] = f"%{q}%"
    if not include_history:
        clauses.append("d.is_current")
    where = " AND ".join(clauses)

    async with factory() as session:
        total = int(
            await session.scalar(
                text(f"SELECT count(*) FROM core.documents d WHERE {where}"),
                {k: v for k, v in params.items() if k != "limit" and k != "offset"},
            )
            or 0
        )
        rows = _rows(
            await session.execute(
                text(
                    "SELECT d.id, d.source::text AS source, d.source_id,"
                    " d.doc_type::text AS doc_type, d.title, d.author, d.url,"
                    " d.updated_at, d.is_current,"
                    " (SELECT count(*) FROM core.chunks c WHERE c.document_id = d.id)"
                    "   AS chunks"
                    f" FROM core.documents d WHERE {where}"
                    " ORDER BY d.updated_at DESC, d.id LIMIT :limit OFFSET :offset"
                ),
                params,
            )
        )
    for row in rows:
        row["id"] = str(row["id"])
        row["updated_at"] = _iso(row["updated_at"])
    return {"total": total, "limit": limit, "offset": offset, "documents": rows}


@router.get("/documents/{document_id}", summary="One document with its chunks")
async def document_detail(
    document_id: str,
    _: AdminDep,
    full: bool = False,
) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"not a document id: {document_id}") from exc

    factory = _factory()
    async with factory() as session:
        row = (
            await session.execute(
                text(
                    "SELECT d.id, d.source::text AS source, d.source_id,"
                    " d.parent_source_id, d.doc_type::text AS doc_type, d.title, d.content,"
                    " d.url, d.author, d.acl, d.content_hash, d.created_at, d.updated_at,"
                    " d.fetched_at, d.is_current"
                    " FROM core.documents d WHERE d.id = :id"
                ),
                {"id": parsed},
            )
        ).first()
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

        chunk_rows = _rows(
            await session.execute(
                text(
                    "SELECT c.id, c.seq, c.text, c.meta, c.content_hash, c.embedded_at,"
                    " (c.embedding IS NOT NULL) AS has_embedding"
                    " FROM core.chunks c WHERE c.document_id = :id ORDER BY c.seq"
                ),
                {"id": parsed},
            )
        )

    document = dict(row._mapping)
    document["id"] = str(document["id"])
    for key in ("created_at", "updated_at", "fetched_at"):
        document[key] = _iso(document[key])
    content = str(document.pop("content") or "")
    document["content_chars"] = len(content)
    document["content"] = content[:MAX_DOCUMENT_CHARS] if full else content[:MAX_TEXT_PREVIEW]

    chunks: list[dict[str, Any]] = []
    for chunk in chunk_rows:
        body = str(chunk.pop("text") or "")
        chunk["id"] = str(chunk["id"])
        chunk["embedded_at"] = _iso(chunk["embedded_at"])
        chunk["chars"] = len(body)
        chunk["text"] = body if full else body[:MAX_TEXT_PREVIEW]
        chunks.append(chunk)
    return {"document": document, "chunks": chunks}


# ---------------------------------------------------------------------------
# Read-only: traces
# ---------------------------------------------------------------------------


@router.get("/traces", summary="Recent request traces")
async def traces(_: AdminDep, limit: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
    factory = _factory()
    async with factory() as session:
        rows = _rows(
            await session.execute(
                text(
                    "SELECT trace_id, count(*) AS events,"
                    " min(created_at) AS started_at, max(created_at) AS ended_at,"
                    " array_agg(DISTINCT agent) AS agents,"
                    " coalesce(sum(latency_ms), 0) AS total_latency_ms"
                    " FROM audit.events GROUP BY trace_id"
                    " ORDER BY max(id) DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        )
    for row in rows:
        row["started_at"] = _iso(row["started_at"])
        row["ended_at"] = _iso(row["ended_at"])
        row["total_latency_ms"] = int(row["total_latency_ms"] or 0)
        row["agents"] = sorted(str(agent) for agent in (row["agents"] or []))
    return {"traces": rows}


@router.get("/traces/{trace_id}", summary="Every event of one trace, in order")
async def trace_detail(trace_id: str, _: AdminDep) -> dict[str, Any]:
    factory = _factory()
    events = await read_trace(factory, trace_id)
    if not events:
        raise HTTPException(status_code=404, detail="trace not found")
    items = [
        {
            "id": event.id,
            "agent": event.agent,
            "event": event.event,
            "latency_ms": event.latency_ms,
            "created_at": _iso(event.created_at),
            "payload": event.payload,
        }
        for event in events
    ]
    started = events[0].created_at
    ended = events[-1].created_at
    return {
        "trace_id": trace_id,
        "elapsed_ms": int((ended - started).total_seconds() * 1000),
        "total_latency_ms": sum(event.latency_ms or 0 for event in events),
        "events": items,
    }


# ---------------------------------------------------------------------------
# Retrieval + PII preview
# ---------------------------------------------------------------------------


class SearchIn(BaseModel):
    """One retrieval preview; optionally scrubbed exactly like the pipeline."""

    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=320)
    top_k: int = Field(default=8, ge=1, le=50)
    sources: list[SourceName] | None = None
    scrub: bool = False
    known_person_names: list[str] | None = None


@router.post("/search", summary="Run retrieval (and optionally PII scrubbing) on a query")
async def search(req: SearchIn, _: AdminDep) -> dict[str, Any]:
    from pii.scrubber import scrub
    from retrieval.errors import RetrievalConfigError
    from retrieval.search import RetrieveRequest, retrieve

    factory = _factory()
    started = time.perf_counter()
    try:
        pack = await retrieve(
            RetrieveRequest(
                query_text=req.query,
                sources=req.sources or None,
                top_k=req.top_k,
            ),
            user_id=req.user_id,
            session_factory=factory,
        )
    except RetrievalConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    chunks: list[dict[str, Any]] = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "source_id": chunk.source_id,
            "title": chunk.title,
            "url": chunk.url,
            "score": chunk.score,
            "updated_at": _iso(chunk.updated_at),
            "stale": chunk.stale,
            "text": chunk.text,
            "meta": chunk.meta,
            "redacted_types": [],
        }
        for chunk in pack.chunks
    ]
    redaction_count = 0
    if req.scrub and pack.chunks:
        names = req.known_person_names or _fixture_person_names()
        scrubbed = await scrub(
            pack,
            trace_id=None,
            known_person_names=names,
            session_factory=factory,
        )
        redaction_count = scrubbed.redaction_count
        for chunk, clean in zip(chunks, scrubbed.chunks, strict=True):
            chunk["text"] = clean.text
            chunk["redacted_types"] = clean.redacted_types

    return {
        "query": req.query,
        "user_id": req.user_id,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "degraded_reason": pack.degraded_reason,
        "dropped_below_floor": pack.dropped_below_floor,
        "redaction_count": redaction_count,
        "chunks": chunks,
    }


def _fixture_person_names() -> list[str]:
    """Dev convenience: the fixture corpus roster, when the corpus is present."""
    from evals.fixtures import FIXTURE_DIR, FixtureCorpusError, load_fixture_corpus

    if not FIXTURE_DIR.is_dir():
        return []
    try:
        corpus = load_fixture_corpus()
    except FixtureCorpusError:
        return []
    names = corpus.manifest.get("pii", {}).get("kinds", {}).get("person", [])
    return [str(name) for name in names]


# ---------------------------------------------------------------------------
# Mutating: sync, backfill, seed
# ---------------------------------------------------------------------------


class SyncIn(BaseModel):
    source: SourceName = "jira"
    full: bool = False


@router.post("/sync", summary="Run one connector sync")
async def sync(req: SyncIn, _: AdminDep) -> dict[str, Any]:
    from connectors import service as connector_service
    from connectors.base import ConnectorAuthError, ConnectorConfigError, ConnectorError
    from connectors.contracts import SyncRequest

    config = _connector_settings()
    if req.source == "jira" and not config.get("jira_configured"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Jira is not configured: set JIRA_BASE_URL, JIRA_EMAIL and "
                "JIRA_API_TOKEN (see .env.example)"
            ),
        )
    if req.source == "gmail" and not config.get("gmail_configured"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Gmail is not configured: set GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
                "(a Google app password — see .env.example)"
            ),
        )
    if req.source == "notion":
        raise HTTPException(
            status_code=501,
            detail="the notion connector is not implemented yet (M2)",
        )
    _factory()  # a sync persists documents; fail early with a clear message

    async with _sync_lock:
        try:
            async with asyncio.timeout(SYNC_TIMEOUT_S):
                result = await connector_service.sync(
                    SyncRequest(source=req.source, full=req.full)
                )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"sync did not finish within {SYNC_TIMEOUT_S:.0f}s; it may still be "
                    "running. Re-running is safe — syncs are idempotent."
                ),
            ) from exc
        except ConnectorAuthError as exc:
            raise HTTPException(
                status_code=502, detail=f"{req.source} rejected the credentials: {exc}"
            ) from exc
        except ConnectorConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConnectorError as exc:
            raise HTTPException(status_code=502, detail=f"sync failed: {exc}") from exc

    payload = result.model_dump(mode="json")
    payload["trace_id"] = getattr(result, "trace_id", None)
    return payload


@router.post("/cache/invalidate", summary="Clear and re-version the answer cache")
async def cache_invalidate(_: AdminDep) -> dict[str, Any]:
    """Drop every cached answer and bump the data version.

    ``clear`` deletes the answer entries; ``bump_version`` guarantees no entry
    written against an older version can ever be read again, even if ``clear``
    missed a key mid-write. No database required.
    """
    from memory.answer_cache import get_answer_cache

    cache = get_answer_cache(get_settings())
    if cache is None:
        return {
            "invalidated": False,
            "reason": "no answer cache configured (REDIS_URL is unset)",
        }
    try:
        cleared = await cache.clear()
        version = await cache.bump_version()
    except Exception as exc:  # a cache problem must not crash the dashboard
        raise HTTPException(status_code=502, detail=f"cache invalidation failed: {exc}") from exc
    return {"invalidated": True, "cleared": cleared, "version": version}


class BackfillIn(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    only_missing: bool = False
    force: bool = False
    source: SourceName | None = None
    embedder: Literal["auto", "fake"] = "auto"


@router.post("/backfill", summary="Chunk and embed documents (bounded run)")
async def backfill(req: BackfillIn, _: AdminDep) -> dict[str, Any]:
    from indexing.embedder import build_embedder
    from indexing.errors import IndexingConfigError
    from indexing.indexer import Indexer, IndexRequest, get_indexer
    from memory.answer_cache import invalidate_answer_cache

    factory = _factory()
    ids = await _select_document_ids(
        factory, source=req.source, only_missing=req.only_missing, limit=req.limit
    )
    if not ids:
        return {
            "documents": 0,
            "chunks_written": 0,
            "chunks_skipped": 0,
            "embed_tokens": 0,
            "pages": 0,
            "answer_cache_version": None,
            "note": "no matching documents",
        }

    settings = get_settings()
    embedder = None
    indexer: Indexer
    if req.embedder == "fake":
        embedder = build_embedder(settings, kind="fake")
        indexer = Indexer(session_factory=factory, embedder=embedder)
    else:
        indexer = get_indexer()

    started = time.perf_counter()
    totals: dict[str, Any] = {
        "documents": 0,
        "chunks_written": 0,
        "chunks_skipped": 0,
        "embed_tokens": 0,
        "pages": 0,
    }
    try:
        async with _backfill_lock, asyncio.timeout(BACKFILL_TIMEOUT_S):
            for start in range(0, len(ids), BACKFILL_PAGE_SIZE):
                page = ids[start : start + BACKFILL_PAGE_SIZE]
                result = await indexer.run(
                    IndexRequest(document_ids=page, force_reembed=req.force)
                )
                totals["documents"] += len(page)
                totals["chunks_written"] += result.chunks_written
                totals["chunks_skipped"] += result.chunks_skipped
                totals["embed_tokens"] += result.embed_tokens
                totals["pages"] += 1
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "message": (
                    f"backfill stopped after {BACKFILL_TIMEOUT_S:.0f}s; it is idempotent, "
                    "re-run to continue"
                ),
                "progress": totals,
            },
        ) from exc
    except IndexingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if embedder is not None:
            await embedder.aclose()

    totals["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    totals["embedder"] = req.embedder
    if totals["chunks_written"] or req.force:
        # New or re-embedded chunks can change what a repeated question answers.
        totals["answer_cache_version"] = await invalidate_answer_cache()
    return totals


async def _select_document_ids(
    factory: async_sessionmaker[AsyncSession],
    *,
    source: SourceName | None,
    only_missing: bool,
    limit: int,
) -> list[uuid.UUID]:
    sql = "SELECT d.id FROM core.documents d WHERE d.is_current"
    params: dict[str, Any] = {"limit": limit}
    if source:
        sql += " AND d.source::text = :source"
        params["source"] = source
    if only_missing:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM core.chunks c WHERE c.document_id = d.id)"
        )
    sql += " ORDER BY d.id LIMIT :limit"
    async with factory() as session:
        return [uuid.UUID(str(row[0])) for row in (await session.execute(text(sql), params)).all()]


class SeedIn(BaseModel):
    confirm: bool = False


@router.post("/seed", summary="Load the synthetic fixture corpus into this database")
async def seed(req: SeedIn, _: AdminDep) -> dict[str, Any]:
    """Dev-only convenience: makes an empty dashboard demonstrate something.

    Writes ``fixtures/corpus`` with deterministic ids (idempotent). Refused in
    prod, and requires ``confirm: true`` so a stray click cannot overwrite a
    database that was synced from a real tenant.
    """
    from evals.fixtures import FixtureCorpusError, load_fixture_corpus, seed_fixture_corpus
    from memory.answer_cache import invalidate_answer_cache

    settings = get_settings()
    if settings.app_env == "prod":
        raise HTTPException(status_code=403, detail="refusing to seed fixtures in prod")
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail=(
                "seeding overwrites documents that share these deterministic ids; "
                "send confirm=true"
            ),
        )
    factory = _factory()
    try:
        corpus = load_fixture_corpus()
    except FixtureCorpusError as exc:
        raise HTTPException(status_code=404, detail=f"no fixture corpus available: {exc}") from exc
    result = await seed_fixture_corpus(factory, corpus)
    await invalidate_answer_cache()
    return {
        "documents_seeded": result.documents_seeded,
        "ids": [str(item) for item in result.ids[:5]],
        "next": "run POST /admin/backfill to chunk and embed them",
    }


__all__ = ["require_admin", "router"]
