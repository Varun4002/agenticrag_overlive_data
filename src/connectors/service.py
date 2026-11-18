"""Thin orchestration over the source connectors.

The service is intentionally boring: it picks the adapter for ``req.source``,
supplies the database session the adapter needs, commits, and returns the
adapter's :class:`SyncResult`. All sync behaviour lives in the adapters; all
DB-write logic lives in ``connectors.normalizer``.

It also owns the connector's **audit trail** (docs/08-decisions.md, integration
gap #2). Three events, all with ``agent="connector"``:

- ``connector.sync.started``   — source, mode
- ``connector.sync.completed`` — counts, watermark, ``partial`` flag, latency
- ``connector.sync.failed``    — error type/message, ``retryable``, latency
  (targeted fetches emit the same trio under ``connector.fetch.*``)

Callers (CLI now, orchestrator/worker later) use exactly two functions:

    await sync(SyncRequest(source="jira", full=False))
    await fetch_targeted("jira", source_id="PHX-101")

Both accept an optional ``trace_id`` (generated when absent) and return it, so a
sync can be correlated with the request that triggered it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from connectors.base import ConnectorConfigError, ConnectorError
from connectors.contracts import Connector, SyncRequest, SyncResult
from connectors.gmail import GmailConnector
from connectors.jira import JiraConnector
from connectors.normalizer import CanonicalDoc, SqlAlchemyDocumentStore
from connectors.notion import NotionConnector
from connectors.settings import ConnectorSettings, get_connector_settings, split_csv
from db import session as db_session
from db.tracing import Tracer, make_sink
from memory.answer_cache import invalidate_answer_cache

logger = logging.getLogger(__name__)

SUPPORTED_SOURCES: tuple[str, ...] = ("jira", "notion", "gmail")
# Emitting component name in ``audit.events`` (matches docs/02-architecture.md).
AGENT = "connector"
_MAX_ERROR_CHARS = 300


async def sync(
    req: SyncRequest,
    *,
    trace_id: str | None = None,
    tracer: Tracer | None = None,
    connector: Connector | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    settings: ConnectorSettings | None = None,
) -> SyncResult:
    """Run one sync for ``req.source`` and record it in ``audit.events``.

    ``settings``/``transport``/``session_factory``/``tracer`` exist so the CLI and
    tests can inject configuration and fakes; production callers pass nothing. A
    :class:`ConnectorAuthError` propagates (it is a configuration problem for a
    human, not a partial sync), every other per-item failure is reported inside
    the returned :class:`SyncResult` — and the audit event says ``partial``.
    """
    tracer = tracer or Tracer(
        make_sink(session_factory or db_session.get_session_factory()), trace_id
    )
    started = time.perf_counter()
    await tracer.emit(AGENT, "connector.sync.started", {"source": req.source, "full": req.full})

    try:
        result = await _dispatch_sync(
            req,
            connector=connector,
            transport=transport,
            session_factory=session_factory,
            settings=settings,
        )
    except Exception as exc:
        await tracer.emit(
            AGENT,
            "connector.sync.failed",
            {
                "source": req.source,
                "full": req.full,
                **_error_payload(exc),
            },
            latency_ms=_elapsed_ms(started),
        )
        raise

    # Invalidate cached answers only when the corpus actually changed: a
    # scheduled no-op sync must not throw away a warm cache. Failure here is
    # logged, never fatal to a sync that already committed.
    cache_version: int | None = None
    if result.documents_upserted or result.documents_retired:
        try:
            cache_version = await invalidate_answer_cache()
        except Exception as exc:  # the sync is committed; a cache must not undo that
            logger.warning(
                "answer-cache invalidation raised after a committed sync: %s", exc
            )
            cache_version = None

    await tracer.emit(
        AGENT,
        "connector.sync.completed",
        {
            "source": result.source,
            "full": result.full,
            "partial": not result.ok,
            "documents_upserted": result.documents_upserted,
            "documents_unchanged": result.documents_unchanged,
            "documents_retired": result.documents_retired,
            "documents_skipped": result.documents_skipped,
            "watermark": result.watermark.isoformat(),
            "errors": len(result.errors),
            "error_types": sorted({error.error_type for error in result.errors}),
            "requests": result.requests,
            "retries": result.retries,
            "answer_cache_version": cache_version,
        },
        latency_ms=_elapsed_ms(started),
    )
    result.trace_id = tracer.trace_id
    return result


async def _dispatch_sync(
    req: SyncRequest,
    *,
    connector: Connector | None,
    transport: httpx.AsyncBaseTransport | None,
    session_factory: async_sessionmaker[AsyncSession] | None,
    settings: ConnectorSettings | None,
) -> SyncResult:
    if connector is not None:
        return await connector.sync(req)

    config = settings or get_connector_settings()
    if req.source == "jira":
        return await _sync_jira(
            req, config=config, transport=transport, session_factory=session_factory
        )
    if req.source == "gmail":
        return await _sync_gmail(req, config=config, session_factory=session_factory)

    stub = _build_stub(req.source, config)
    try:
        return await stub.sync(req)
    finally:
        await stub.aclose()


async def fetch_targeted(
    source: str,
    source_id: str | None = None,
    *,
    query: str | None = None,
    limit: int = 10,
    trace_id: str | None = None,
    tracer: Tracer | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    settings: ConnectorSettings | None = None,
) -> list[CanonicalDoc]:
    """Ad-hoc fetch for freshness ("due today"), used by the planner when data is stale.

    Returns canonical documents and writes nothing to ``core.documents``:
    persisting a targeted fetch is a sync decision, not a fetch decision. The
    fetch itself *is* audited (``connector.fetch.*``), because "the planner
    re-read live data" is exactly the kind of decision a trace must explain.
    """
    tracer = tracer or Tracer(
        make_sink(session_factory or db_session.get_session_factory()), trace_id
    )
    requested: dict[str, Any] = {
        "source": source,
        "source_id": source_id,
        "query": query,
        "limit": limit,
    }
    started = time.perf_counter()
    await tracer.emit(AGENT, "connector.fetch.started", dict(requested))

    connector = build_connector(source, transport=transport, settings=settings)
    try:
        docs = await connector.fetch_targeted(source_id=source_id, query=query, limit=limit)
    except Exception as exc:
        await tracer.emit(
            AGENT,
            "connector.fetch.failed",
            {**requested, **_error_payload(exc)},
            latency_ms=_elapsed_ms(started),
        )
        raise
    finally:
        await connector.aclose()

    payload: dict[str, Any] = {**requested, "documents": len(docs)}
    stats = getattr(connector, "stats", None)
    if stats is not None:
        payload.update(stats.as_payload())
    await tracer.emit(
        AGENT, "connector.fetch.completed", payload, latency_ms=_elapsed_ms(started)
    )
    return docs


def build_connector(
    source: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    settings: ConnectorSettings | None = None,
) -> Connector:
    """Construct a read-only connector with no database wiring (fetch-only use)."""
    config = settings or get_connector_settings()
    if source == "jira":
        return JiraConnector(settings=config, transport=transport)
    if source == "gmail":
        # No store: a targeted fetch reads and returns, it never persists.
        return GmailConnector(store=None, settings=config)
    return _build_stub(source, config)


async def _sync_jira(
    req: SyncRequest,
    *,
    config: ConnectorSettings,
    transport: httpx.AsyncBaseTransport | None,
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> SyncResult:
    factory = session_factory or db_session.get_session_factory()
    if factory is None:
        raise ConnectorConfigError(
            "DATABASE_URL is not set: a Jira sync persists canonical documents and its "
            "watermark, so it cannot run without a database"
        )

    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        jira = JiraConnector(store=store, settings=config, transport=transport)
        try:
            result = await jira.sync(req)
        except ConnectorError:
            # sync() records the terminal failure in core.sync_state just before
            # raising; commit that so operators see "error" instead of a silent gap.
            await _commit_best_effort(session)
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await jira.aclose()
        await session.commit()
        return result


async def _sync_gmail(
    req: SyncRequest,
    *,
    config: ConnectorSettings,
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> SyncResult:
    # Credentials first: a caller with neither a database nor a password should
    # get the actionable error (what to put in the environment), not a DB one.
    config.require_gmail()
    factory = session_factory or db_session.get_session_factory()
    if factory is None:
        raise ConnectorConfigError(
            "DATABASE_URL is not set: a Gmail sync persists canonical documents and its "
            "watermark, so it cannot run without a database"
        )

    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        gmail = GmailConnector(store, config)
        try:
            result = await gmail.sync(req)
        except ConnectorError:
            # sync() records the terminal failure in core.sync_state just before
            # raising; commit that so operators see "error" instead of a silent gap.
            await _commit_best_effort(session)
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await gmail.aclose()
        await session.commit()
        return result


async def _commit_best_effort(session: AsyncSession) -> None:
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        logger.warning("could not persist sync failure state: %s", exc)
        await session.rollback()


def _build_stub(source: str, config: ConnectorSettings) -> Connector:
    """Notion placeholder; raises ``NotImplementedError`` on use."""
    if source == "notion":
        return NotionConnector(database_ids=split_csv(config.notion_database_ids))
    raise ConnectorConfigError(
        f"unknown source {source!r}; supported sources are {', '.join(SUPPORTED_SOURCES)}"
    )


def describe_settings(settings: ConnectorSettings | None = None) -> dict[str, Any]:
    """Non-secret view of connector config, for startup logs and diagnostics."""
    config = settings or get_connector_settings()
    return {
        "jira_configured": bool(
            config.jira_base_url and config.jira_email and config.jira_api_token
        ),
        "jira_base_url": config.jira_base_url or None,
        "jira_projects": list(config.allowed_projects),
        "jira_search_path": config.jira_search_path,
        "gmail_configured": config.gmail_configured,
        "gmail_labels": list(config.allowed_gmail_labels),
        # The address is not a secret (it is the ACL identity); the app password is
        # a SecretStr and never appears here.
        "gmail_address": config.gmail_address or None,
    }


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """Failure fields for ``connector.*.failed``: no secrets, no full bodies."""
    message = str(exc).replace("\n", " ").strip()[:_MAX_ERROR_CHARS]
    return {
        "error_type": type(exc).__name__,
        "error": message or type(exc).__name__,
        "retryable": bool(getattr(exc, "retryable", False)),
    }


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


__all__ = [
    "AGENT",
    "SUPPORTED_SOURCES",
    "build_connector",
    "describe_settings",
    "fetch_targeted",
    "sync",
]
