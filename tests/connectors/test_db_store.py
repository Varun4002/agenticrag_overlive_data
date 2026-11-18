"""SQLAlchemy store tests against a **scratch** Postgres database.

Skips unless ``CONNECTOR_TEST_DATABASE_URL`` points at a throwaway database (see
``src/connectors/HANDOFF.md`` for the two commands that prepare one):

    sudo -u postgres createdb -O rag agentic_rag_conn_test
    DATABASE_URL=...agentic_rag_conn_test ... .venv/bin/alembic upgrade head

It refuses to run against the configured dev database: these tests truncate
``core.documents``/``core.sync_state`` and must never touch shared data.
"""

from __future__ import annotations

import copy
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from jira_fixtures import FIXED_NOW, JiraStub, issue_from, load_json, make_settings
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from config import Settings
from connectors import service
from connectors.contracts import SyncRequest
from connectors.jira import JiraConnector
from connectors.normalizer import (
    CanonicalDoc,
    InMemoryDocumentStore,
    SqlAlchemyDocumentStore,
    SyncStateRecord,
    UpsertOutcome,
    apply_document,
    jira_issue_to_docs,
)
from db.models import AuditEvent, DocType, Document, Source, SyncState

SCRATCH_URL_ENV = "CONNECTOR_TEST_DATABASE_URL"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = os.environ.get(SCRATCH_URL_ENV)
    if not url:
        pytest.skip(
            f"{SCRATCH_URL_ENV} is not set — skipping Postgres-backed connector tests "
            "(see src/connectors/HANDOFF.md)"
        )
    # Two independent sources: tests/db uses DATABASE_URL, the app reads .env.
    dev_urls = {os.environ.get("DATABASE_URL", ""), Settings().database_url or ""}
    for dev_url in dev_urls:
        if dev_url and _database_name(url) == _database_name(dev_url):
            pytest.fail(
                f"{SCRATCH_URL_ENV} points at the dev database ({_database_name(url)}); "
                "these tests truncate tables and must run on a scratch database"
            )

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            present = await connection.execute(text("SELECT to_regclass('core.documents')"))
            if present.scalar_one() is None:
                pytest.skip(
                    "scratch database is not migrated — run "
                    "`DATABASE_URL=<scratch> .venv/bin/alembic upgrade head`"
                )
        async with engine.begin() as connection:
            await connection.execute(text("TRUNCATE core.documents, core.sync_state CASCADE"))
    except Exception as exc:  # pragma: no cover - environment problem, not a code path
        await engine.dispose()
        pytest.skip(f"{SCRATCH_URL_ENV} is unreachable ({type(exc).__name__}: {exc})")

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield session_factory
    finally:
        await engine.dispose()


def _database_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


def doc_for(
    issue: dict[str, object], *, base_url: str = "https://acme.atlassian.net"
) -> CanonicalDoc:
    return jira_issue_to_docs(issue, base_url=base_url, fetched_at=FIXED_NOW)[0]


async def test_sql_store_creates_supersedes_and_retires(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    issue_v1 = issue_from("search_page_1.json", 0)
    issue_v2 = issue_from("search_updated_page.json", 0)

    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)

        created = await apply_document(store, doc_for(issue_v1))
        assert created.outcome is UpsertOutcome.created

        unchanged = await apply_document(store, doc_for(issue_v1))
        assert unchanged.outcome is UpsertOutcome.unchanged
        assert unchanged.document_id == created.document_id

        updated = await apply_document(store, doc_for(issue_v2))
        assert updated.outcome is UpsertOutcome.updated
        assert updated.superseded_id == created.document_id
        await session.commit()

    async with factory() as session:
        rows = (
            await session.execute(select(Document).order_by(Document.created_at))
        ).scalars().all()
        assert len(rows) == 2
        current = [row for row in rows if row.is_current]
        assert len(current) == 1
        assert "Status: Done" in current[0].content
        retired = [row for row in rows if not row.is_current]
        # History rows keep the native id; only current rows are uniquely indexed.
        assert retired[0].source_id == "PHX-101"
        assert retired[0].acl["groups"] == ["jira:project:PHX"]
        assert len(retired[0].content_hash) == 64

    # A third revision: allowed by the partial unique index, no id rewriting.
    issue_v3 = copy.deepcopy(issue_v2)
    issue_v3["fields"]["summary"] = "Write connector retry/backoff policy (follow-up)"
    issue_v3["fields"]["updated"] = "2026-11-18T10:00:00.000+0000"
    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        third = await apply_document(store, doc_for(issue_v3))
        assert third.outcome is UpsertOutcome.updated
        await session.commit()

    async with factory() as session:
        history = (
            await session.execute(
                select(Document).where(Document.source_id.like("PHX-101%"))
            )
        ).scalars().all()
        assert len(history) == 3
        assert sum(1 for row in history if row.is_current) == 1
        # Every revision carries the native source_id.
        assert {row.source_id for row in history} == {"PHX-101"}


async def test_sql_store_sync_state_roundtrip(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        assert (await store.load_sync_state("jira")).watermark is None

        first = FIXED_NOW - timedelta(hours=2)
        await store.save_sync_state(_state("jira", first, "ok"))
        await session.commit()

    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        loaded = await store.load_sync_state("jira")
        assert loaded.watermark == first
        assert loaded.last_status == "ok"

        await store.save_sync_state(_state("jira", FIXED_NOW, "partial", error="boom"))
        await session.commit()

    async with factory() as session:
        row = (
            await session.execute(select(SyncState).where(SyncState.source == "jira"))
        ).scalar_one()
        assert row.watermark == FIXED_NOW
        assert row.last_status == "partial"
        assert row.error == "boom"
        assert row.last_run_at is not None


async def test_jira_full_sync_against_postgres(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    stub = (
        JiraStub()
        .with_search_payloads(load_json("search_page_1.json"), load_json("search_page_2.json"))
        .with_fixture_comments("PHX-101")
        .with_fixture_comments("PHX-102")
        .with_fixture_comments("PHX-103")
    )
    async with factory() as session:
        jira = JiraConnector(
            store=SqlAlchemyDocumentStore(session),
            settings=make_settings(),
            transport=stub.transport,
            sleep=_no_sleep,
            now=lambda: FIXED_NOW,
        )
        try:
            result = await jira.sync(SyncRequest(source="jira"))
        finally:
            await jira.aclose()
        await session.commit()

    assert result.errors == []
    assert result.documents_upserted == 6
    assert result.watermark == datetime(2026, 11, 16, 14, 22, tzinfo=UTC)

    async with factory() as session:
        rows = (
            await session.execute(
                select(Document).where(Document.is_current.is_(True)).order_by(Document.source_id)
            )
        ).scalars().all()
        assert [row.source_id for row in rows] == [
            "PHX-101",
            "PHX-101/comment/20101",
            "PHX-101/comment/20102",
            "PHX-102",
            "PHX-103",
            "PHX-103/comment/20103",
        ]
        parents = {row.source_id: row.parent_source_id for row in rows}
        assert parents["PHX-101"] is None
        assert parents["PHX-101/comment/20101"] == "PHX-101"
        acl = next(row.acl for row in rows if row.source_id == "PHX-101")
        assert acl["groups"] == ["jira:project:PHX"]
        assert "email:ada@example.com" in acl["users"]

        state = (await session.execute(select(SyncState))).scalar_one()
        assert state.last_status == "ok"


async def test_inmemory_and_sql_stores_agree(factory: async_sessionmaker[AsyncSession]) -> None:
    """The SQL store and the in-memory reference store must make the same calls."""
    issue_v1 = issue_from("search_page_1.json", 0)
    issue_v2 = issue_from("search_updated_page.json", 0)

    memory = InMemoryDocumentStore()
    async with factory() as session:
        sql = SqlAlchemyDocumentStore(session)
        for store in (memory, sql):
            first = await apply_document(store, doc_for(issue_v1))
            second = await apply_document(store, doc_for(issue_v2))
            assert first.outcome is UpsertOutcome.created
            assert second.outcome is UpsertOutcome.updated
        await session.commit()

    assert len(memory.rows) == 2
    async with factory() as session:
        rows = (
            await session.execute(
                select(Document).where(Document.source_id.like("PHX-101%"))
            )
        ).scalars().all()
        assert len(rows) == 2
        assert sum(1 for row in rows if row.is_current) == 1


async def test_partial_unique_index_allows_history_but_one_current(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """Schema contract from migration 0003 (docs/08-decisions.md #2).

    (a) one current row per ``(source, source_id)``; (b) any number of
    ``is_current=false`` history rows under the same native id.
    """
    def row(*, title: str, is_current: bool, content_hash: str) -> Document:
        return Document(
            source=Source.jira,
            source_id="PHX-777",
            doc_type=DocType.issue,
            title=title,
            content="c",
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
            fetched_at=FIXED_NOW,
            acl={},
            content_hash=content_hash,
            is_current=is_current,
        )

    async with factory() as session:
        session.add(row(title="current", is_current=True, content_hash="h"))
        for index in range(3):
            session.add(row(title=f"history {index}", is_current=False, content_hash=f"h{index}"))
        await session.commit()

    async with factory() as session:
        count = (
            await session.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.source_id == "PHX-777")
            )
        ).scalar_one()
        current_title = (
            await session.execute(
                select(Document.title).where(
                    Document.source_id == "PHX-777", Document.is_current.is_(True)
                )
            )
        ).scalar_one()
    assert count == 4
    assert current_title == "current"

    # A second *current* row for the same item is rejected by the partial index.
    async with factory() as session:
        session.add(row(title="conflict", is_current=True, content_hash="h9"))
        with pytest.raises(IntegrityError, match="uq_documents_current_source_id"):
            await session.commit()
        await session.rollback()


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "src/db/migrations"))
    return config


async def test_migration_0003_roundtrip_keeps_history(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """downgrade -1 / upgrade head must not lose document versions."""
    url = os.environ.get(SCRATCH_URL_ENV)
    assert url is not None  # the factory fixture guarantees this

    # Three revisions sharing the native source_id (1 current + 2 history).
    issue_v1 = issue_from("search_page_1.json", 0)
    issue_v2 = issue_from("search_updated_page.json", 0)
    issue_v3 = copy.deepcopy(issue_v2)
    issue_v3["fields"]["summary"] = "Write connector retry/backoff policy (follow-up)"
    issue_v3["fields"]["updated"] = "2026-11-18T10:00:00.000+0000"

    async with factory() as session:
        store = SqlAlchemyDocumentStore(session)
        for issue in (issue_v1, issue_v2, issue_v3):
            await apply_document(store, doc_for(issue))
        await session.commit()

    config = _alembic_config()
    with patch.dict(os.environ, {"DATABASE_URL": url}):
        command.downgrade(config, "0002_local_embedding")

        # Nothing lost: the old constraint only allows one plain-id history row,
        # so the older revision is re-suffixed for the trip down.
        async with factory() as session:
            rows = (
                await session.execute(
                    select(Document.source_id, Document.is_current).where(
                        Document.source_id.like("PHX-101%")
                    )
                )
            ).all()
            assert len(rows) == 3
            assert sum(1 for _, is_current in rows if is_current) == 1
            assert sum(1 for source_id, _ in rows if "#history-" in source_id) == 1

        command.upgrade(config, "head")

    async with factory() as session:
        rows = (
            await session.execute(
                select(Document.source_id, Document.title).where(
                    Document.source_id.like("PHX-101%")
                )
            )
        ).all()
        current_title = (
            await session.execute(
                select(Document.title).where(
                    Document.source_id == "PHX-101", Document.is_current.is_(True)
                )
            )
        ).scalar_one()

    assert len(rows) == 3
    assert {source_id for source_id, _ in rows} == {"PHX-101"}  # suffix stripped again
    assert "follow-up" in current_title


async def test_service_sync_commits_against_postgres(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The service wrapper (session, commit, store wiring) end to end."""
    stub = (
        JiraStub()
        .with_search_payloads(load_json("search_page_1.json"), load_json("search_page_2.json"))
        .with_fixture_comments("PHX-101")
        .with_fixture_comments("PHX-102")
        .with_fixture_comments("PHX-103")
    )
    settings = make_settings()
    result = await service.sync(
        SyncRequest(source="jira"),
        transport=stub.transport,
        session_factory=factory,
        settings=settings,
    )

    assert result.errors == []
    assert result.documents_upserted == 6
    assert result.full is True

    # Audit trail lands in the real ``audit.events`` table (integration gap #2).
    assert result.trace_id is not None
    async with factory() as session:
        events = (
            await session.execute(
                select(AuditEvent.event, AuditEvent.agent, AuditEvent.latency_ms)
                .where(AuditEvent.trace_id == result.trace_id)
                .order_by(AuditEvent.id)
            )
        ).all()
    assert [event for event, _, _ in events] == [
        "connector.sync.started",
        "connector.sync.completed",
    ]
    assert {agent for _, agent, _ in events} == {"connector"}
    assert events[1][2] is not None

    # Second run in a fresh session: the first run's rows are visible and skipped.
    again = await service.sync(
        SyncRequest(source="jira", full=True),
        transport=stub.transport,
        session_factory=factory,
        settings=settings,
    )
    assert again.documents_upserted == 0
    assert again.documents_unchanged == 6
    assert again.errors == []


def _state(
    source: str, watermark: datetime, status: str, *, error: str | None = None
) -> SyncStateRecord:
    return SyncStateRecord(
        source=source, watermark=watermark, last_run_at=FIXED_NOW, last_status=status, error=error
    )


async def _no_sleep(_seconds: float) -> None:
    return None
