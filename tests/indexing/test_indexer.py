"""Indexer tests against a real scratch Postgres (skipped without one).

Covers the contract, idempotent upserts, embedding-cache behaviour, supersession,
per-batch transactions, and an end-to-end pass over the whole fixture corpus.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import EMBEDDING_DIMENSIONS
from db.models import Document as DocumentRow
from evals.fixtures import fixture_document_ids, load_fixture_corpus, seed_fixture_corpus
from indexing.chunker import SourceDocument, chunk_document
from indexing.embedder import DeterministicFakeEmbedder, EmbeddingResult
from indexing.errors import EmbeddingServerError, IndexingConfigError
from indexing.indexer import Indexer, IndexRequest, IndexResult, chunk_id_for
from indexing.tokens import sha256_text

pytestmark = pytest.mark.usefixtures("clean_index_db")

CORPUS = load_fixture_corpus()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class RecordingEmbedder:
    """Deterministic vectors plus a log of every text it was asked to embed."""

    max_batch_size: int = 128
    dimensions: int = EMBEDDING_DIMENSIONS
    model: str = "recording-fake"
    calls: list[list[str]] = field(default_factory=list)
    _inner: DeterministicFakeEmbedder = field(init=False)

    def __post_init__(self) -> None:
        self._inner = DeterministicFakeEmbedder(dimensions=self.dimensions)

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        return await self._inner.embed(texts)

    async def aclose(self) -> None:
        return None


@dataclass
class FailingEmbedder(RecordingEmbedder):
    """Fails one chosen call (1-based) to exercise partial-batch writes."""

    fail_on_call: int = 1

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingServerError("simulated provider failure")
        return await self._inner.embed(texts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_PARAGRAPH = (
    "The billing platform team reviewed the cutover plan again and confirmed that the "
    "dual-write window can start on the agreed date, provided the vendor schema is final."
)


def issue_content(
    *,
    key: str = "PHX-900",
    status: str = "In Progress",
    due: str = "2026-11-20",
    body: str = "Description body for the synthetic issue.",
) -> str:
    return (
        f"Key: {key}\n"
        f"Summary: Synthetic issue {key}\n"
        "Type: Task\n"
        f"Status: {status}\n"
        "Priority: High\n"
        "Project: Project Phoenix (PHX)\n"
        "Assignee: Daniel Okonkwo\n"
        f"Due date: {due}\n"
        "Labels: synthetic\n\n"
        f"{body}"
    )


async def insert_document(
    factory: async_sessionmaker[AsyncSession],
    *,
    source_id: str = "PHX-900",
    content: str | None = None,
    title: str | None = None,
    is_current: bool = True,
    labels: list[str] | None = None,
    doc_type: str = "issue",
    source: str = "jira",
) -> uuid.UUID:
    document_id = uuid.uuid4()
    payload = content if content is not None else issue_content(key=source_id)
    async with factory() as session, session.begin():
        await session.execute(
            insert(DocumentRow).values(
                id=document_id,
                source=source,
                source_id=source_id,
                doc_type=doc_type,
                title=title or f"Synthetic {source_id}",
                content=payload,
                url=f"https://example.test/{source_id}",
                author="Daniel Okonkwo",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
                updated_at=datetime(2026, 11, 1, tzinfo=UTC),
                acl={"labels": labels or ["phoenix"]},
                content_hash=sha256_text(payload),
                is_current=is_current,
            )
        )
    return document_id


async def chunk_rows(
    factory: async_sessionmaker[AsyncSession], document_id: uuid.UUID
) -> list[dict[str, Any]]:
    async with factory() as session:
        rows = await session.execute(
            text(
                "SELECT id, seq, text, content_hash, meta, embedding IS NOT NULL AS has_vector, "
                "embedded_at FROM core.chunks WHERE document_id = :doc ORDER BY seq"
            ).bindparams(doc=document_id)
        )
        return [dict(row._mapping) for row in rows]


async def audit_events(
    factory: async_sessionmaker[AsyncSession], event: str = "indexer.completed"
) -> list[dict[str, Any]]:
    async with factory() as session:
        rows = await session.execute(
            text(
                "SELECT agent, event, payload, latency_ms FROM audit.events "
                "WHERE event = :event ORDER BY id"
            ).bindparams(event=event)
        )
        return [dict(row._mapping) for row in rows]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_contract_models_roundtrip() -> None:
    document_id = uuid.uuid4()
    request = IndexRequest(document_ids=[document_id], force_reembed=True)
    assert request.model_dump(mode="json") == {
        "document_ids": [str(document_id)],
        "force_reembed": True,
    }
    assert IndexRequest(document_ids=[document_id]).force_reembed is False

    result = IndexResult(chunks_written=3, chunks_skipped=2, embed_tokens=41)
    assert result.model_dump() == {"chunks_written": 3, "chunks_skipped": 2, "embed_tokens": 41}


def test_chunk_id_is_deterministic() -> None:
    document_id = uuid.uuid4()
    assert chunk_id_for(document_id, 0) == chunk_id_for(str(document_id), 0)
    assert chunk_id_for(document_id, 0) != chunk_id_for(document_id, 1)
    assert chunk_id_for(document_id, 0) != chunk_id_for(uuid.uuid4(), 0)


# ---------------------------------------------------------------------------
# Indexing behaviour
# ---------------------------------------------------------------------------


async def test_index_writes_chunks_with_vectors_and_hashes(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    embedder = RecordingEmbedder()
    result = await Indexer(session_factory=index_session_factory, embedder=embedder).run(
        IndexRequest(document_ids=[document_id])
    )

    rows = await chunk_rows(index_session_factory, document_id)
    assert result.chunks_written == len(rows) >= 2
    assert result.chunks_skipped == 0
    assert result.embed_tokens > 0
    assert [row["seq"] for row in rows] == list(range(len(rows)))
    for row in rows:
        assert row["has_vector"] is True
        assert row["embedded_at"] is not None
        assert row["content_hash"] == sha256_text(row["text"])
        assert row["id"] == chunk_id_for(document_id, row["seq"])
        assert row["meta"]["doc_type"] == "issue"


async def test_second_index_skips_everything_and_keeps_ids(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    embedder = RecordingEmbedder()
    indexer = Indexer(session_factory=index_session_factory, embedder=embedder)

    first = await indexer.run(IndexRequest(document_ids=[document_id]))
    ids_after_first = [row["id"] for row in await chunk_rows(index_session_factory, document_id)]
    embedder.calls.clear()

    second = await indexer.run(IndexRequest(document_ids=[document_id]))
    ids_after_second = [row["id"] for row in await chunk_rows(index_session_factory, document_id)]

    assert second.chunks_written == 0
    assert second.chunks_skipped == first.chunks_written
    assert second.embed_tokens == 0
    assert embedder.calls == [], "unchanged chunks must never re-embed"
    assert ids_after_first == ids_after_second


async def test_identical_sections_are_embedded_once(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    note = "Same duplicated note text."
    content = f"# Notes page\n\n## Notes\n\n{note}\n\n## Notes\n\n{note}"
    document_id = await insert_document(
        index_session_factory,
        source="notion",
        doc_type="page",
        source_id="note-page-1",
        content=content,
        title="Notes page",
    )
    embedder = RecordingEmbedder()
    result = await Indexer(session_factory=index_session_factory, embedder=embedder).run(
        IndexRequest(document_ids=[document_id])
    )

    rows = await chunk_rows(index_session_factory, document_id)
    assert len(rows) == 2
    assert rows[0]["text"] == rows[1]["text"]
    assert result.chunks_written == 2
    assert len(embedder.texts) == 1, "identical text must cost one embedding"


async def test_changed_document_reembeds_only_changed_chunks(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    changed_id = await insert_document(index_session_factory, source_id="PHX-901")
    stable_id = await insert_document(index_session_factory, source_id="PHX-902")
    embedder = RecordingEmbedder()
    indexer = Indexer(session_factory=index_session_factory, embedder=embedder)
    await indexer.run(IndexRequest(document_ids=[changed_id, stable_id]))
    embedder.calls.clear()

    new_body = "Description body replaced after a review."
    async with index_session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == changed_id)
            .values(content=issue_content(key="PHX-901", body=new_body))
        )

    result = await indexer.run(IndexRequest(document_ids=[changed_id, stable_id]))
    assert result.chunks_written == 1
    assert result.chunks_skipped >= 1
    assert len(embedder.texts) == 1
    assert new_body in embedder.texts[0]


async def test_meta_only_change_rewrites_without_reembedding(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    embedder = RecordingEmbedder()
    indexer = Indexer(session_factory=index_session_factory, embedder=embedder)
    await indexer.run(IndexRequest(document_ids=[document_id]))
    before = await chunk_rows(index_session_factory, document_id)
    embedder.calls.clear()

    async with index_session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == document_id)
            .values(acl={"labels": ["phoenix", "retagged"]})
        )

    result = await indexer.run(IndexRequest(document_ids=[document_id]))
    after = await chunk_rows(index_session_factory, document_id)

    # A metadata-only refresh costs no tokens and counts as skipped, because
    # the contract defines "skipped" as "unchanged content_hash".
    assert result.chunks_written == 0
    assert result.chunks_skipped == len(after)
    assert result.embed_tokens == 0
    assert embedder.calls == []
    assert all("retagged" in row["meta"]["labels"] for row in after)
    assert [row["id"] for row in after] == [row["id"] for row in before]
    assert [row["content_hash"] for row in after] == [row["content_hash"] for row in before]


async def test_force_reembed_embeds_unchanged_chunks(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    embedder = RecordingEmbedder()
    indexer = Indexer(session_factory=index_session_factory, embedder=embedder)
    await indexer.run(IndexRequest(document_ids=[document_id]))
    embedder.calls.clear()

    result = await indexer.run(IndexRequest(document_ids=[document_id], force_reembed=True))
    assert result.chunks_written > 0
    assert result.chunks_skipped == 0
    assert result.embed_tokens > 0
    assert embedder.calls


async def test_superseded_document_chunks_are_deleted(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    indexer = Indexer(session_factory=index_session_factory, embedder=RecordingEmbedder())
    await indexer.run(IndexRequest(document_ids=[document_id]))
    assert await chunk_rows(index_session_factory, document_id)

    async with index_session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow).where(DocumentRow.id == document_id).values(is_current=False)
        )

    result = await indexer.run(IndexRequest(document_ids=[document_id]))
    assert await chunk_rows(index_session_factory, document_id) == []
    assert result == IndexResult(chunks_written=0, chunks_skipped=0, embed_tokens=0)


async def test_shrinking_document_deletes_trailing_chunks(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    long_body = "\n\n".join([f"{LONG_PARAGRAPH} Section {index}." for index in range(40)])
    document_id = await insert_document(
        index_session_factory, content=issue_content(key="PHX-903", body=long_body)
    )
    indexer = Indexer(session_factory=index_session_factory, embedder=RecordingEmbedder())
    await indexer.run(IndexRequest(document_ids=[document_id]))
    before = await chunk_rows(index_session_factory, document_id)
    assert len(before) > 2

    async with index_session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == document_id)
            .values(content=issue_content(key="PHX-903", body="Short now."))
        )

    await indexer.run(IndexRequest(document_ids=[document_id]))
    after = await chunk_rows(index_session_factory, document_id)
    assert len(after) < len(before)
    assert [row["seq"] for row in after] == list(range(len(after)))


async def test_emptied_document_removes_all_chunks(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    indexer = Indexer(session_factory=index_session_factory, embedder=RecordingEmbedder())
    await indexer.run(IndexRequest(document_ids=[document_id]))

    async with index_session_factory() as session, session.begin():
        await session.execute(
            update(DocumentRow).where(DocumentRow.id == document_id).values(content="   ")
        )
    result = await indexer.run(IndexRequest(document_ids=[document_id]))

    assert await chunk_rows(index_session_factory, document_id) == []
    assert result.chunks_written == 0


async def test_missing_document_is_skipped(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    embedder = RecordingEmbedder()
    result = await Indexer(session_factory=index_session_factory, embedder=embedder).run(
        IndexRequest(document_ids=[uuid.uuid4()])
    )
    assert result == IndexResult(chunks_written=0, chunks_skipped=0, embed_tokens=0)
    assert embedder.calls == []


async def test_empty_request_is_a_no_op(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    result = await Indexer(
        session_factory=index_session_factory, embedder=RecordingEmbedder()
    ).run(IndexRequest(document_ids=[]))
    assert result == IndexResult(chunks_written=0, chunks_skipped=0, embed_tokens=0)
    assert await audit_events(index_session_factory) == []


async def test_duplicate_document_ids_are_indexed_once(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    result = await Indexer(
        session_factory=index_session_factory, embedder=RecordingEmbedder()
    ).run(IndexRequest(document_ids=[document_id, document_id]))
    rows = await chunk_rows(index_session_factory, document_id)
    assert result.chunks_written == len(rows)
    assert result.chunks_skipped == 0


# ---------------------------------------------------------------------------
# Batching, failures, audit
# ---------------------------------------------------------------------------


async def test_batches_follow_configured_batch_size(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    long_body = "\n\n".join([f"{LONG_PARAGRAPH} Section {index}." for index in range(60)])
    document_id = await insert_document(
        index_session_factory, content=issue_content(key="PHX-904", body=long_body)
    )
    embedder = RecordingEmbedder()
    result = await Indexer(
        session_factory=index_session_factory, embedder=embedder, batch_size=2
    ).run(IndexRequest(document_ids=[document_id]))

    assert len(embedder.calls) >= 2
    assert all(len(call) <= 2 for call in embedder.calls)
    assert result.embed_tokens > 0
    assert len(await chunk_rows(index_session_factory, document_id)) == result.chunks_written


async def test_failed_batch_leaves_other_batches_written(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    sections = [
        f"## Section {index}\n\n{LONG_PARAGRAPH} Part {index}." for index in range(4)
    ]
    document_id = await insert_document(
        index_session_factory,
        source="notion",
        doc_type="page",
        source_id="multi-section-1",
        title="Multi section",
        content="# Multi section\n\n" + "\n\n".join(sections),
    )
    expected_chunks = len(
        chunk_document(
            SourceDocument(
                source="notion",
                source_id="multi-section-1",
                doc_type="page",
                title="Multi section",
                content="# Multi section\n\n" + "\n\n".join(sections),
            )
        )
    )
    assert expected_chunks >= 3, "fixture must produce enough chunks to span batches"

    embedder = FailingEmbedder(fail_on_call=2)
    result = await Indexer(
        session_factory=index_session_factory, embedder=embedder, batch_size=1
    ).run(IndexRequest(document_ids=[document_id]))

    written = await chunk_rows(index_session_factory, document_id)
    assert result.chunks_written == len(written) == expected_chunks - 1

    events = await audit_events(index_session_factory)
    assert len(events) == 1
    assert events[0]["payload"]["failed_batches"] == 1
    assert events[0]["payload"]["chunks_written"] == result.chunks_written
    assert events[0]["agent"] == "indexer"
    assert events[0]["latency_ms"] is not None


async def test_audit_event_records_the_run(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    document_id = await insert_document(index_session_factory)
    result = await Indexer(
        session_factory=index_session_factory, embedder=RecordingEmbedder()
    ).run(IndexRequest(document_ids=[document_id]))

    events = await audit_events(index_session_factory)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["documents_indexed"] == 1
    assert payload["chunks_written"] == result.chunks_written
    assert payload["embed_tokens"] == result.embed_tokens
    assert payload["embedding_dimensions"] == EMBEDDING_DIMENSIONS
    assert payload["embedding_model"] == "recording-fake"


def test_dimension_mismatch_fails_before_touching_rows(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IndexingConfigError, match="vector\\(768\\)"):
        Indexer(
            session_factory=index_session_factory,
            embedder=DeterministicFakeEmbedder(dimensions=1536),
        )


def test_invalid_batch_size_is_rejected(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IndexingConfigError, match="batch_size"):
        Indexer(
            session_factory=index_session_factory,
            embedder=RecordingEmbedder(),
            batch_size=0,
        )


# ---------------------------------------------------------------------------
# Whole-corpus integration pass (the M1 slice)
# ---------------------------------------------------------------------------


async def test_index_the_whole_fixture_corpus(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeded = await seed_fixture_corpus(index_session_factory, CORPUS)
    assert seeded.documents_seeded == len(CORPUS.documents)

    embedder = RecordingEmbedder(max_batch_size=8)
    indexer = Indexer(session_factory=index_session_factory, embedder=embedder)

    first = await indexer.run(IndexRequest(document_ids=fixture_document_ids(CORPUS)))
    async with index_session_factory() as session:
        total_chunks = (
            await session.execute(text("SELECT count(*) FROM core.chunks"))
        ).scalar_one()
        total_docs = (
            await session.execute(text("SELECT count(*) FROM core.documents"))
        ).scalar_one()
        vectors = (
            await session.execute(
                text("SELECT count(*) FROM core.chunks WHERE embedding IS NOT NULL")
            )
        ).scalar_one()

    assert total_docs == len(CORPUS.documents)
    assert total_chunks == first.chunks_written > len(CORPUS.documents)
    assert vectors == total_chunks
    assert first.embed_tokens > 0

    embedder.calls.clear()
    second = await indexer.run(IndexRequest(document_ids=fixture_document_ids(CORPUS)))
    assert second.chunks_written == 0
    assert second.chunks_skipped == total_chunks
    assert second.embed_tokens == 0
    assert embedder.calls == []


async def test_written_chunks_are_retrievable_by_fts_and_vector(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Sanity check that the writes are usable: FTS finds text, HNSW finds vectors."""
    await seed_fixture_corpus(index_session_factory, CORPUS)
    indexer = Indexer(session_factory=index_session_factory, embedder=RecordingEmbedder())
    await indexer.run(IndexRequest(document_ids=fixture_document_ids(CORPUS)))

    async with index_session_factory() as session:
        fts_hits = (
            await session.execute(
                text(
                    "SELECT count(*) FROM core.chunks WHERE fts @@ to_tsquery('english', 'cutover')"
                )
            )
        ).scalar_one()
        nearest = await session.scalar(
            text("SELECT text FROM core.chunks WHERE embedding IS NOT NULL LIMIT 1")
        )
        assert nearest is not None
        probe = await session.execute(
            text(
                "SELECT seq FROM core.chunks "
                "ORDER BY embedding <=> (SELECT embedding FROM core.chunks LIMIT 1) LIMIT 3"
            )
        )
        assert len(probe.all()) == 3
    assert fts_hits > 0


async def test_seed_is_idempotent(
    index_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await seed_fixture_corpus(index_session_factory, CORPUS)
    await seed_fixture_corpus(index_session_factory, CORPUS)
    async with index_session_factory() as session:
        total = (await session.execute(text("SELECT count(*) FROM core.documents"))).scalar_one()
        rows = (await session.execute(select(DocumentRow.id, DocumentRow.content_hash))).all()
    assert total == len(CORPUS.documents)
    assert len({row[0] for row in rows}) == len(CORPUS.documents)
