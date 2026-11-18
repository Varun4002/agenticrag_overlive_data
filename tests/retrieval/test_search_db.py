"""Retrieval against a real scratch Postgres (skipped without one).

Seeded from ``fixtures/corpus`` through the real indexer, so these tests exercise
the actual schema: pgvector ANN, the generated ``tsvector``, jsonb ACLs, and the
audit trail. Every assertion is about behaviour the brief demands — relevance,
ACL denial, ``is_current`` defence, recency, staleness, floor, fallback, dedupe.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from fakes import (
    ATLAS_USER,
    CORPUS_NOW,
    PHOENIX_USER,
    STRANGER_USER,
    FailingEmbedder,
    IndexedCorpus,
    count_chunks,
    insert_chunk,
    insert_document,
    retire_document,
    set_document_acl,
)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import Document as DocumentRow
from db.tracing import read_trace
from orchestrator.pipeline import EvidencePack
from retrieval.options import RetrievalOptions
from retrieval.search import EntityRef, RetrieveRequest

pytestmark = pytest.mark.usefixtures("clean_retrieval_db")

DAY = timedelta(days=1)


def document_ids(pack: EvidencePack) -> list[str]:
    return [chunk.document_id for chunk in pack.chunks]


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


async def test_known_relevant_chunk_ranks_top_three(indexed_corpus: IndexedCorpus) -> None:
    assert indexed_corpus.documents == 44
    assert indexed_corpus.chunks > 40

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="Cut over billing webhooks to Ledger v2"),
        now=CORPUS_NOW,
    )

    assert pack.degraded_reason is None
    assert pack.chunks
    assert pack.chunks[0].source_id == "PHX-150"
    assert any(chunk.source_id == "PHX-150" for chunk in pack.chunks[:3])
    # Score is the fused hybrid score, descending, and every chunk carries its
    # citation coordinates.
    scores = [chunk.score for chunk in pack.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(chunk.title and chunk.source in {"jira", "notion", "gmail"} for chunk in pack.chunks)


async def test_resolved_entity_filter_restricts_to_one_project(
    indexed_corpus: IndexedCorpus,
) -> None:
    req = RetrieveRequest(
        query_text="cutover readiness and open blockers",
        entity_filters=[EntityRef(name="Project Atlas", aliases=["ATL", "Atlas"])],
    )

    pack = await indexed_corpus.retrieve(req, user_id=ATLAS_USER, now=CORPUS_NOW)

    assert pack.chunks
    for chunk in pack.chunks:
        assert "atlas" in f"{chunk.source_id} {chunk.title} {chunk.meta}".lower()


async def test_entity_filter_with_no_matches_returns_empty_pack(
    indexed_corpus: IndexedCorpus,
) -> None:
    pack = await indexed_corpus.retrieve(
        RetrieveRequest(
            query_text="what is the cutover date",
            entity_filters=[EntityRef(name="Zephyr Quokka Initiative")],
        ),
        now=CORPUS_NOW,
    )

    assert pack.chunks == []
    assert pack.dropped_below_floor == 0
    assert pack.degraded_reason is None


async def test_sources_filter_is_applied_in_sql(indexed_corpus: IndexedCorpus) -> None:
    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover due date", sources=["notion"]), now=CORPUS_NOW
    )

    assert pack.chunks
    assert {chunk.source for chunk in pack.chunks} == {"notion"}


async def test_top_k_bounds_the_pack(indexed_corpus: IndexedCorpus) -> None:
    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="project phoenix status cutover ledger", top_k=3),
        now=CORPUS_NOW,
    )

    assert len(pack.chunks) <= 3


# ---------------------------------------------------------------------------
# ACL — Layer 2 guardrail
# ---------------------------------------------------------------------------


async def test_acl_denies_a_user_who_is_not_listed(indexed_corpus: IndexedCorpus) -> None:
    visible_to_owner = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover billing webhooks"), now=CORPUS_NOW
    )
    denied = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover billing webhooks"),
        user_id=STRANGER_USER,
        now=CORPUS_NOW,
    )

    assert visible_to_owner.chunks, "the owner must still see the corpus"
    assert denied.chunks == []
    # The rows exist — they were excluded by the WHERE clause, not by absence.
    async with indexed_corpus.factory() as session:
        total = len((await session.execute(text("SELECT id FROM core.chunks"))).all())
    assert total > 0


async def test_acl_wildcard_grants_access(indexed_corpus: IndexedCorpus) -> None:
    async with indexed_corpus.factory() as session:
        document_id = (
            await session.execute(
                select(DocumentRow.id).where(DocumentRow.source_id == "PHX-150")
            )
        ).scalar_one()
    await set_document_acl(
        indexed_corpus.factory, document_id, {"users": ["*"], "groups": [], "labels": []}
    )

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover billing webhooks"),
        user_id=STRANGER_USER,
        now=CORPUS_NOW,
    )

    assert pack.chunks
    assert {chunk.document_id for chunk in pack.chunks} == {str(document_id)}


# ---------------------------------------------------------------------------
# Defensive is_current filter (integration gap #1)
# ---------------------------------------------------------------------------


async def test_superseded_document_chunks_are_excluded(indexed_corpus: IndexedCorpus) -> None:
    async with indexed_corpus.factory() as session:
        document_id = (
            await session.execute(
                select(DocumentRow.id).where(DocumentRow.source_id == "PHX-150")
            )
        ).scalar_one()

    # Retirement without chunk cleanup is exactly the gap: chunks stay behind.
    await retire_document(indexed_corpus.factory, document_id)
    assert await count_chunks(indexed_corpus.factory, document_id) > 0

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="Cut over billing webhooks to Ledger v2"), now=CORPUS_NOW
    )

    assert str(document_id) not in document_ids(pack)
    assert pack.chunks, "other documents are still retrievable"


# ---------------------------------------------------------------------------
# Recency and staleness
# ---------------------------------------------------------------------------


async def test_recency_orders_equally_relevant_chunks(indexed_corpus: IndexedCorpus) -> None:
    query = "zephyrine quokka release rehearsal"
    text_body = "Zephyrine quokka release rehearsal for the billing webhooks handover."
    vector = (await indexed_corpus.embedder.embed([query])).vectors[0]

    fresh = await insert_document(
        indexed_corpus.factory,
        source_id="PHX-950",
        title="PHX-950 — Zephyrine quokka release rehearsal",
        content=text_body,
        updated_at=CORPUS_NOW,
        acl={"users": [PHOENIX_USER]},
    )
    old = await insert_document(
        indexed_corpus.factory,
        source_id="PHX-951",
        title="PHX-950 — Zephyrine quokka release rehearsal",
        content=text_body,
        updated_at=CORPUS_NOW - DAY * 200,
        acl={"users": [PHOENIX_USER]},
    )
    for document_id in (fresh, old):
        await insert_chunk(
            indexed_corpus.factory, document_id=document_id, text=text_body, embedding=vector
        )

    pack = await indexed_corpus.retrieve(RetrieveRequest(query_text=query), now=CORPUS_NOW)

    ordered = document_ids(pack)
    assert str(fresh) in ordered and str(old) in ordered
    assert ordered.index(str(fresh)) < ordered.index(str(old))


async def test_freshness_window_flags_but_returns_old_chunks(
    indexed_corpus: IndexedCorpus,
) -> None:
    fresh_window = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover billing webhooks", freshness_window=DAY * 30),
        now=CORPUS_NOW,
    )
    tight_window = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="cutover billing webhooks", freshness_window=DAY),
        now=CORPUS_NOW,
    )

    assert fresh_window.chunks and not any(chunk.stale for chunk in fresh_window.chunks)
    # Older than a day, so flagged — but still returned for the planner to judge.
    assert tight_window.chunks
    assert any(chunk.stale for chunk in tight_window.chunks)


# ---------------------------------------------------------------------------
# Score floor
# ---------------------------------------------------------------------------


async def test_score_floor_drops_the_tail(indexed_corpus: IndexedCorpus) -> None:
    req = RetrieveRequest(query_text="phoenix cutover status ledger webhooks")

    everything = await indexed_corpus.retrieve(req, options=RetrievalOptions(score_floor=0.0),
                                                now=CORPUS_NOW)
    floored = await indexed_corpus.retrieve(req, options=RetrievalOptions(score_floor=0.03),
                                            now=CORPUS_NOW)

    assert floored.dropped_below_floor > 0
    assert len(floored.chunks) < len(everything.chunks)
    assert all(chunk.score >= 0.03 for chunk in floored.chunks)


async def test_min_vector_similarity_gate_can_empty_the_pack(
    indexed_corpus: IndexedCorpus,
) -> None:
    options = RetrievalOptions(min_vector_similarity=0.999, score_floor=0.0)

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="zzzq quokka zzzq"),
        options=options,
        now=CORPUS_NOW,
    )

    # No FTS match, and every ANN neighbour is below the gate.
    assert pack.chunks == []


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


async def test_near_identical_chunks_are_collapsed(indexed_corpus: IndexedCorpus) -> None:
    query = "quokka webhook replay dedupe"
    body = (
        "Quokka webhook replay dedupe: the replay harness re-sends every Nimbus "
        "webhook twice, so the local ledger must be idempotent before cutover."
    )
    vector = (await indexed_corpus.embedder.embed([body])).vectors[0]
    document_id = await insert_document(
        indexed_corpus.factory,
        source_id="PHX-960",
        title="PHX-960 — Quokka webhook replay dedupe",
        content=body,
        updated_at=CORPUS_NOW,
        acl={"users": [PHOENIX_USER]},
    )
    for seq in (0, 1):
        await insert_chunk(
            indexed_corpus.factory,
            document_id=document_id,
            text=body,
            seq=seq,
            heading_path=("description",),
            embedding=vector,
        )

    trace_id = str(uuid.uuid4())
    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text=query), trace_id=trace_id, now=CORPUS_NOW
    )

    matching = [chunk for chunk in pack.chunks if chunk.document_id == str(document_id)]
    assert len(matching) == 1
    payload = await event_payload(indexed_corpus.factory, trace_id, "retrieval.ran")
    assert payload["deduped"] >= 1


# ---------------------------------------------------------------------------
# Degradation and audit
# ---------------------------------------------------------------------------


async def test_embedder_failure_falls_back_to_full_text(indexed_corpus: IndexedCorpus) -> None:
    trace_id = str(uuid.uuid4())

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="Cut over billing webhooks to Ledger v2"),
        trace_id=trace_id,
        embedder=FailingEmbedder(),
        now=CORPUS_NOW,
    )

    assert pack.degraded_reason is not None
    assert pack.degraded_reason.startswith("embedding_unavailable")
    assert pack.chunks, "FTS-only retrieval still returns evidence"
    assert pack.chunks[0].source_id == "PHX-150"

    fallback = await event_payload(indexed_corpus.factory, trace_id, "retrieval.fallback")
    assert fallback["mode"] == "fts_only"
    assert "simulated embedding provider outage" in fallback["reason"]

    ran = await event_payload(indexed_corpus.factory, trace_id, "retrieval.ran")
    assert ran["vector_candidates"] == 0
    assert ran["fts_candidates"] > 0
    assert ran["degraded_reason"] == pack.degraded_reason


async def test_audit_event_records_the_query_and_counts(indexed_corpus: IndexedCorpus) -> None:
    trace_id = str(uuid.uuid4())

    pack = await indexed_corpus.retrieve(
        RetrieveRequest(
            query_text="which phoenix tasks are blocked",
            sources=["jira"],
            freshness_window=DAY,
            top_k=5,
        ),
        trace_id=trace_id,
        now=CORPUS_NOW,
    )

    payload = await event_payload(indexed_corpus.factory, trace_id, "retrieval.ran")

    assert payload["query"] == "which phoenix tasks are blocked"
    assert payload["sources"] == ["jira"]
    assert payload["top_k"] == 5
    assert payload["freshness_window_s"] == DAY.total_seconds()
    assert payload["returned"] == len(pack.chunks)
    assert payload["vector_candidates"] > 0
    assert payload["fts_candidates"] > 0
    assert payload["top_score"] == pytest.approx(pack.chunks[0].score)
    assert payload["degraded_reason"] is None
    assert payload["embed_tokens"] > 0

    events = await read_trace(indexed_corpus.factory, trace_id)
    latency = [event.latency_ms for event in events if event.event == "retrieval.ran"]
    assert latency and latency[0] is not None


async def event_payload(
    factory: async_sessionmaker[AsyncSession], trace_id: str, event: str
) -> dict[str, object]:
    events = await read_trace(factory, trace_id)
    matching = [record for record in events if record.event == event]
    assert matching, f"expected a {event} event, got {[r.event for r in events]}"
    return matching[0].payload


# ---------------------------------------------------------------------------
# End-to-end shape
# ---------------------------------------------------------------------------


async def test_pack_fields_map_onto_the_frozen_contract(indexed_corpus: IndexedCorpus) -> None:
    pack = await indexed_corpus.retrieve(
        RetrieveRequest(query_text="phoenix goals and okrs"), user_id=PHOENIX_USER, now=CORPUS_NOW
    )

    assert pack.question == "phoenix goals and okrs"
    assert pack.chunks
    chunk = pack.chunks[0]
    assert chunk.chunk_id and chunk.document_id
    assert chunk.text
    assert chunk.url is not None
    assert chunk.updated_at.tzinfo is not None
    assert isinstance(chunk.meta, dict) and "heading_path" in chunk.meta
    assert 0.0 < chunk.score < 1.0
    # Re-serialising must not lose anything (the scrubber consumes this shape).
    assert EvidencePack.model_validate(pack.model_dump()) == pack


async def test_retrieve_is_repeatable(indexed_corpus: IndexedCorpus) -> None:
    req = RetrieveRequest(query_text="phoenix cutover blockers")

    first = await indexed_corpus.retrieve(req, now=CORPUS_NOW)
    second = await indexed_corpus.retrieve(req, now=CORPUS_NOW)

    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert [chunk.score for chunk in first.chunks] == pytest.approx(
        [chunk.score for chunk in second.chunks]
    )
