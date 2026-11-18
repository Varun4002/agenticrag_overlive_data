"""Hybrid retrieval: pgvector ANN + Postgres FTS, fused, ACL-filtered in SQL.

Public contract (``docs/agents/03-retrieval-agent.md``)::

    async def retrieve(req: RetrieveRequest, *, user_id: str) -> EvidencePack

The pack is the frozen :class:`orchestrator.pipeline.EvidencePack`; this module
never redefines it.

What the query does
-------------------
Two candidate lists are pulled from Postgres, each already restricted by the
guardrail predicates, then fused in Python by the pure ranker:

1. **Vector**: ``ORDER BY embedding <=> :query_embedding`` (cosine, served by the
   HNSW index).
2. **Full text**: ``WHERE fts @@ websearch_to_tsquery('english', :q)`` ordered by
   ``ts_rank_cd``.

``websearch_to_tsquery`` never raises on malformed input (quotes, ``-``, ``or``),
which matters because the query text is planner/model output, not a trusted
literal.

ACL semantics (Layer 2 guardrail)
---------------------------------
A document is visible to a request only when::

    jsonb_exists(documents.acl -> 'users', :user_id)
    OR jsonb_exists(documents.acl -> 'users', '*')

That is the complete rule, deliberately:

- **Only the requesting user's own entry or a ``"*"`` wildcard grants access.**
  A document with no ``users`` key, an empty list, or a malformed ``acl`` is
  invisible: the default is deny, never allow.
- ``acl.groups`` / ``acl.labels`` are mirrored by the connectors but are **not**
  consulted here. Resolving a user's group membership needs an identity source
  this single-user MVP does not have; reading them as "any group grants access"
  would widen access to anyone, which is the failure the guardrail exists to
  prevent. When the identity layer lands, this predicate is the one place to
  change.
- The predicate is part of the ``WHERE`` clause of **both** candidate queries
  (:func:`build_candidate_query`), so an unauthorised chunk is never fetched,
  never ranked, and cannot be recovered by post-filtering (there is none).

Defensive filters
-----------------
``documents.is_current IS TRUE`` is applied even though supersession is supposed
to delete chunks: integration gap #1 in ``docs/08-decisions.md`` says nothing
wires retirement to chunk cleanup yet, so a retired document's chunks can still
be present. Filtering here means they can never surface.

Failure modes
-------------
Zero candidates → empty pack (no padding). Query embedding failure → FTS-only,
``degraded_reason`` set, ``retrieval.fallback`` event emitted. A database that is
not configured raises :class:`~retrieval.errors.RetrievalConfigError`: that is a
deployment error, not a degraded answer.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Select, Text, func, literal_column, or_, select
from sqlalchemy import cast as sa_cast
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import EMBEDDING_DIMENSIONS, Source
from db.models import Chunk as ChunkRow
from db.models import Document as DocumentRow
from db.session import get_session_factory
from db.tracing import Tracer, make_sink
from indexing.embedder import Embedder, build_embedder
from orchestrator.pipeline import EvidenceChunk, EvidencePack, SourceName
from retrieval.errors import RetrievalConfigError, RetrievalQueryError
from retrieval.options import RetrievalOptions
from retrieval.ranker import Candidate, ScoredCandidate, entity_needles, rank_candidates

logger = logging.getLogger(__name__)

AGENT = "retrieval"

#: Wildcard entry in ``documents.acl.users``: the document is visible to any user.
ACL_WILDCARD = "*"

#: Postgres ``english`` text-search configuration, spelled as a literal so the
#: bind parameter is not sent as ``text`` (there is no implicit cast to
#: ``regconfig`` and the two-argument overload would not resolve).
_ENGLISH: Any = literal_column("'english'::regconfig")

# ---------------------------------------------------------------------------
# Request contract
# ---------------------------------------------------------------------------


class EntityRef(BaseModel):
    """A resolved entity from the planner/memory agent.

    Shaped so the memory agent can hand a registry row over directly
    (``canonical_name`` → ``name``, ``aliases``, ``source_refs``). Matching is
    case-insensitive substring against chunk labels, heading path, document
    title and ``source_id``; pass the aliases — a registry entry for
    "Project Phoenix" carries ``PHX``/``Phoenix``, and those are what appear in
    the corpus.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    entity_type: str | None = None
    source_refs: dict[str, str] = Field(default_factory=dict)

    @property
    def terms(self) -> list[str]:
        """Every string that should be treated as a name for this entity."""
        values = [self.name, *self.aliases, *self.source_refs.values()]
        return [value.strip() for value in values if value and value.strip()]


class RetrieveRequest(BaseModel):
    """One retrieval request, as planned by the orchestrator."""

    query_text: str = Field(min_length=1)
    entity_filters: list[EntityRef] = Field(default_factory=list)
    #: Restrict to these sources; ``None`` means all of them.
    sources: list[SourceName] | None = None
    #: Chunks older than this are still returned, flagged ``stale``.
    freshness_window: timedelta | None = None
    top_k: int = Field(default=12, ge=1, le=50)


# ---------------------------------------------------------------------------
# SQL predicates (shared by both candidate queries)
# ---------------------------------------------------------------------------


def acl_predicate(user_id: str) -> Any:
    """Layer 2 ACL filter, as a SQL expression over ``core.documents``.

    See the module docstring for the exact rule. Fail-closed: no matching
    ``users`` entry and no wildcard means the row is not visible.
    """
    if not user_id or not user_id.strip():
        raise RetrievalQueryError("user_id must be a non-empty string to enforce the ACL")
    users = DocumentRow.acl["users"]
    return or_(
        func.jsonb_exists(users, user_id),
        func.jsonb_exists(users, ACL_WILDCARD),
    )


def _like_pattern(needle: str) -> str:
    """``%needle%`` with LIKE metacharacters escaped (user/entity text)."""
    escaped = (
        needle.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


def entity_predicate(needles: Sequence[str]) -> Any:
    """Cheap SQL prefilter for resolved entities: any needle, any field.

    Deliberately fuzzy and permissive (OR across needles and fields) — it is a
    candidate-narrowing aid, not the authority on relevance. Ranking applies the
    matching boost that actually orders the results.
    """
    clauses: list[Any] = []
    for needle in needles:
        pattern = _like_pattern(needle)
        clauses.extend(
            (
                func.lower(DocumentRow.title).like(pattern, escape="\\"),
                func.lower(DocumentRow.source_id).like(pattern, escape="\\"),
                func.lower(sa_cast(ChunkRow.meta["labels"], Text)).like(pattern, escape="\\"),
                func.lower(sa_cast(ChunkRow.meta["heading_path"], Text)).like(pattern, escape="\\"),
            )
        )
    return or_(*clauses)


def build_candidate_query(
    *,
    user_id: str,
    order: Literal["vector", "fts"],
    limit: int,
    query_text: str | None = None,
    embedding: Sequence[float] | None = None,
    sources: Sequence[str] | None = None,
    needles: Sequence[str] = (),
    apply_entity_filter: bool = True,
) -> Select[Any]:
    """Build one candidate query with every guardrail predicate inside it.

    Exposed (rather than inlined) so tests can compile the statement and assert
    the ACL predicate is part of the ``WHERE`` clause instead of trusting that
    the results happen to be filtered.
    """
    if limit < 1:
        raise RetrievalQueryError("limit must be >= 1")

    columns: list[Any] = [
        ChunkRow.id.label("chunk_id"),
        ChunkRow.document_id.label("document_id"),
        DocumentRow.source.label("source"),
        DocumentRow.source_id.label("source_id"),
        DocumentRow.title.label("title"),
        DocumentRow.url.label("url"),
        DocumentRow.updated_at.label("updated_at"),
        ChunkRow.text.label("text"),
        ChunkRow.meta.label("meta"),
    ]

    statement = select(*columns)
    if order == "vector":
        if embedding is None:
            raise RetrievalQueryError("vector search requires an embedding")
        vector = [float(value) for value in embedding]
        distance = ChunkRow.embedding.cosine_distance(vector)
        statement = statement.add_columns((1.0 - distance).label("score"))
        statement = statement.where(ChunkRow.embedding.is_not(None))
        statement = statement.order_by(distance, ChunkRow.id)
    elif order == "fts":
        if not query_text:
            raise RetrievalQueryError("full-text search requires query_text")
        tsquery = func.websearch_to_tsquery(_ENGLISH, query_text)
        statement = statement.add_columns(func.ts_rank_cd(ChunkRow.fts, tsquery).label("score"))
        statement = statement.where(ChunkRow.fts.op("@@")(tsquery))
        statement = statement.order_by(
            func.ts_rank_cd(ChunkRow.fts, tsquery).desc(), ChunkRow.id
        )
    else:  # pragma: no cover - Literal is checked statically
        raise RetrievalQueryError(f"unknown candidate order {order!r}")

    statement = statement.join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
    # Fail-closed ACL + defensive is_current: both in the WHERE clause.
    statement = statement.where(DocumentRow.is_current.is_(True), acl_predicate(user_id))
    if sources:
        statement = statement.where(DocumentRow.source.in_([Source(value) for value in sources]))
    if needles and apply_entity_filter:
        statement = statement.where(entity_predicate(needles))
    return cast(Select[Any], statement.limit(limit))


# ---------------------------------------------------------------------------
# Row → candidate
# ---------------------------------------------------------------------------


def _candidate_from_row(
    row: Any,
    *,
    vector_rank: int | None = None,
    fts_rank: int | None = None,
    score: float | None = None,
) -> Candidate:
    mapping = row._mapping
    source = mapping["source"]
    meta = mapping["meta"]
    return Candidate(
        chunk_id=str(mapping["chunk_id"]),
        document_id=str(mapping["document_id"]),
        source=str(getattr(source, "value", source)),
        source_id=str(mapping["source_id"]),
        title=str(mapping["title"]),
        url=None if mapping["url"] is None else str(mapping["url"]),
        updated_at=mapping["updated_at"],
        text=str(mapping["text"]),
        meta=dict(meta) if isinstance(meta, Mapping) else {},
        vector_rank=vector_rank,
        vector_similarity=score if vector_rank is not None else None,
        fts_rank=fts_rank,
        fts_score=score if fts_rank is not None else None,
    )


def merge_candidates(
    vector_rows: Sequence[Candidate], fts_rows: Sequence[Candidate]
) -> list[Candidate]:
    """Union the two ranked lists into one candidate per chunk (both ranks kept)."""
    merged: dict[str, Candidate] = {candidate.chunk_id: candidate for candidate in vector_rows}
    for candidate in fts_rows:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            merged[candidate.chunk_id] = replace(
                existing, fts_rank=candidate.fts_rank, fts_score=candidate.fts_score
            )
    return list(merged.values())


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def _run_query(session: AsyncSession, statement: Select[Any]) -> Sequence[Any]:
    return cast(Sequence[Any], (await session.execute(statement)).all())


async def retrieve(
    req: RetrieveRequest,
    *,
    user_id: str,
    trace_id: str | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
    options: RetrievalOptions | None = None,
    now: datetime | None = None,
) -> EvidencePack:
    """Retrieve the best ACL-clean candidates for one question.

    ``trace_id`` continues the request trace; a fresh one is generated when the
    caller has none. The keyword-only collaborators default to the process-wide
    settings-driven ones and exist so tests (and the eval harness) can inject a
    scratch database, an offline embedder, or pinned ranking options.
    """
    started = time.perf_counter()
    opts = options or RetrievalOptions.from_env()
    clock = now or datetime.now(UTC)
    factory = session_factory or get_session_factory()
    if factory is None:
        raise RetrievalConfigError(
            "retrieval needs DATABASE_URL: there is no database to search"
        )

    tracer = Tracer(make_sink(factory), trace_id)
    needles = entity_needles(
        term for entity_filter in req.entity_filters for term in entity_filter.terms
    )

    # --- query embedding (degradable) -----------------------------------
    degraded_reason: str | None = None
    embedding: list[float] | None = None
    embed_tokens = 0
    owned_embedder = embedder is None
    active_embedder: Embedder | None = embedder
    embedder_model = getattr(embedder, "model", None)
    try:
        if active_embedder is None:
            active_embedder = build_embedder()
            embedder_model = active_embedder.model
        result = await active_embedder.embed([req.query_text])
        vector = result.vectors[0] if result.vectors else []
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RetrievalQueryError(
                f"embedder {active_embedder.model!r} returned {len(vector)} dimensions; "
                f"core.chunks.embedding is vector({EMBEDDING_DIMENSIONS})"
            )
        embedding = [float(value) for value in vector]
        embed_tokens = result.tokens
    except Exception as exc:  # any embedder failure degrades; it never crashes retrieval
        degraded_reason = f"embedding_unavailable: {type(exc).__name__}: {exc}"
        embedding = None
        logger.warning(
            "retrieval falling back to full-text only (encoder %r): %s", embedder_model, exc
        )
        await tracer.emit(
            AGENT,
            "retrieval.fallback",
            {
                "reason": degraded_reason,
                "mode": "fts_only",
                "query": req.query_text,
                "embedder": embedder_model,
                "top_k": req.top_k,
            },
        )
    finally:
        if owned_embedder and active_embedder is not None:
            try:
                await active_embedder.aclose()
            except Exception as exc:  # closing the client must not mask results
                logger.warning("failed to close retrieval embedder: %s", exc)

    # --- candidate generation -------------------------------------------
    vector_rows: list[Candidate] = []
    fts_rows: list[Candidate] = []
    async with factory() as session:
        if embedding is not None:
            vector_statement = build_candidate_query(
                user_id=user_id,
                order="vector",
                limit=opts.candidate_depth,
                embedding=embedding,
                sources=req.sources,
                needles=needles,
                apply_entity_filter=opts.entity_filter_mode == "filter",
            )
            for row in await _run_query(session, vector_statement):
                similarity = float(row._mapping["score"])
                # pgvector returns NaN for a zero query vector and SQL places
                # it last; a non-finite score is not a candidate.
                if not math.isfinite(similarity):
                    continue
                if similarity < opts.min_vector_similarity:
                    continue
                vector_rows.append(
                    _candidate_from_row(
                        row, vector_rank=len(vector_rows) + 1, score=similarity
                    )
                )

        fts_statement = build_candidate_query(
            user_id=user_id,
            order="fts",
            limit=opts.candidate_depth,
            query_text=req.query_text,
            sources=req.sources,
            needles=needles,
            apply_entity_filter=opts.entity_filter_mode == "filter",
        )
        fts_rows = [
            _candidate_from_row(row, fts_rank=position, score=float(row._mapping["score"]))
            for position, row in enumerate(await _run_query(session, fts_statement), start=1)
        ]

    # --- fuse, score, select --------------------------------------------
    candidates = merge_candidates(vector_rows, fts_rows)
    outcome = rank_candidates(
        candidates,
        options=opts,
        now=clock,
        query_text=req.query_text,
        entity_terms=[term for entity_filter in req.entity_filters for term in entity_filter.terms],
        freshness_window=req.freshness_window,
    )
    selected = outcome.chunks[: req.top_k]
    pack = EvidencePack(
        question=req.query_text,
        chunks=[_to_evidence_chunk(item) for item in selected],
        dropped_below_floor=outcome.dropped_below_floor,
        degraded_reason=degraded_reason,
    )

    await tracer.emit(
        AGENT,
        "retrieval.ran",
        {
            "query": req.query_text,
            "user_id": user_id,
            "sources": list(req.sources) if req.sources else None,
            "entity_terms": needles,
            "freshness_window_s": (
                req.freshness_window.total_seconds() if req.freshness_window else None
            ),
            "top_k": req.top_k,
            "vector_candidates": len(vector_rows),
            "fts_candidates": len(fts_rows),
            "merged_candidates": len(candidates),
            "deduped": outcome.deduped,
            "dropped_below_floor": outcome.dropped_below_floor,
            "returned": len(pack.chunks),
            "stale_chunk_ids": [chunk.chunk_id for chunk in pack.chunks if chunk.stale],
            "top_score": pack.chunks[0].score if pack.chunks else None,
            "degraded_reason": degraded_reason,
            "embedder": embedder_model,
            "embed_tokens": embed_tokens,
            "score_floor": opts.score_floor,
            "rrf_k": opts.rrf_k,
            "min_vector_similarity": opts.min_vector_similarity,
        },
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return pack


def _to_evidence_chunk(scored: ScoredCandidate) -> EvidenceChunk:
    candidate = scored.candidate
    return EvidenceChunk(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        text=candidate.text,
        score=scored.score,
        source=_source_name(candidate.source),
        source_id=candidate.source_id,
        title=candidate.title,
        url=candidate.url,
        updated_at=candidate.updated_at,
        stale=scored.stale,
        meta=dict(candidate.meta),
    )


def _source_name(value: str) -> SourceName:
    if value not in {"jira", "notion", "gmail"}:  # pragma: no cover - schema enum
        raise RetrievalQueryError(f"unknown source {value!r} in core.documents")
    return cast(SourceName, value)


__all__ = [
    "ACL_WILDCARD",
    "AGENT",
    "EntityRef",
    "RetrieveRequest",
    "acl_predicate",
    "build_candidate_query",
    "entity_predicate",
    "merge_candidates",
    "retrieve",
]
