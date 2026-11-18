"""Hybrid retrieval over ``core.chunks`` (component 03).

Public surface:

- :class:`~retrieval.search.RetrieveRequest` / :class:`~retrieval.search.EntityRef`
  — what the planner sends.
- :func:`~retrieval.search.retrieve` — ``async def retrieve(req, *, user_id) ->
  EvidencePack``, returning the Lead-frozen
  :class:`orchestrator.pipeline.EvidencePack`.
- :mod:`retrieval.ranker` — pure scoring (RRF, recency decay, date proximity,
  entity boost, dedupe); no database, directly unit-testable.
- :class:`~retrieval.options.RetrievalOptions` — every tunable in one place.

This package only reads ``core.documents`` / ``core.chunks``. It is not the
writer of either (the connector and indexer agents own those), and it never
calls an LLM.
"""

from __future__ import annotations

from retrieval.errors import RetrievalConfigError, RetrievalError, RetrievalQueryError
from retrieval.options import RetrievalOptions
from retrieval.ranker import (
    Candidate,
    RankOutcome,
    ScoredCandidate,
    date_proximity_multiplier,
    entity_multiplier,
    entity_needles,
    find_dates,
    rank_candidates,
    recency_multiplier,
    reciprocal_rank_fusion,
    text_overlap,
)
from retrieval.search import (
    ACL_WILDCARD,
    EntityRef,
    RetrieveRequest,
    acl_predicate,
    build_candidate_query,
    entity_predicate,
    merge_candidates,
    retrieve,
)

__all__ = [
    "ACL_WILDCARD",
    "Candidate",
    "EntityRef",
    "RankOutcome",
    "RetrievalConfigError",
    "RetrievalError",
    "RetrievalOptions",
    "RetrievalQueryError",
    "RetrieveRequest",
    "ScoredCandidate",
    "acl_predicate",
    "build_candidate_query",
    "date_proximity_multiplier",
    "entity_multiplier",
    "entity_needles",
    "entity_predicate",
    "find_dates",
    "merge_candidates",
    "rank_candidates",
    "recency_multiplier",
    "reciprocal_rank_fusion",
    "retrieve",
    "text_overlap",
]
