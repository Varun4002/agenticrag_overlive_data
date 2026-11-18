"""Contract and SQL-shape tests that need no database.

The ACL test lives here on purpose: the guardrail requirement is that the filter
is *in the statement*, not that a particular row happened to be missing from a
result set. Compiling the statement and inspecting the SQL proves that directly,
and it runs without Postgres.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from orchestrator.pipeline import EvidencePack as FrozenEvidencePack
from retrieval import errors, search
from retrieval.options import RetrievalOptions
from retrieval.search import (
    ACL_WILDCARD,
    EntityRef,
    RetrieveRequest,
    acl_predicate,
    build_candidate_query,
    retrieve,
)


def compiled(statement: object) -> tuple[str, list[object]]:
    result = statement.compile(  # type: ignore[attr-defined]
        dialect=postgresql.dialect(), compile_kwargs={"render_postcompile": True}
    )
    return str(result), list(result.params.values())


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_retrieval_returns_the_lead_frozen_evidence_pack() -> None:
    assert search.EvidencePack is FrozenEvidencePack


async def test_retrieve_without_a_database_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "get_session_factory", lambda: None)

    with pytest.raises(errors.RetrievalConfigError):
        await retrieve(RetrieveRequest(query_text="anything"), user_id="someone@example.com")


def test_request_defaults_and_bounds() -> None:
    req = RetrieveRequest(query_text="What is due this week?")

    assert req.top_k == 12
    assert req.entity_filters == []
    assert req.sources is None
    assert req.freshness_window is None

    with pytest.raises(ValidationError):
        RetrieveRequest(query_text="")
    with pytest.raises(ValidationError):
        RetrieveRequest(query_text="x", top_k=0)
    with pytest.raises(ValidationError):
        RetrieveRequest(query_text="x", top_k=51)
    with pytest.raises(ValidationError):
        RetrieveRequest(query_text="x", sources=["slack"])  # type: ignore[list-item]


def test_entity_ref_terms_cover_name_aliases_and_source_refs() -> None:
    entity = EntityRef(
        name="Project Phoenix",
        aliases=["Phoenix", " PHX "],
        source_refs={"jira": "PHX", "notion": "Phoenix pages"},
    )

    assert entity.terms == ["Project Phoenix", "Phoenix", "PHX", "PHX", "Phoenix pages"]


# ---------------------------------------------------------------------------
# ACL predicate — Layer 2 guardrail
# ---------------------------------------------------------------------------


def test_acl_predicate_is_part_of_the_where_clause() -> None:
    statement = build_candidate_query(user_id="priya@example.com", order="fts", limit=10,
                                      query_text="cutover")
    sql, params = compiled(statement)

    assert "jsonb_exists" in sql
    assert "WHERE" in sql
    # The ACL check precedes the ORDER BY, i.e. it filters rows rather than
    # decorating a result set that was already fetched.
    assert sql.index("WHERE") < sql.index("jsonb_exists") < sql.index("ORDER BY")
    assert "priya@example.com" in params
    assert ACL_WILDCARD in params


def test_acl_predicate_rejects_an_empty_identity() -> None:
    with pytest.raises(errors.RetrievalQueryError):
        acl_predicate("   ")
    with pytest.raises(errors.RetrievalQueryError):
        build_candidate_query(user_id="", order="fts", limit=5, query_text="x")


def test_candidate_query_filters_sources_in_sql() -> None:
    statement = build_candidate_query(
        user_id="me@example.com", order="fts", limit=5, query_text="status", sources=["notion"]
    )
    sql, _ = compiled(statement)

    assert "core.documents.source IN" in sql
    assert "is_current" in sql


def test_candidate_query_applies_entity_filter_only_when_asked() -> None:
    filtered = build_candidate_query(
        user_id="me@example.com",
        order="fts",
        limit=5,
        query_text="cutover",
        needles=["phoenix"],
    )
    unfiltered = build_candidate_query(
        user_id="me@example.com",
        order="fts",
        limit=5,
        query_text="cutover",
        needles=["phoenix"],
        apply_entity_filter=False,
    )

    filtered_sql, filtered_params = compiled(filtered)
    unfiltered_sql, _ = compiled(unfiltered)

    assert "lower(core.documents.title) LIKE" in filtered_sql
    assert "lower(core.documents.title) LIKE" not in unfiltered_sql
    assert "%phoenix%" in filtered_params


def test_entity_like_patterns_escape_metacharacters() -> None:
    statement = build_candidate_query(
        user_id="me@example.com",
        order="fts",
        limit=5,
        query_text="cutover",
        needles=["100%_done"],
    )
    _, params = compiled(statement)

    assert "%100\\%\\_done%" in params


def test_vector_query_requires_an_embedding_and_skips_null_vectors() -> None:
    with pytest.raises(errors.RetrievalQueryError):
        build_candidate_query(user_id="me@example.com", order="vector", limit=5)

    statement = build_candidate_query(
        user_id="me@example.com", order="vector", limit=5, embedding=[0.0] * 768
    )
    sql, params = compiled(statement)

    assert "<=>" in sql
    assert "embedding IS NOT NULL" in sql
    assert len([value for value in params if isinstance(value, list)]) == 1


def test_fts_query_uses_websearch_tsquery_with_an_explicit_config() -> None:
    statement = build_candidate_query(
        user_id="me@example.com", order="fts", limit=5, query_text='"cutover" -atlas or phoenix'
    )
    sql, params = compiled(statement)

    assert "websearch_to_tsquery('english'::regconfig" in sql
    assert "ts_rank_cd" in sql
    assert '@@' in sql
    # The raw query text is a bind parameter, never interpolated.
    assert '"cutover" -atlas or phoenix' in params


def test_candidate_query_rejects_zero_limit() -> None:
    with pytest.raises(errors.RetrievalQueryError):
        build_candidate_query(user_id="me@example.com", order="fts", limit=0, query_text="x")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_options_from_env_overrides_defaults() -> None:
    options = RetrievalOptions.from_env(
        {
            "RETRIEVAL_RRF_K": "30",
            "RETRIEVAL_RECENCY_HALF_LIFE_DAYS": "7",
            "RETRIEVAL_SCORE_FLOOR": "0.02",
            "RETRIEVAL_TOP_K": "5",
        }
    )

    assert options.rrf_k == 30
    assert options.recency_half_life == timedelta(days=7)
    assert options.score_floor == pytest.approx(0.02)
    # Unknown/irrelevant variables are ignored; defaults survive.
    assert options.candidate_depth == RetrievalOptions().candidate_depth


def test_options_reject_nonsense() -> None:
    with pytest.raises(ValueError):
        RetrievalOptions(rrf_k=0)
    with pytest.raises(ValueError):
        RetrievalOptions(score_floor=-1)
    with pytest.raises(ValueError):
        RetrievalOptions(entity_filter_mode="sometimes")
    with pytest.raises(ValueError):
        RetrievalOptions.from_env({"RETRIEVAL_RRF_K": "sixty"})
