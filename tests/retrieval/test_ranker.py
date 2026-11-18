"""Pure ranking tests — no database, no clock, no network.

These pin the scoring model: RRF arithmetic, bounded recency decay, date
proximity, entity boosts, duplicate collapse, the per-document cap, the score
floor, and staleness flagging. The DB tests in ``test_search_db.py`` only prove
that the SQL feeds this model correctly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from retrieval.options import RetrievalOptions
from retrieval.ranker import (
    Candidate,
    date_proximity_multiplier,
    entity_multiplier,
    entity_needles,
    find_dates,
    rank_candidates,
    recency_multiplier,
    reciprocal_rank_fusion,
    text_overlap,
)

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)


def candidate(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    text: str = "Ledger v2 cutover rehearsal notes",
    updated_at: datetime = NOW,
    source_id: str = "PHX-150",
    title: str = "PHX-150 — Cut over billing webhooks to Ledger v2",
    labels: tuple[str, ...] = ("phoenix",),
    heading_path: tuple[str, ...] = ("description",),
    vector_rank: int | None = None,
    fts_rank: int | None = None,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=document_id,
        source="jira",
        source_id=source_id,
        title=title,
        url=None,
        updated_at=updated_at,
        text=text,
        meta={"heading_path": list(heading_path), "labels": list(labels)},
        vector_rank=vector_rank,
        fts_rank=fts_rank,
    )


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def test_rrf_sums_reciprocal_ranks_and_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion({"a": 1, "b": 2}, {"a": 1, "c": 1}, k=60)

    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 61)
    # Found by both retrievers beats found by one at the same rank.
    assert fused["a"] > fused["c"] > fused["b"]


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion({}, {}) == {}
    assert reciprocal_rank_fusion({"a": 1}, {}) == {"a": pytest.approx(1 / 61)}


def test_rrf_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": 0}, {}, k=60)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, {}, k=0)


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def test_recency_multiplier_is_bounded_exponential_decay() -> None:
    options = RetrievalOptions()

    assert recency_multiplier(timedelta(0), half_life=options.recency_half_life) == pytest.approx(
        1.0
    )
    # One half-life: half of the recency weight is lost.
    assert recency_multiplier(
        DAY * 30, half_life=options.recency_half_life, weight=0.35
    ) == pytest.approx(0.65 + 0.35 * 0.5)
    # Arbitrarily old: asymptote to 1 - weight, never zero.
    assert recency_multiplier(
        DAY * 3000, half_life=options.recency_half_life, weight=0.35
    ) == pytest.approx(0.65, abs=1e-6)


def test_recency_multiplier_is_monotone_and_ignores_future_timestamps() -> None:
    values = [recency_multiplier(DAY * days) for days in (0, 1, 7, 30, 90)]
    assert values == sorted(values, reverse=True)
    assert recency_multiplier(-DAY * 5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Date extraction and proximity
# ---------------------------------------------------------------------------


def test_find_dates_parses_iso_and_english_month_forms() -> None:
    assert find_dates("Cutover moved to 2026-12-04.", reference_year=2026) == (date(2026, 12, 4),)
    assert find_dates("review on Nov 20", reference_year=2026) == (date(2026, 11, 20),)
    assert find_dates("review on 20 November 2027", reference_year=2026) == (date(2027, 11, 20),)
    assert find_dates("window 2026-11-17..2026-11-23", reference_year=2026) == (
        date(2026, 11, 17),
        date(2026, 11, 23),
    )


def test_find_dates_ignores_keys_and_impossible_dates() -> None:
    assert find_dates("PHX-150 Ledger v2 rollout", reference_year=2026) == ()
    assert find_dates("2026-02-30 is not a date", reference_year=2026) == ()
    assert find_dates("", reference_year=2026) == ()


def test_date_proximity_bonus_rewards_closer_dates() -> None:
    query = [date(2026, 12, 4)]

    exact = date_proximity_multiplier(query, [date(2026, 12, 4)])
    one_week = date_proximity_multiplier(query, [date(2026, 11, 27)])
    far = date_proximity_multiplier(query, [date(2026, 1, 1)])

    assert exact == pytest.approx(1.25)
    assert exact > one_week > far > 1.0
    # No date in the query => no bonus at all.
    assert date_proximity_multiplier([], [date(2026, 12, 4)]) == 1.0
    assert date_proximity_multiplier(query, []) == 1.0


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


def test_entity_needles_expand_multi_word_terms() -> None:
    assert entity_needles(["Project Phoenix"]) == ["project phoenix", "project", "phoenix"]
    assert entity_needles(["PHX", "PHX"]) == ["phx"]
    assert entity_needles(["  ", ""]) == []


def test_entity_multiplier_counts_matched_terms() -> None:
    fields = [
        "PHX-150 — Cut over billing webhooks to Ledger v2",
        "PHX-150",
        "phoenix",
        "description",
    ]

    # "Project Phoenix" matches via the "phoenix" label; "Ledger v2" via the title.
    assert entity_multiplier(["Project Phoenix", "Ledger v2"], fields=fields, weight=0.5) == (
        pytest.approx(1.5)
    )
    assert entity_multiplier(["Ledger v2"], fields=fields, weight=0.5) == pytest.approx(1.5)
    assert entity_multiplier(["Atlas"], fields=fields, weight=0.5) == 1.0
    assert entity_multiplier([], fields=fields, weight=0.5) == 1.0
    assert entity_multiplier(["Atlas"], fields=[], weight=0.5) == 1.0


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def test_text_overlap_is_jaccard_of_words() -> None:
    assert text_overlap("alpha beta gamma", "alpha beta gamma") == 1.0
    assert text_overlap("alpha beta", "gamma delta") == 0.0
    assert text_overlap("", "anything") == 0.0
    assert text_overlap("alpha beta gamma", "alpha beta delta") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


def test_rank_candidates_orders_by_score_and_annotates_staleness() -> None:
    fresh = candidate("fresh", vector_rank=1, fts_rank=1)
    stale = candidate(
        "stale",
        document_id="doc-2",
        updated_at=NOW - DAY * 10,
        vector_rank=2,
        fts_rank=2,
    )

    outcome = rank_candidates(
        [stale, fresh], now=NOW, query_text="ledger cutover", freshness_window=DAY
    )

    assert [item.candidate.chunk_id for item in outcome.chunks] == ["fresh", "stale"]
    assert outcome.chunks[0].stale is False
    assert outcome.chunks[1].stale is True
    assert outcome.stale_chunk_ids == ["stale"]
    # Stale chunks are returned, not dropped: the planner decides.
    assert outcome.dropped_below_floor == 0


def test_rank_candidates_prefers_the_newer_of_equally_relevant_chunks() -> None:
    older = candidate("older", document_id="doc-old", updated_at=NOW - DAY * 200, fts_rank=1)
    newer = candidate("newer", document_id="doc-new", updated_at=NOW, fts_rank=2)

    outcome = rank_candidates([older, newer], now=NOW, query_text="ledger cutover")

    # The rank difference is 2%; the recency difference is far larger.
    assert [item.candidate.chunk_id for item in outcome.chunks] == ["newer", "older"]


def test_rank_candidates_applies_date_proximity_bonus_only_for_date_queries() -> None:
    december = candidate(
        "december",
        text="Cutover moved to 2026-12-04 after the Nimbus delay",
        updated_at=NOW - DAY * 300,
        vector_rank=1,
    )
    november = candidate(
        "november",
        document_id="doc-2",
        text="Cutover moved to 2026-11-20 after the Nimbus delay",
        updated_at=NOW - DAY * 300,
        fts_rank=1,
    )

    dated = rank_candidates(
        [november, december], now=NOW, query_text="Is the cutover on 2026-12-04?"
    )
    undated = rank_candidates([november, december], now=NOW, query_text="When is the cutover?")

    assert dated.chunks[0].candidate.chunk_id == "december"
    assert dated.chunks[0].score > undated.chunks[0].score


def test_rank_candidates_applies_entity_boost() -> None:
    phoenix = candidate("phoenix", vector_rank=1, fts_rank=1)
    atlas = candidate(
        "atlas",
        document_id="doc-2",
        source_id="ATL-88",
        title="ATL-88 — Warehouse ETL decommission",
        labels=("atlas",),
        vector_rank=1,
    )

    outcome = rank_candidates(
        [atlas, phoenix],
        now=NOW,
        query_text="ledger cutover",
        entity_terms=["Phoenix"],
    )

    assert outcome.chunks[0].candidate.chunk_id == "phoenix"
    assert outcome.chunks[0].score > outcome.chunks[1].score


def test_rank_candidates_collapses_near_duplicates_same_document_and_heading() -> None:
    body = (
        "The cutover to Ledger v2 is blocked on the Nimbus webhook schema upgrade. "
        "Nimbus has not shipped schema v3, so the billing webhook replay test cannot run, "
        "and the cutover date stays provisional until the vendor delivers."
    )
    first = candidate("first", text=body, vector_rank=1, fts_rank=1)
    duplicate = candidate("duplicate", text=body, vector_rank=2, fts_rank=2)

    outcome = rank_candidates([first, duplicate], now=NOW, query_text="ledger cutover")

    assert [item.candidate.chunk_id for item in outcome.chunks] == ["first"]
    assert outcome.deduped == 1


def test_rank_candidates_keeps_similar_text_under_a_different_heading() -> None:
    body = "Nimbus has not shipped schema v3, so the webhook replay test cannot run."
    description = candidate("description", text=body, heading_path=("description",), fts_rank=1)
    comment = candidate(
        "comment",
        text=body,
        document_id="doc-1",
        heading_path=("comment",),
        fts_rank=2,
    )

    outcome = rank_candidates([description, comment], now=NOW, query_text="nimbus schema")

    assert {item.candidate.chunk_id for item in outcome.chunks} == {"description", "comment"}
    assert outcome.deduped == 0


def test_rank_candidates_caps_chunks_per_document() -> None:
    candidates = [
        candidate(
            f"c{index}",
            text=f"distinct section {index} about topic{index}",
            heading_path=(f"h{index}",),
            vector_rank=index + 1,
            fts_rank=index + 1,
        )
        for index in range(6)
    ]
    options = RetrievalOptions(max_chunks_per_document=2)

    outcome = rank_candidates(candidates, options=options, now=NOW, query_text="topic")

    assert len(outcome.chunks) == 2
    assert outcome.deduped == 4


def test_rank_candidates_drops_tail_below_score_floor() -> None:
    strong = candidate("strong", vector_rank=1, fts_rank=1)  # 2/61 ≈ 0.0328
    weak = candidate("weak", document_id="doc-2", fts_rank=40)  # 1/100 = 0.01

    outcome = rank_candidates(
        [strong, weak], options=RetrievalOptions(score_floor=0.02), now=NOW
    )

    assert [item.candidate.chunk_id for item in outcome.chunks] == ["strong"]
    assert outcome.dropped_below_floor == 1


def test_rank_candidates_is_deterministic_on_ties() -> None:
    left = candidate("a", vector_rank=1)
    right = candidate("b", document_id="doc-2", vector_rank=1)

    first = rank_candidates([left, right], now=NOW)
    second = rank_candidates([right, left], now=NOW)

    assert [item.candidate.chunk_id for item in first.chunks] == ["a", "b"]
    assert [item.candidate.chunk_id for item in second.chunks] == ["a", "b"]


def test_rank_candidates_with_no_candidates_returns_empty_outcome() -> None:
    outcome = rank_candidates([], now=NOW, query_text="anything")

    assert outcome.chunks == []
    assert outcome.dropped_below_floor == 0
    assert outcome.deduped == 0
    assert outcome.stale_chunk_ids == []
