"""Sufficiency: deterministic coverage scoring, fails closed."""

from __future__ import annotations

import pytest

from evidence.sufficiency import question_aspects, score_sufficiency
from orchestrator.pipeline import Claim


def _claim(text: str) -> Claim:
    return Claim(claim_id=0, text=text, chunk_ids=["c1"])


def test_full_coverage_can_answer() -> None:
    result = score_sufficiency(
        "When is the PHX-150 cutover?", [_claim("PHX-150 cutover is 2026-11-20")]
    )
    assert result.score == pytest.approx(1.0)
    assert result.can_answer is True


def test_partial_coverage_is_below_threshold() -> None:
    result = score_sufficiency("When is the PHX-150 cutover?", [_claim("The cutover moved again")])
    assert result.score < 0.5
    assert result.can_answer is False
    assert "PHX-150" in result.missing


def test_no_claims_scores_zero() -> None:
    result = score_sufficiency("What is blocked on PHX-131?", [])
    assert result.score == 0.0
    assert result.can_answer is False


def test_question_without_aspects_fails_closed() -> None:
    result = score_sufficiency("???", [_claim("PHX-150 cutover is 2026-11-20")])
    assert result.score == 0.0
    assert result.can_answer is False


def test_threshold_is_inclusive() -> None:
    result = score_sufficiency(
        "When is the PHX-150 cutover?",
        [_claim("PHX-150 cutover is 2026-11-20")],
        threshold=1.0,
    )
    assert result.can_answer is True


def test_ticket_keys_outweigh_keywords() -> None:
    aspects = question_aspects("When is the PHX-150 cutover?")
    weights = {aspect.text: aspect.weight for aspect in aspects}
    assert weights["PHX-150"] > weights["cutover"]


def test_keyword_aspects_are_downweighted() -> None:
    # Generic keywords got 1.0 originally; that let answer-shape words alone
    # outvote the entities a question is actually about.
    aspects = question_aspects("When is the PHX-150 cutover?")
    weights = {aspect.text: aspect.weight for aspect in aspects}
    assert weights["cutover"] == pytest.approx(0.25)


_CONTEXT = {"c1": "PHX-142 · phoenix"}


def test_project_name_satisfied_by_cited_chunk_provenance() -> None:
    # Live failure: "Which Project Phoenix tasks are overdue?" refused at 0.22
    # because the claim never repeats "Phoenix"; the chunk it cites does.
    result = score_sufficiency(
        "Which Project Phoenix tasks are overdue?",
        [_claim("PHX-142 is due 2026-11-10")],
        context=_CONTEXT,
    )
    assert result.can_answer is True
    assert "Phoenix" in result.covered


def test_scaffolding_words_are_not_aspects() -> None:
    scaffolding = {"give", "list", "each", "tell", "summarize", "summarise", "show", "please"}
    aspects = question_aspects("Give me a status report on Project Phoenix: what shipped?")
    assert not [aspect for aspect in aspects if aspect.text.lower() in scaffolding]


def test_generic_keywords_cannot_alone_sink_a_question() -> None:
    result = score_sufficiency(
        "Which Project Phoenix tasks are overdue?",
        [_claim("PHX-142 is due 2026-11-10")],
        context={"c1": "phoenix"},
    )
    assert result.can_answer is True
    assert {"overdue", "tasks", "project"} <= set(result.missing)


def test_uncited_chunks_do_not_count_for_coverage() -> None:
    # Nothing cites c9, so its provenance must not rescue missing aspects.
    result = score_sufficiency(
        "Which Project Zenon tasks are overdue?",
        [_claim("PHX-142 is due 2026-11-10")],
        context={"c1": "phoenix", "c9": "zenon"},
    )
    assert result.can_answer is False
    assert "Zenon" in result.missing


def test_ticket_key_missing_still_refuses() -> None:
    # The gate must still bite: provenance coverage must not invent a ticket.
    result = score_sufficiency(
        "What is the PHX-150 cutover?",
        [_claim("The cutover moved to 2026-11-20")],
        context={"c1": "phoenix billing cutover"},
    )
    assert result.can_answer is False
    assert "PHX-150" in result.missing
