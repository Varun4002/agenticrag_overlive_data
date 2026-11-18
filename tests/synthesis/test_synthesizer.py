"""Synthesis: citations are mandatory, refusals are deterministic."""

from __future__ import annotations

import pytest
from synth_helpers import FakeClient, chunk, claim, pack, verified

from config import Settings
from db.tracing import InMemoryEventSink
from orchestrator.pipeline import SynthesisResult
from synthesis.prompts import MAX_OUTPUT_TOKENS
from synthesis.synthesizer import (
    META_CITATION_PHRASES,
    QUOTE_CHARS,
    _best_supporting_chunk,
    _dedupe_lines,
    _is_decorative_line,
    _is_negated,
    _line_key,
    _quote_for,
    _strip_meta_citation,
    synthesize,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


async def test_cited_answer_is_kept_and_citations_resolved() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient("The PHX-150 cutover is scheduled for 2026-11-20 [1].")

    result = await synthesize(
        "When is the PHX-150 cutover?", verified_, pack_, client=client, settings=_settings()
    )

    assert result.refused is False
    assert "[1]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[0].source == "jira"
    assert result.used_claim_ids == [0]
    assert result.refusal_reason is None


async def test_uncited_factual_sentence_is_stripped() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient(
        "The cutover was delayed twice by the vendor. "
        "The PHX-150 cutover is 2026-11-20 [1]."
    )

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert "delayed twice" not in result.answer
    assert "2026-11-20" in result.answer
    assert result.refused is False


async def test_answer_with_no_supported_sentences_becomes_refusal() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    sink = InMemoryEventSink()
    client = FakeClient("I believe the cutover is soon and everything will be fine.")

    result = await synthesize(
        "q",
        verified_,
        pack_,
        trace_id="trace-refuse",
        client=client,
        sink=sink,
        settings=_settings(),
    )

    assert result.refused is True
    assert result.citations == []
    assert "can't answer" in result.answer
    assert result.refusal_reason == "the draft answer could not be tied back to the evidence"
    assert "synthesis.refused" in [event.event for event in sink.events_for("trace-refuse")]


async def test_unanswerable_evidence_refuses_without_calling_the_model() -> None:
    client = FakeClient("this should never be used")
    result = await synthesize(
        "q",
        verified(claim("nothing", "c1"), can_answer=False),
        pack(chunk("c1", "irrelevant")),
        client=client,
        settings=_settings(),
    )

    assert client.requests == []
    assert result.refused is True
    assert "coverage 0.10" in result.answer
    assert result.refusal_reason == (
        "insufficient evidence: question coverage 0.10 is below the 0.50 threshold"
    )


async def test_invalid_marker_is_removed_and_sentence_dropped() -> None:
    pack_ = pack(chunk("c1", "PHX-131 is blocked waiting on the vendor contract."))
    verified_ = verified(claim("PHX-131 is blocked", "c1"))
    client = FakeClient("The task is blocked by the vendor contract [9].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert client.requests  # model was called; the answer just failed validation


async def test_markers_are_renumbered_to_dense_indices() -> None:
    pack_ = pack(
        chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."),
        chunk("c2", "PHX-131 is blocked waiting on the vendor contract.", source_id="PHX-131"),
    )
    verified_ = verified(
        claim("PHX-150 cutover is 2026-11-20", "c1", claim_id=0),
        claim("PHX-131 is blocked", "c2", claim_id=1),
    )
    # Only the second claim is cited; its marker must become [1].
    client = FakeClient("PHX-131 is blocked by the vendor [2].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.answer.strip() == "PHX-131 is blocked by the vendor [1]."
    assert result.used_claim_ids == [1]
    assert [citation.chunk_id for citation in result.citations] == ["c2"]


async def test_placeholders_are_passed_through_verbatim() -> None:
    pack_ = pack(chunk("c1", "PHX-150 is owned by [PERSON_1] and due 2026-11-20."))
    verified_ = verified(claim("PHX-150 is owned by [PERSON_1] and due 2026-11-20", "c1"))
    client = FakeClient("PHX-150 is owned by [PERSON_1] and due 2026-11-20 [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert "[PERSON_1]" in result.answer
    assert "[1]" in result.answer


async def test_trailing_citation_marker_attaches_to_the_previous_sentence() -> None:
    """`... because the vendor slipped. [1]` is one cited sentence, not two."""
    pack_ = pack(chunk("c1", "PHX-150 cutover moved to 2026-11-27 because Nimbus slipped."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-27", "c1"))
    client = FakeClient("The cutover moved to 2026-11-27 because Nimbus slipped. [1]")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert "Nimbus slipped" in result.answer
    assert result.answer.strip().endswith("[1]")
    assert len(result.citations) == 1


async def test_marker_only_answer_refuses_instead_of_answering_with_brackets() -> None:
    """A model that emits only `[1]` has not answered anything."""
    pack_ = pack(chunk("c1", "PHX-150 cutover is 2026-11-27."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-27", "c1"))
    client = FakeClient("[1]")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert result.citations == []
    assert "[" not in result.answer or "can't answer" in result.answer


async def test_multi_chunk_claim_cites_the_chunk_that_supports_it() -> None:
    """chunk_ids is evidence, not a priority list: cite the chunk that says the value."""
    pack_ = pack(
        chunk("c-old", "The Jira due date still says 2026-11-20 for PHX-150."),
        chunk("c-new", "PHX-150 cutover moved to 2026-12-04, per the vendor email."),
    )
    # Old id listed first on purpose: list order must not decide the citation.
    verified_ = verified(claim("The cutover date for PHX-150 is 2026-12-04", "c-old", "c-new"))
    client = FakeClient("The cutover date for PHX-150 is 2026-12-04 [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert result.citations[0].chunk_id == "c-new"
    assert "2026-12-04" in result.citations[0].quote


async def test_citation_tie_break_is_first_listed_chunk() -> None:
    text = "PHX-150 cutover date is 2026-11-20 per the Jira field."
    pack_ = pack(chunk("c-first", text), chunk("c-second", text))
    verified_ = verified(claim("PHX-150 cutover date is 2026-11-20", "c-first", "c-second"))
    client = FakeClient("The PHX-150 cutover is 2026-11-20 [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.citations[0].chunk_id == "c-first"


async def test_single_chunk_claim_citation_unchanged() -> None:
    pack_ = pack(chunk("c1", "PHX-131 is blocked waiting on the vendor contract."))
    verified_ = verified(claim("PHX-131 is blocked", "c1"))
    client = FakeClient("PHX-131 is blocked by the vendor contract [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert [citation.chunk_id for citation in result.citations] == ["c1"]


def test_best_supporting_chunk_returns_none_when_no_candidate_is_in_the_pack() -> None:
    by_id = {"c1": chunk("c1", "PHX-150 cutover is 2026-11-20.")}
    assert _best_supporting_chunk(claim("PHX-150 cutover is 2026-11-20", "gone"), by_id) is None


async def test_citation_dump_line_is_not_the_answer() -> None:
    """A bare `Citations: [1] [2] [3]` line keeps its markers but says nothing."""
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient(
        "Based on the verified claims, the status report on Project Phoenix is as follows:\n"
        "Citations: [1] [2] [3]"
    )

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert len(client.requests) == 1  # the model was asked; its dump just failed
    assert result.citations == []
    assert "[" not in result.answer


async def test_duplicated_reference_line_is_dropped() -> None:
    """A model that repeats its own sentence after the marker has not said it twice."""
    body = (
        "The cutover date for PHX-150 is 2026-12-04, and it has changed "
        "from the originally reported 2026-11-20"
    )
    pack_ = pack(chunk("c1", "PHX-150 cutover moved to 2026-12-04, per the vendor email."))
    verified_ = verified(claim("The cutover date for PHX-150 is 2026-12-04", "c1"))
    client = FakeClient(f"{body} [1].\n[1] {body}.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert result.answer.count("changed from the originally reported") == 1
    assert len(result.answer.splitlines()) == 1
    assert "\n[1] " not in result.answer
    assert len(result.citations) == 1


def test_is_decorative_line_flags_a_marker_dump() -> None:
    assert _is_decorative_line("Citations: [1] [2] [3]") is True
    assert _is_decorative_line("REFERENCES: [1]") is True
    assert _is_decorative_line("[1] [2]") is True
    with_content = "Citations: [1] The cutover date for PHX-150 is 2026-12-04."
    assert _is_decorative_line(with_content) is False


def test_line_key_ignores_markers_punctuation_and_case() -> None:
    marked = "The cutover date for PHX-150 is 2026-12-04 [1]."
    reference = "[1] The cutover date for PHX-150 is 2026-12-04."
    assert _line_key(marked) == _line_key(reference)
    assert _line_key("THE  Cutover [2].") == _line_key("the cutover")


def test_dedupe_lines_keeps_the_first_occurrence() -> None:
    kept, duplicates = _dedupe_lines(["a b [1]", "A B.", "c [2]"])
    assert kept == ["a b [1]", "c [2]"]
    assert duplicates == ["A B."]


def test_best_supporting_chunk_keeps_first_candidate_without_overlap() -> None:
    """Zero overlap must still yield a citation, not a dropped claim."""
    by_id = {
        "c1": chunk("c1", "Nothing here matches the assertion."),
        "c2": chunk("c2", "Also unrelated text."),
    }
    chosen = _best_supporting_chunk(claim("wholly different words", "c1", "c2"), by_id)
    assert chosen is not None
    assert chosen.chunk_id == "c1"


# ---------------------------------------------------------------------------
# Citation quote: the window that supports the claim, not a blind prefix
# ---------------------------------------------------------------------------

SUPPORT = "The Nimbus cutover moved to 2026-12-04 after Finance signed the renewal."
FILLER = "The weekly status sync covered unrelated scheduling noise."


def _long_chunk_text() -> str:
    """A chunk whose support sits well past the first ``QUOTE_CHARS`` characters."""
    text = " ".join([FILLER] * 15) + " " + SUPPORT + " " + " ".join([FILLER] * 5)
    assert len(text) >= 1200
    assert text.index(SUPPORT) > QUOTE_CHARS
    return text


async def test_quote_contains_the_supporting_sentence_from_deep_in_the_chunk() -> None:
    """Live defect: the delivered quote was the head, so it did not support the claim."""
    text = _long_chunk_text()
    pack_ = pack(chunk("c1", text))
    verified_ = verified(claim("The Nimbus cutover moved to 2026-12-04", "c1"))
    client = FakeClient("The Nimbus cutover moved to 2026-12-04 [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    quote = result.citations[0].quote
    assert "2026-12-04" in quote
    assert "Finance signed the renewal" in quote
    assert len(quote) <= QUOTE_CHARS
    assert "2026-12-04" not in text[:QUOTE_CHARS]  # the old prefix could not have had it


async def test_quote_falls_back_to_the_chunk_head_when_nothing_overlaps() -> None:
    text = " ".join([FILLER] * 10)
    pack_ = pack(chunk("c1", text))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient("PHX-150 cutover is 2026-11-20 [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert result.citations[0].quote == text[:QUOTE_CHARS]


def test_quote_for_trims_at_a_word_boundary() -> None:
    """A chunk with no sentence break must not ship half a word."""
    text = " ".join(["unbroken"] * 60)  # 539 chars, no full stop
    quote = _quote_for(text, "unbroken")

    assert 0 < len(quote) <= QUOTE_CHARS
    assert not quote.endswith("unbroke")
    assert quote.split() == ["unbroken"] * len(quote.split())
    # The split really is on whitespace and the tail is untouched.
    assert text.startswith(quote)


# ---------------------------------------------------------------------------
# Meta narration about the marker ("as indicated in claim [1]")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", META_CITATION_PHRASES)
def test_meta_citation_phrase_is_stripped_and_the_marker_kept(phrase: str) -> None:
    line = f"PHX-142 waits on Okta admin access, {phrase} [1]."

    assert _strip_meta_citation(line) == "PHX-142 waits on Okta admin access [1]."


def test_meta_citation_phrase_with_an_article_is_stripped() -> None:
    line = "PHX-142 waits on Okta admin access as indicated in the claim [1]."

    assert _strip_meta_citation(line) == "PHX-142 waits on Okta admin access [1]."


def test_parenthetical_claim_is_stripped_to_its_marker() -> None:
    assert _strip_meta_citation("PHX-142 is blocked (claim [1]).") == "PHX-142 is blocked [1]."
    assert _strip_meta_citation("PHX-142 is blocked (claim).") == "PHX-142 is blocked."


def test_text_without_meta_narration_is_returned_unchanged() -> None:
    """The tidy-up must not rewrite ordinary prose (commas, double spaces)."""
    line = "PHX-142 is blocked,  pending  review [1]."

    assert _strip_meta_citation(line) == line


async def test_live_meta_narration_is_stripped_from_the_answer() -> None:
    """Live answer: `"...access issues, as indicated in claim [1]."`."""
    body = "PHX-142 is waiting on the resolution of Okta admin access issues"
    pack_ = pack(chunk("c1", f"{body}."))
    verified_ = verified(claim(f"{body}.", "c1"))
    client = FakeClient(f"{body}, as indicated in claim [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert result.answer == f"{body} [1]."


async def test_answer_that_is_only_meta_narration_and_a_marker_refuses() -> None:
    text = "PHX-142 is waiting on the resolution of Okta admin access issues."
    pack_ = pack(chunk("c1", text))
    verified_ = verified(claim(text, "c1"))
    client = FakeClient("As indicated in claim [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "[" not in result.answer



# ---------------------------------------------------------------------------
# Marker resolution, citation dedupe, truncated completions
# ---------------------------------------------------------------------------


async def test_non_contiguous_claim_id_is_reported_as_is() -> None:
    """Markers are positions; ``used_claim_ids`` must report real ``claim_id``s.

    The old code returned ``marker - 1``, which silently reported a list index
    as a claim id whenever the ids were not ``0..n``.
    """
    pack_ = pack(chunk("c2", "PHX-131 is blocked waiting on the vendor contract."))
    verified_ = verified(claim("PHX-131 is blocked", "c2", claim_id=5))
    client = FakeClient("PHX-131 is blocked by the vendor contract [1].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert [citation.chunk_id for citation in result.citations] == ["c2"]
    assert result.used_claim_ids == [5]


async def test_claims_sharing_a_chunk_yield_one_citation() -> None:
    """Two claims on the same chunk are one delivered citation, still cited twice."""
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(
        claim("PHX-150 cutover is 2026-11-20", "c1", claim_id=0),
        claim("PHX-150 cutover date is 2026-11-20", "c1", claim_id=1),
    )
    client = FakeClient("The cutover is 2026-11-20 [1] and was confirmed [2].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert [citation.chunk_id for citation in result.citations] == ["c1"]
    # Both markers now point at the single delivered citation.
    assert result.answer.strip().count("[1]") == 2
    assert "[2]" not in result.answer
    assert result.used_claim_ids == [0, 1]


async def test_dedupe_keeps_first_use_order_and_renumbers_around_it() -> None:
    pack_ = pack(
        chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."),
        chunk("c2", "PHX-131 is blocked waiting on the vendor contract.", source_id="PHX-131"),
    )
    verified_ = verified(
        claim("PHX-150 cutover is 2026-11-20", "c1", claim_id=0),
        claim("PHX-131 is blocked", "c2", claim_id=1),
        claim("PHX-150 cutover is 2026-11-20 per Jira", "c1", claim_id=2),
    )
    client = FakeClient("The cutover is 2026-11-20 [1]. PHX-131 is blocked [2]. Jira agrees [3].")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert [citation.chunk_id for citation in result.citations] == ["c1", "c2"]
    assert result.answer == (
        "The cutover is 2026-11-20 [1]. PHX-131 is blocked [2]. Jira agrees [1]."
    )


async def test_truncated_completion_refuses_instead_of_shipping_a_cut_off_answer() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    sink = InMemoryEventSink()
    client = FakeClient(
        "The PHX-150 cutover is scheduled for 2026-11-20 [1] and the vendor",
        finish_reason="length",
    )

    result = await synthesize(
        "q",
        verified_,
        pack_,
        trace_id="trace-truncated",
        client=client,
        sink=sink,
        settings=_settings(),
    )

    assert result.refused is True
    assert result.citations == []
    assert "can't answer" in result.answer
    assert "vendor" not in result.answer
    assert result.refusal_reason == "answer truncated by the token limit"
    refused = [
        event for event in sink.events_for("trace-truncated") if event.event == "synthesis.refused"
    ]
    assert len(refused) == 1
    assert refused[0].payload["reason"] == "answer truncated by the token limit"
    assert refused[0].payload["finish_reason"] == "length"


async def test_normal_finish_reason_answers_normally() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient("The PHX-150 cutover is 2026-11-20 [1].", finish_reason="stop")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert [citation.chunk_id for citation in result.citations] == ["c1"]
    assert result.refusal_reason is None


def test_output_budget_leaves_room_for_a_six_claim_answer() -> None:
    """1024 cut off a live six-claim answer; the cap must not regress below that."""
    assert MAX_OUTPUT_TOKENS >= 2048


def test_synthesis_result_round_trips_with_and_without_refusal_reason() -> None:
    """The additive field is optional: both shapes validate and round-trip."""
    answered = SynthesisResult(answer="The cutover is 2026-11-27 [1].")
    assert answered.refusal_reason is None
    assert SynthesisResult.model_validate(answered.model_dump()).refusal_reason is None

    refused = SynthesisResult(
        answer="I can't answer that from the evidence I could verify.",
        refused=True,
        refusal_reason="answer truncated by the token limit",
    )
    reloaded = SynthesisResult.model_validate(refused.model_dump())
    assert reloaded.refused is True
    assert reloaded.refusal_reason == "answer truncated by the token limit"


# ---------------------------------------------------------------------------
# Marker-less drafts: deterministic claim matching (ruling #21)
# ---------------------------------------------------------------------------


async def test_markerless_draft_is_cited_to_the_claim_it_restates() -> None:
    """The live 8B incident: a faithful seven-sentence answer with no markers."""
    text = "Google sent three security alert emails on 18 Nov 2026 at 08:12:29 +0000."
    pack_ = pack(chunk("c1", text))
    verified_ = verified(claim(text, "c1"))
    sink = InMemoryEventSink()
    client = FakeClient(text)

    result = await synthesize(
        "q",
        verified_,
        pack_,
        trace_id="trace-markerless",
        client=client,
        sink=sink,
        settings=_settings(),
    )

    assert result.refused is False
    assert result.answer.strip() == f"{text} [1]"
    assert [citation.chunk_id for citation in result.citations] == ["c1"]
    assert result.used_claim_ids == [0]
    completed = [
        event
        for event in sink.events_for("trace-markerless")
        if event.event == "synthesis.completed"
    ]
    assert len(completed) == 1
    assert completed[0].payload["implied_citations"] == 1


async def test_markerless_sentence_below_the_overlap_threshold_is_dropped() -> None:
    pack_ = pack(chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("PHX-150 cutover is 2026-11-20", "c1"))
    client = FakeClient("The cutover was delayed twice because the vendor slipped again.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "delayed twice" not in result.answer
    assert result.refusal_reason == "the draft answer could not be tied back to the evidence"


async def test_markerless_sentence_with_a_contradicting_date_is_dropped() -> None:
    """Numeric guard: 2026-12-04 must not be cited to the claim saying 2026-11-20."""
    pack_ = pack(chunk("c1", "The PHX-150 cutover date is 2026-11-20 per the Jira field."))
    verified_ = verified(claim("The PHX-150 cutover date is 2026-11-20", "c1"))
    client = FakeClient("The PHX-150 cutover date is 2026-12-04.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "2026-12-04" not in result.answer
    assert result.citations == []


async def test_compliant_draft_keeps_its_markers_and_reports_no_implied_citations() -> None:
    pack_ = pack(
        chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."),
        chunk("c2", "PHX-131 is blocked waiting on the vendor contract.", source_id="PHX-131"),
    )
    verified_ = verified(
        claim("PHX-150 cutover is 2026-11-20", "c1", claim_id=0),
        claim("PHX-131 is blocked", "c2", claim_id=1),
    )
    sink = InMemoryEventSink()
    client = FakeClient("The cutover is 2026-11-20 [1]. PHX-131 is blocked [2].")

    result = await synthesize(
        "q",
        verified_,
        pack_,
        trace_id="trace-compliant",
        client=client,
        sink=sink,
        settings=_settings(),
    )

    assert result.answer == "The cutover is 2026-11-20 [1]. PHX-131 is blocked [2]."
    completed = [
        event
        for event in sink.events_for("trace-compliant")
        if event.event == "synthesis.completed"
    ]
    assert completed[0].payload["implied_citations"] == 0


async def test_each_markerless_sentence_gets_its_own_claim_marker() -> None:
    pack_ = pack(
        chunk("c1", "PHX-150 cutover date is 2026-11-20 per the Jira field."),
        chunk("c2", "PHX-131 is blocked waiting on the vendor contract.", source_id="PHX-131"),
    )
    verified_ = verified(
        claim("PHX-150 cutover date is 2026-11-20", "c1", claim_id=0),
        claim("PHX-131 is blocked on the vendor contract", "c2", claim_id=1),
    )
    client = FakeClient(
        "PHX-131 is blocked on the vendor contract. The PHX-150 cutover date is 2026-11-20."
    )

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert "vendor contract. [1]" in result.answer
    assert "2026-11-20. [2]" in result.answer
    assert [citation.chunk_id for citation in result.citations] == ["c2", "c1"]
    assert result.used_claim_ids == [1, 0]


async def test_implied_citation_reports_the_claim_id_not_the_marker_position() -> None:
    """Matching produces a position; ``used_claim_ids`` must still report the id."""
    pack_ = pack(chunk("c2", "PHX-131 is blocked waiting on the vendor contract."))
    verified_ = verified(claim("PHX-131 is blocked on the vendor contract", "c2", claim_id=5))
    client = FakeClient("PHX-131 is blocked on the vendor contract.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is False
    assert result.used_claim_ids == [5]
    assert [citation.chunk_id for citation in result.citations] == ["c2"]


# ---------------------------------------------------------------------------
# Matching guards: overlap threshold, numeric, polarity
# ---------------------------------------------------------------------------


async def test_negated_sentence_is_not_cited_to_a_positive_claim() -> None:
    """Polarity guard: "not delayed" restates the inverse of "delayed"."""
    pack_ = pack(chunk("c1", "The PHX-150 cutover is delayed."))
    verified_ = verified(claim("The PHX-150 cutover is delayed", "c1"))
    client = FakeClient("The PHX-150 cutover is not delayed.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "not delayed" not in result.answer
    assert result.citations == []


async def test_negated_claim_does_not_back_a_positive_sentence() -> None:
    """The inverse direction: a negated claim must not cite a positive sentence."""
    pack_ = pack(chunk("c1", "The PHX-150 cutover is not delayed."))
    verified_ = verified(claim("The PHX-150 cutover is not delayed", "c1"))
    client = FakeClient("The PHX-150 cutover is delayed.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "cutover is delayed" not in result.answer
    assert result.citations == []


async def test_unsupported_tail_is_rejected_at_the_new_threshold() -> None:
    """0.667 overlap: a supported head must not carry the vendor agreement along."""
    pack_ = pack(chunk("c1", "The cutover is 2026-11-20."))
    verified_ = verified(claim("The cutover is 2026-11-20", "c1"))
    client = FakeClient("The cutover is 2026-11-20 and the vendor agreed.")

    result = await synthesize("q", verified_, pack_, client=client, settings=_settings())

    assert result.refused is True
    assert "vendor agreed" not in result.answer
    assert result.citations == []


async def test_live_draft_sentence_scores_above_the_threshold() -> None:
    """The incident's real paraphrase pair scores 0.733, so it is still recovered."""
    claim_text = (
        "You received a security alert email from Google on 18 Nov 2026 at 08:12:29 +0000, "
        "suggesting the notice was routine"
    )
    sentence = (
        "You received three security alert emails from Google on 18 Nov 2026 at different times."
    )
    pack_ = pack(chunk("c1", claim_text))
    verified_ = verified(claim(claim_text, "c1"))
    sink = InMemoryEventSink()
    client = FakeClient(sentence)

    result = await synthesize(
        "q",
        verified_,
        pack_,
        trace_id="trace-live-overlap",
        client=client,
        sink=sink,
        settings=_settings(),
    )

    assert result.refused is False
    assert "different times. [1]" in result.answer
    assert [citation.chunk_id for citation in result.citations] == ["c1"]
    completed = [
        event
        for event in sink.events_for("trace-live-overlap")
        if event.event == "synthesis.completed"
    ]
    assert completed[0].payload["implied_citations"] == 1


async def test_apostrophe_negation_counts_on_either_side() -> None:
    """``_TOKEN_RE`` splits ``isn't`` into ``isn`` + ``t``; polarity must see it whole."""
    assert _is_negated("PHX-131 isn't blocked.") is True
    assert _is_negated("PHX-131 is not blocked.") is True
    assert _is_negated("PHX-131 is blocked.") is False
    positive = chunk("c1", "PHX-131 is blocked by the vendor contract.")

    from_positive_claim = await synthesize(
        "q",
        verified(claim("PHX-131 is blocked by the vendor contract", "c1")),
        pack(positive),
        client=FakeClient("PHX-131 isn't blocked by the vendor contract."),
        settings=_settings(),
    )
    from_negative_claim = await synthesize(
        "q",
        verified(claim("PHX-131 isn't blocked by the vendor contract", "c1")),
        pack(positive),
        client=FakeClient("PHX-131 is blocked by the vendor contract."),
        settings=_settings(),
    )

    # A negated sentence against a positive claim, and the reverse: both refuse.
    assert from_positive_claim.refused is True
    assert from_negative_claim.refused is True

