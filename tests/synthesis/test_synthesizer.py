"""Synthesis: citations are mandatory, refusals are deterministic."""

from __future__ import annotations

from synth_helpers import FakeClient, chunk, claim, pack, verified

from config import Settings
from db.tracing import InMemoryEventSink
from synthesis.synthesizer import (
    _best_supporting_chunk,
    _dedupe_lines,
    _is_decorative_line,
    _line_key,
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

