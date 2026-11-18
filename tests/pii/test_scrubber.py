"""End-to-end scrubbing behavior against the shared fixture corpus.

The corpus plants every PII class on purpose (``fixtures/corpus/manifest.json``);
these tests prove none of it survives scrubbing, that placeholders stay stable
across chunks, and that ordinary work data (titles, projects, deadlines) does.
"""

from __future__ import annotations

import pytest
from pii_fixtures import make_chunk, make_pack

from orchestrator.pipeline import EvidencePack
from pii import scrub

pytestmark = pytest.mark.usefixtures("engines")


async def test_no_planted_pii_survives_the_corpus(
    corpus_evidence_pack: EvidencePack,
    person_names: list[str],
    planted_pii: list[str],
) -> None:
    result = await scrub(
        corpus_evidence_pack, trace_id="trace-corpus", known_person_names=person_names
    )

    blob = "\n".join(f"{chunk.title}\n{chunk.text}" for chunk in result.chunks).casefold()
    leaked = [value for value in planted_pii if value.casefold() in blob]
    assert leaked == [], f"raw PII survived scrubbing: {leaked}"
    assert result.redaction_count > 0
    assert any(chunk.redacted_types for chunk in result.chunks)


async def test_pack_shape_is_preserved(
    corpus_evidence_pack: EvidencePack, person_names: list[str]
) -> None:
    result = await scrub(
        corpus_evidence_pack, trace_id="trace-shape", known_person_names=person_names
    )

    assert result.question == corpus_evidence_pack.question
    assert result.degraded_reason == corpus_evidence_pack.degraded_reason
    assert [c.chunk_id for c in result.chunks] == [
        c.chunk_id for c in corpus_evidence_pack.chunks
    ]
    for original, scrubbed in zip(
        corpus_evidence_pack.chunks, result.chunks, strict=True
    ):
        assert scrubbed.document_id == original.document_id
        assert scrubbed.score == original.score
        assert scrubbed.source == original.source
        assert scrubbed.source_id == original.source_id
        assert scrubbed.url == original.url
        assert scrubbed.updated_at == original.updated_at
        assert scrubbed.stale == original.stale
        assert scrubbed.meta == original.meta


async def test_placeholders_are_stable_across_chunks(person_names: list[str]) -> None:
    pack = make_pack(
        "Priya Raghunathan owns the weekly status report.",
        "Escalate to Priya Raghunathan at priya.raghunathan@northwind.example.",
        "Priya is on +1-415-555-0142.",
    )

    result = await scrub(pack, trace_id="trace-stable", known_person_names=person_names)
    texts = [chunk.text for chunk in result.chunks]

    # Same person, three mention forms -> one placeholder.
    assert texts[0] == "[PERSON_1] owns the weekly status report."
    assert "[PERSON_1]" in texts[1]
    assert texts[2] == "[PERSON_1] is on [PHONE_1]."
    # Same email value re-used later keeps its number.
    assert "[EMAIL_1]" in texts[1]

    repeat = await scrub(
        make_pack("Priya Raghunathan again.", "Daniel Okonkwo now."),
        trace_id="trace-order",
        known_person_names=person_names,
    )
    assert [chunk.text for chunk in repeat.chunks] == [
        "[PERSON_1] again.",
        "[PERSON_2] now.",
    ]


async def test_redacted_types_and_count_match_the_redactions(
    person_names: list[str],
) -> None:
    pack = make_pack(
        "Reach Priya Raghunathan at priya.raghunathan@northwind.example "
        "or +1-415-555-0142."
    )

    result = await scrub(pack, trace_id="trace-count", known_person_names=person_names)

    assert result.redaction_count == 3
    assert result.chunks[0].redacted_types == ["email", "person", "phone"]


async def test_titles_are_scrubbed_too(person_names: list[str]) -> None:
    pack = EvidencePack(
        question="Who runs the migration?",
        chunks=[
            make_chunk(
                "Owned by Mei-Ling Zhou.",
                title="1:1 notes — Mei-Ling Zhou",
            )
        ],
    )

    result = await scrub(pack, trace_id="trace-title", known_person_names=person_names)

    assert "Mei-Ling Zhou" not in result.chunks[0].title
    assert result.chunks[0].title == "1:1 notes — [PERSON_1]"
    assert result.chunks[0].text == "Owned by [PERSON_1]."
    assert result.redaction_count == 2


async def test_first_name_and_full_name_share_a_placeholder(
    person_names: list[str],
) -> None:
    pack = make_pack(
        "Ravi Chandrasekaran is the billing analyst.",
        "Ravi owns reconciliation and finance sign-off.",
    )

    result = await scrub(pack, trace_id="trace-coref", known_person_names=person_names)

    assert result.chunks[0].text == "[PERSON_1] is the billing analyst."
    assert result.chunks[1].text == "[PERSON_1] owns reconciliation and finance sign-off."


async def test_scrub_without_registry_still_redacts_name_shaped_people() -> None:
    """Without a registry, precision drops but nothing name-shaped is passed through."""
    pack = make_pack("Weekly sync with Grace Adeyemi and the billing team.")

    result = await scrub(pack, trace_id="trace-ner")

    assert "Grace Adeyemi" not in result.chunks[0].text
    assert "[PERSON_1]" in result.chunks[0].text
    assert "the billing team" in result.chunks[0].text
