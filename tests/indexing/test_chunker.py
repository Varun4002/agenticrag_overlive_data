"""Chunker tests: golden boundaries per document type, meta, determinism."""

from __future__ import annotations

import hashlib

import pytest

from evals.fixtures import FixtureCorpus, FixtureDocument, load_fixture_corpus
from indexing.chunker import (
    REQUIRED_META_KEYS,
    Chunk,
    Chunker,
    SourceDocument,
    chunk_document,
    strip_email_quotes,
)
from indexing.tokens import estimate_tokens

CORPUS: FixtureCorpus = load_fixture_corpus()


def source_document(doc: FixtureDocument) -> SourceDocument:
    """What the indexer builds from a ``core.documents`` row."""
    return SourceDocument(
        source=doc.source,
        source_id=doc.source_id,
        doc_type=doc.doc_type,
        title=doc.title,
        content=doc.content,
        document_id=doc.key,
        parent_source_id=doc.parent_source_id,
        author=doc.author,
        url=doc.url,
        labels=tuple(doc.labels),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def chunks_of(source: str, source_id: str) -> list[Chunk]:
    return chunk_document(source_document(CORPUS.get(source, source_id)))


SENTENCE = (
    "The migration job copies account records in batches of fifty thousand rows and "
    "writes a checkpoint after every batch."
)


def long_page(sections: int = 3, sentences: int = 6) -> SourceDocument:
    paragraphs = [
        f"{SENTENCE} Part {index}.{part}." for index in range(sections) for part in range(sentences)
    ]
    body = "\n\n".join(paragraphs)
    return SourceDocument(
        source="notion",
        source_id="synthetic-page",
        doc_type="page",
        title="Synthetic Page",
        content=f"# Synthetic Page\n\n{body}",
    )


# ---------------------------------------------------------------------------
# Jira
# ---------------------------------------------------------------------------


def test_jira_issue_splits_field_card_from_description() -> None:
    chunks = chunks_of("jira", "PHX-150")
    assert len(chunks) >= 2, "field card and description must be separate chunks"

    card, description = chunks[0], chunks[1]
    assert card.meta["section"] == "fields"
    assert "Key: PHX-150" in card.text
    assert "Due date: 2026-11-20" in card.text
    assert "Status: In Progress" in card.text
    assert "Dual-write both consumers" not in card.text

    assert description.meta["section"] == "description"
    assert "Dual-write both consumers" in description.text
    assert card.meta["doc_type"] == "issue"
    assert card.meta["source"] == "jira"
    assert card.meta["heading_path"] == []


def test_jira_field_chunk_exposes_structured_fields_and_labels() -> None:
    card = chunks_of("jira", "PHX-150")[0]
    fields = card.meta["fields"]
    assert fields["Status"] == "In Progress"
    assert fields["Assignee"] == "Daniel Okonkwo"
    assert fields["Due date"] == "2026-11-20"
    # Labels come from the issue field block *and* the document ACL labels.
    assert {"billing", "webhooks", "cutover", "phoenix"} <= set(card.meta["labels"])


def test_jira_issue_without_field_block_falls_back_to_description() -> None:
    doc = SourceDocument(
        source="jira",
        source_id="PHX-999",
        doc_type="issue",
        title="PHX-999 — No card",
        content="Just prose with a colon: and no field block at all.\n\nSecond paragraph.",
    )
    chunks = chunk_document(doc)
    assert [chunk.meta["section"] for chunk in chunks] == ["description"]
    assert "Just prose with a colon" in chunks[0].text


def test_jira_comment_uses_issue_key_context() -> None:
    chunks = chunks_of("jira", "PHX-150#c10432")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.meta["section"] == "comment"
    assert chunk.text.startswith("PHX-150 · comment · Slipping the cutover to 2026-11-27")
    assert "We are moving the webhook cutover to 2026-11-27" in chunk.text
    assert chunk.meta["parent_source_id"] == "PHX-150"


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------


def test_notion_sections_never_merge_across_headings() -> None:
    chunks = [
        chunk
        for chunk in chunks_of("notion", "5d9f2c41-7b3e-4a18-9c62-0b1d8e4f7a10")
        if chunk.meta["section"] == "heading"
    ]
    paths = [tuple(chunk.meta["heading_path"]) for chunk in chunks]
    # Heading paths are level-aware and rooted at the page's own H1, which is
    # also the document title (deduped out of the visible breadcrumb).
    assert ("Phoenix — Goals and OKRs FY26", "Guardrails") in paths
    assert (
        "Phoenix — Goals and OKRs FY26",
        "Q4 objectives",
        "Ship Ledger v2 to general availability",
    ) in paths

    # A section body must not leak the next heading's content.
    guardrails = next(c for c in chunks if c.meta["heading_path"][-1] == "Guardrails")
    assert "No account id re-keying" in guardrails.text
    assert "Priya Raghunathan is the Phoenix program manager" not in guardrails.text
    for chunk in chunks:
        assert "No account id re-keying" not in chunk.text or "Stakeholders" not in chunk.text


def test_notion_breadcrumb_carries_heading_path_without_markdown_noise() -> None:
    purpose = chunks_of("notion", "d1a9e6b3-2c58-4f70-9b4d-7e8a3c6f0d21")[0]
    assert purpose.meta["heading_path"][-1] == "Purpose"
    assert purpose.text.startswith("Alice Migration Cutover Plan · Purpose\n\n")
    assert "Alice Migration is the internal codename" in purpose.text
    assert not purpose.text.lstrip().startswith("#")


def test_notion_long_section_is_split_with_overlap() -> None:
    chunks = chunk_document(long_page(sections=1, sentences=40), chunker=Chunker())
    assert len(chunks) > 1
    assert all(chunk.token_count <= 500 for chunk in chunks)

    previous_body = chunks[0].text.split("\n\n", 1)[1]
    next_body = chunks[1].text.split("\n\n", 1)[1]
    last_paragraph = previous_body.split("\n\n")[-1]
    assert last_paragraph in next_body, "the tail of chunk N must reappear at the start of N+1"


def test_page_section_document_chunks_under_its_own_title() -> None:
    chunks = chunks_of("notion", "f2d6a8c1-4e79-4b05-8a3f-1d9c7e5b2a83")
    assert chunks
    assert chunks[0].meta["doc_type"] == "page_section"
    assert chunks[0].text.startswith("Runbook: webhook replay")
    assert chunks[0].meta["parent_source_id"] == "e4b7c1d9-8f26-4a53-b0e9-5c2d8a7f1b46"


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_strips_gmail_wrote_block_and_quoted_lines() -> None:
    chunks = chunks_of("gmail", "<CAF8x2b1.phoenix-cutover.2@mail.northwind.example>")
    assert len(chunks) == 1
    text = chunks[0].text
    assert "2026-11-20 is no longer realistic" in text
    assert "Before Thursday's review" not in text, "quoted reply must be stripped"
    assert "wrote:" not in text
    assert "Priya Raghunathan <priya.raghunathan@northwind.example>" not in text


def test_email_strips_original_message_header_block() -> None:
    chunks = chunks_of("gmail", "<CAF8x2b1.phoenix-cutover.4@mail.northwind.example>")
    text = chunks[0].text
    assert "Forwarding the original thread" in text
    assert "-----Original Message-----" not in text
    assert "From: Daniel Okonkwo" not in text


def test_email_strips_signature_after_dashdash() -> None:
    doc = SourceDocument(
        source="gmail",
        source_id="<sig@example>",
        doc_type="email",
        title="Signature test",
        content="Short body.\n\n-- \nPriya Raghunathan\nProgram Manager\n+1-415-555-0142\n",
    )
    chunks = chunk_document(doc)
    assert chunks[0].text.startswith("Email · Signature test")
    assert "Short body." in chunks[0].text
    assert "Program Manager" not in chunks[0].text


def test_strip_email_quotes_is_public_and_handles_forwarded_chain() -> None:
    body = "New text here.\n\n---------- Forwarded message ---------\nFrom: someone\nold"
    assert strip_email_quotes(body) == "New text here."


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("document", CORPUS.documents, ids=lambda doc: doc.key)
def test_every_corpus_document_chunks_cleanly(document: FixtureDocument) -> None:
    chunks = chunk_document(source_document(document))
    assert chunks, f"{document.key} produced no chunks"
    assert [chunk.seq for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.token_count <= Chunker().max_tokens
        assert chunk.content_hash == hashlib.sha256(chunk.text.encode()).hexdigest()
        assert chunk.meta["content_hash"] == chunk.content_hash
        for key in REQUIRED_META_KEYS:
            assert key in chunk.meta, f"{document.key}: meta missing {key}"
        assert chunk.meta["doc_type"] == document.doc_type
        assert chunk.meta["source"] == document.source
        assert chunk.meta["source_id"] == document.source_id
        assert chunk.meta["token_count"] == chunk.token_count


def test_chunking_is_deterministic() -> None:
    document = CORPUS.get("notion", "8a1e7c62-3f4d-4b90-a7e5-2c9d6f1b8e34")
    first = chunk_document(source_document(document))
    second = chunk_document(source_document(document))
    assert [(c.seq, c.text, c.content_hash) for c in first] == [
        (c.seq, c.text, c.content_hash) for c in second
    ]
    assert [c.meta for c in first] == [c.meta for c in second]


def test_identical_text_produces_identical_content_hash() -> None:
    left = SourceDocument(
        source="notion", source_id="a", doc_type="page", title="Same", content="Same body."
    )
    right = SourceDocument(
        source="notion", source_id="b", doc_type="page", title="Same", content="Same body."
    )
    assert chunk_document(left)[0].content_hash == chunk_document(right)[0].content_hash


def test_empty_content_yields_no_chunks() -> None:
    doc = SourceDocument(
        source="gmail",
        source_id="<empty@example>",
        doc_type="email",
        title="Empty",
        content="  \n\n ",
    )
    assert chunk_document(doc) == []


def test_oversized_single_paragraph_is_hard_split() -> None:
    doc = SourceDocument(
        source="notion",
        source_id="huge",
        doc_type="page",
        title="Huge",
        content=SENTENCE * 120,  # one paragraph, no blank lines, ~2000 tokens
    )
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 500 for chunk in chunks)
    assert len({chunk.text for chunk in chunks}) == len(chunks), "no duplicate chunks"


def test_overlap_is_configured_at_fifteen_percent() -> None:
    chunker = Chunker()
    assert chunker.overlap_tokens == 75
    with pytest.raises(ValueError):
        Chunker(max_tokens=10)
    with pytest.raises(ValueError):
        Chunker(overlap_ratio=0.75)


def test_entity_ids_hook_is_recorded_in_meta() -> None:
    doc = source_document(CORPUS.get("jira", "PHX-150"))
    chunks = chunk_document(doc, entity_ids=["8f2c1d4e-0000-4000-8000-000000000001"])
    expected = ["8f2c1d4e-0000-4000-8000-000000000001"]
    assert all(chunk.meta["entity_ids"] == expected for chunk in chunks)


def test_token_estimate_is_deterministic_and_reasonable() -> None:
    assert estimate_tokens("one two three") == 3
    assert estimate_tokens("Hello, world!") == 4
    assert estimate_tokens("") == 0
    assert estimate_tokens(SENTENCE) == estimate_tokens(SENTENCE)


def test_oversized_title_is_truncated_so_the_budget_stays_honest() -> None:
    doc = SourceDocument(
        source="notion",
        source_id="long-title",
        doc_type="page",
        title=" ".join(["Extremely long title word"] * 60),
        content="Body text that should survive.",
    )
    chunks = chunk_document(doc)
    breadcrumb, body = chunks[0].text.split("\n\n", 1)
    assert body == "Body text that should survive."
    assert estimate_tokens(breadcrumb) <= round(Chunker().max_tokens * 0.3)
    assert chunks[0].token_count <= Chunker().max_tokens
