"""Fixture-corpus tests: determinism, schema, and the coverage other agents rely on.

The corpus is shared substrate, so these tests pin the properties downstream
agents assume: stable content hashes, both projects present, deliberate PII,
planted prompt injections, contradictions, and the "Alice Migration" false
positive.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from evals.fixtures import (
    FIXTURE_DIR,
    FixtureCorpusError,
    document_id_for,
    fixture_document_ids,
    iter_documents,
    load_fixture_corpus,
)
from indexing.tokens import sha256_text

CORPUS = load_fixture_corpus()

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+1-\d{3}-555-\d{4}")
API_KEY_RE = re.compile(r"sk-[a-z]+-[0-9a-f]{24,}")
ADDRESS = "1800 Harbor Point Drive, Suite 400, Emeryville, CA 94608"
TODAY = date(2026, 11, 17)


def all_text() -> str:
    return "\n".join(doc.content for doc in CORPUS.documents)


# ---------------------------------------------------------------------------
# Determinism and schema
# ---------------------------------------------------------------------------


def test_corpus_is_deterministic_across_loads() -> None:
    first = load_fixture_corpus()
    second = load_fixture_corpus()
    assert [doc.key for doc in first.documents] == [doc.key for doc in second.documents]
    assert [doc.content_hash for doc in first.documents] == [
        doc.content_hash for doc in second.documents
    ]
    assert first.manifest == second.manifest


def test_documents_are_sorted_and_unique() -> None:
    keys = [doc.key for doc in CORPUS.documents]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_corpus_size_and_source_coverage() -> None:
    assert 40 <= len(CORPUS.documents) <= 60
    for source in ("jira", "notion", "gmail"):
        assert CORPUS.by_source(source), f"no {source} documents"

    counts = CORPUS.counts
    assert counts["jira:issue"] >= 10
    assert counts["jira:issue_comment"] >= 8
    assert counts["notion:page"] >= 5
    assert counts["notion:page_section"] >= 1
    assert counts["gmail:email"] >= 5


def test_content_hash_is_the_content_hash() -> None:
    for doc in CORPUS.documents:
        assert doc.content_hash == sha256_text(doc.content)
        assert len(doc.content_hash) == 64


def test_timestamps_are_timezone_aware_and_ordered() -> None:
    for doc in CORPUS.documents:
        assert doc.created_at.tzinfo is not None
        assert doc.updated_at.tzinfo is not None
        assert doc.updated_at >= doc.created_at


def test_document_ids_are_deterministic() -> None:
    ids = fixture_document_ids(CORPUS)
    assert ids == fixture_document_ids(CORPUS)
    assert len(set(ids)) == len(ids)
    first = CORPUS.documents[0]
    assert document_id_for(first) == document_id_for(first)


def test_iter_documents_matches_corpus() -> None:
    assert list(iter_documents(CORPUS)) == list(CORPUS.documents)


# ---------------------------------------------------------------------------
# Content the downstream agents depend on
# ---------------------------------------------------------------------------


def test_both_projects_are_present_and_distinct() -> None:
    jira = CORPUS.by_source("jira")
    phoenix = [doc for doc in jira if doc.source_id.startswith("PHX-")]
    atlas = [doc for doc in jira if doc.source_id.startswith("ATL-")]
    assert len(phoenix) >= 10
    assert len(atlas) >= 4

    phoenix_text = "\n".join(doc.content for doc in phoenix)
    atlas_text = "\n".join(doc.content for doc in atlas)
    assert "Project Phoenix" in phoenix_text
    assert "Ledger v2" in phoenix_text
    assert "ETL" in atlas_text
    # Atlas must be clearly separate work, or cross-project tests prove nothing.
    assert "Ledger v2" not in atlas_text


def test_manifest_projects_match_the_corpus() -> None:
    projects = {project["key"]: project for project in CORPUS.manifest["projects"]}
    assert set(projects) == {"PHX", "ATL"}
    assert projects["PHX"]["name"] == "Project Phoenix"
    assert "PHX" in projects["PHX"]["aliases"]


def test_deliberate_pii_is_present() -> None:
    text = all_text()
    assert EMAIL_RE.findall(text), "no email addresses in the corpus"
    assert PHONE_RE.findall(text), "no phone numbers in the corpus"
    assert API_KEY_RE.findall(text), "no API-key-looking token in the corpus"
    assert ADDRESS in text
    assert "Priya Raghunathan" in text

    kinds = CORPUS.manifest["pii"]["kinds"]
    for value in kinds["email"]:
        assert value in text, f"manifest email {value} missing from corpus"
    for value in kinds["phone"]:
        assert value in text
    for value in kinds["api_key"]:
        assert value in text
    for value in kinds["address"]:
        assert value in text


def test_pii_index_points_at_documents_that_actually_contain_pii() -> None:
    planted = CORPUS.manifest["pii"]["planted_in"]
    assert len(planted) >= 6
    for entry in planted:
        doc = CORPUS.get(entry["source"], entry["source_id"])
        assert doc.content.strip()
        # Each indexed document must carry at least one of the declared kinds.
        assert entry["kinds"], doc.key
    pii_docs = {doc.key for doc in CORPUS.pii_documents()}
    assert any(doc.source == "notion" for doc in CORPUS.pii_documents())
    assert any(doc.source == "gmail" for doc in CORPUS.pii_documents())
    assert any(doc.source == "jira" for doc in CORPUS.pii_documents())
    assert len(pii_docs) == len(planted)


def test_planted_prompt_injections_exist_in_two_sources() -> None:
    injected = CORPUS.injection_documents()
    markers = {entry["marker"] for entry in CORPUS.manifest["injection"]["documents"]}
    assert markers == {"Ignore all previous instructions"}
    for doc in injected:
        assert "Ignore all previous instructions" in doc.content
    assert {doc.source for doc in injected} == {"jira", "notion"}
    assert any(doc.doc_type == "issue_comment" for doc in injected)
    assert any(doc.doc_type == "page" for doc in injected)


def test_contradictions_are_real_conflicts_in_the_cited_documents() -> None:
    items = CORPUS.manifest["contradictions"]["items"]
    assert len(items) >= 3

    cutover = next(item for item in items if item["topic"] == "PHX-150 cutover date")
    assert len(set(cutover["values"])) == 3
    for source in cutover["sources"]:
        doc = CORPUS.get(source["source"], source["source_id"])
        assert source["value"] in doc.content, f"{source['value']} not in {doc.key}"

    for item in items:
        values = set(item["values"])
        assert len(values) >= 2, item["topic"]
        for source in item["sources"]:
            doc = CORPUS.get(source["source"], source["source_id"])
            assert source["value"] in doc.content, f"{item['topic']}: {doc.key}"


def test_person_like_project_name_is_declared_a_codename() -> None:
    false_positives = CORPUS.manifest["false_positive_names"]["items"]
    assert false_positives[0]["value"] == "Alice Migration"

    epic = CORPUS.get("jira", "PHX-131")
    page = CORPUS.get("notion", "d1a9e6b3-2c58-4f70-9b4d-7e8a3c6f0d21")
    assert "Alice Migration" in epic.content
    assert "internal codename" in epic.content
    assert "not named after a person" in epic.content.lower()
    assert "not a person" in page.content


def test_overdue_and_due_this_week_dates_are_consistent() -> None:
    overdue = CORPUS.manifest["overdue"]
    assert len(overdue) >= 3
    for entry in overdue:
        due = date.fromisoformat(entry["due"])
        assert due < TODAY, f"{entry['source_id']} is listed overdue but is not"
        assert (TODAY - due).days == entry["days_overdue"]

        doc = CORPUS.get(entry["source"], entry["source_id"])
        assert f"Due date: {entry['due']}" in doc.content

    for entry in CORPUS.manifest["due_this_week"]:
        due = date.fromisoformat(entry["due"])
        assert TODAY <= due <= date(2026, 11, 23), entry
        doc = CORPUS.get(entry["source"], entry["source_id"])
        assert f"Due date: {entry['due']}" in doc.content


def test_every_due_date_field_is_parseable_from_the_issue_card() -> None:
    for doc in CORPUS.by_doc_type("issue"):
        due_lines = [line for line in doc.content.splitlines() if line.startswith("Due date: ")]
        if not due_lines:
            continue
        value = due_lines[0].removeprefix("Due date: ").strip()
        datetime.strptime(value, "%Y-%m-%d").date()


def test_status_report_and_okr_page_are_discoverable() -> None:
    status = CORPUS.get("notion", "8a1e7c62-3f4d-4b90-a7e5-2c9d6f1b8e34")
    assert "Phoenix Weekly Status" in status.title
    assert "## Risks and changes" in status.content
    assert "2027-01-15" in status.content

    goals = CORPUS.get("notion", "5d9f2c41-7b3e-4a18-9c62-0b1d8e4f7a10")
    assert "2026-12-15" in goals.content
    assert "## Q4 objectives" in goals.content


def test_email_threads_keep_quoted_history_for_the_chunker() -> None:
    cutover = [doc for doc in CORPUS.by_source("gmail") if "phoenix-cutover" in doc.source_id]
    assert len(cutover) == 5
    assert {doc.parent_source_id for doc in cutover} == {
        "<CAF8x2b1.phoenix-cutover@mail.northwind.example>"
    }
    assert any("wrote:" in doc.content for doc in cutover)
    assert any("-----Original Message-----" in doc.content for doc in cutover)


# ---------------------------------------------------------------------------
# Loader failure modes
# ---------------------------------------------------------------------------


def minimal_document(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": "jira",
        "source_id": "PHX-1",
        "parent_source_id": None,
        "doc_type": "issue",
        "title": "PHX-1 — Minimal",
        "content": "Key: PHX-1\nSummary: Minimal\nStatus: To Do\n\nBody.",
        "url": None,
        "author": "Tester",
        "created_at": "2026-11-01T00:00:00+00:00",
        "updated_at": "2026-11-02T00:00:00+00:00",
        "acl": {"labels": ["phoenix"]},
        "labels": [],
    }
    payload.update(overrides)
    return payload


def write_corpus(
    root: Path, documents: list[dict[str, Any]], manifest: dict[str, Any] | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "jira.json").write_text(json.dumps({"documents": documents}), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(manifest if manifest is not None else {"projects": []}), encoding="utf-8"
    )
    return root


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FixtureCorpusError, match="not found"):
        load_fixture_corpus(tmp_path / "does-not-exist")


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    (tmp_path / "jira.json").write_text(json.dumps({"documents": []}), encoding="utf-8")
    with pytest.raises(FixtureCorpusError, match="manifest not found"):
        load_fixture_corpus(tmp_path)


def test_duplicate_documents_are_rejected(tmp_path: Path) -> None:
    write_corpus(tmp_path, [minimal_document(), minimal_document()])
    with pytest.raises(FixtureCorpusError, match="duplicate document"):
        load_fixture_corpus(tmp_path)


def test_invalid_doc_type_is_rejected(tmp_path: Path) -> None:
    write_corpus(tmp_path, [minimal_document(source="jira", doc_type="page")])
    with pytest.raises(FixtureCorpusError, match="not valid for source"):
        load_fixture_corpus(tmp_path)


def test_comment_without_parent_is_rejected(tmp_path: Path) -> None:
    write_corpus(tmp_path, [minimal_document(doc_type="issue_comment")])
    with pytest.raises(FixtureCorpusError, match="requires parent_source_id"):
        load_fixture_corpus(tmp_path)


def test_naive_timestamps_are_rejected(tmp_path: Path) -> None:
    write_corpus(
        tmp_path,
        [minimal_document(created_at="2026-11-01T00:00:00", updated_at="2026-11-01T00:00:00")],
    )
    with pytest.raises(FixtureCorpusError, match="timezone-aware"):
        load_fixture_corpus(tmp_path)


def test_stale_manifest_reference_is_rejected(tmp_path: Path) -> None:
    manifest = {
        "pii": {
            "planted_in": [
                {"source": "jira", "source_id": "PHX-404", "kinds": ["email"]},
            ]
        }
    }
    write_corpus(tmp_path, [minimal_document()], manifest)
    with pytest.raises(FixtureCorpusError, match="missing documents"):
        load_fixture_corpus(tmp_path)


def test_custom_path_loads_and_hashes(tmp_path: Path) -> None:
    write_corpus(tmp_path, [minimal_document()])
    corpus = load_fixture_corpus(tmp_path)
    assert len(corpus.documents) == 1
    assert corpus.documents[0].content_hash == sha256_text(corpus.documents[0].content)
    assert corpus.path == tmp_path


def test_fixture_dir_is_the_documented_location() -> None:
    assert FIXTURE_DIR.name == "corpus"
    assert FIXTURE_DIR.parent.name == "fixtures"
    assert (FIXTURE_DIR / "manifest.json").is_file()
