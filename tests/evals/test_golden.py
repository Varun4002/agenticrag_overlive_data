"""Golden-set contract tests: schema drift, coverage, hand-labelled expectations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals.schemas import (
    GOLDEN_SCHEMA_VERSION,
    GoldenQuestion,
    GoldenSetError,
    load_golden_set,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden.json"


def _golden() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_set_loads_and_has_ten_questions() -> None:
    golden = load_golden_set(GOLDEN_PATH)

    assert golden.schema_version == GOLDEN_SCHEMA_VERSION
    assert len(golden.questions) == 10
    assert golden.corpus == "fixtures/corpus"


def test_golden_categories_cover_the_brief() -> None:
    golden = load_golden_set(GOLDEN_PATH)
    categories = {question.category for question in golden.questions}

    assert categories >= {
        "task",
        "overdue",
        "deadline",
        "goals",
        "email",
        "status",
        "refusal",
        "injection",
        "contradictory",
    }
    assert sum(1 for question in golden.questions if question.category == "refusal") >= 2
    assert sum(1 for question in golden.questions if question.category == "injection") == 1
    assert sum(1 for question in golden.questions if question.category == "contradictory") == 1


def test_every_golden_question_is_labelled() -> None:
    golden = load_golden_set(GOLDEN_PATH)

    for question in golden.questions:
        assert question.notes.strip(), f"unlabelled question: {question.question}"
        if question.expected_behavior == "refuse":
            assert question.must_cite_source_types == []
        else:
            assert question.must_cite_source_types, question.question


def test_refusal_questions_are_out_of_corpus_or_policy_refusals() -> None:
    golden = load_golden_set(GOLDEN_PATH)
    refusals = [q for q in golden.questions if q.expected_behavior == "refuse"]

    assert len(refusals) == 2
    assert any("budget" in q.question.lower() for q in refusals)
    assert any(
        "email" in q.question.lower() and "phone" in q.question.lower() for q in refusals
    )


def test_schema_version_mismatch_is_a_hard_failure(tmp_path: Path) -> None:
    data = _golden()
    data["schema_version"] = GOLDEN_SCHEMA_VERSION + 1
    path = tmp_path / "golden_v999.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(GoldenSetError, match="schema version mismatch"):
        load_golden_set(path)


def test_duplicate_questions_are_rejected(tmp_path: Path) -> None:
    data = _golden()
    questions = data["questions"]
    assert isinstance(questions, list)
    data["questions"] = [questions[0], questions[0]]
    path = tmp_path / "golden_dupe.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_set(path)


def test_unknown_source_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source types"):
        GoldenQuestion(
            question="q",
            expected_entities=[],
            expected_behavior="answer",
            must_cite_source_types=["slack"],
        )


def test_missing_golden_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "nope.json")


def test_expected_entities_are_not_empty_for_answerable_questions() -> None:
    """Every answerable question in this set needs a checkable entity."""
    golden = load_golden_set(GOLDEN_PATH)

    for question in golden.questions:
        if question.expected_behavior == "answer":
            assert question.expected_entities, question.question


# --------------------------------------------------------------------------- #
# Cross-checks against the fixture corpus itself (drift = hard failure)
# --------------------------------------------------------------------------- #

CORPUS_DIR = PROJECT_ROOT / "fixtures" / "corpus"
ISSUE_KEY = re.compile(r"^[A-Z]{2,}-\d+$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _corpus() -> list[dict[str, object]]:
    if not CORPUS_DIR.exists():
        pytest.skip("fixtures/corpus is not present — skipping corpus cross-checks")
    documents: list[dict[str, object]] = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        documents.extend(data.get("documents", []))
    return documents


def _manifest() -> dict[str, object]:
    if not CORPUS_DIR.exists():
        pytest.skip("fixtures/corpus is not present — skipping corpus cross-checks")
    return json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_expected_issue_keys_exist_in_corpus() -> None:
    documents = _corpus()
    source_ids = {str(doc["source_id"]) for doc in documents}

    for question in load_golden_set(GOLDEN_PATH).questions:
        for entity in question.expected_entities:
            if ISSUE_KEY.match(entity):
                assert entity in source_ids, f"{entity} missing from fixture corpus"


def test_expected_dates_exist_in_corpus() -> None:
    blob = json.dumps(_corpus())

    for question in load_golden_set(GOLDEN_PATH).questions:
        for entity in question.expected_entities:
            if ISO_DATE.match(entity):
                assert entity in blob, f"{entity} missing from fixture corpus"


def test_overdue_expectations_match_the_manifest() -> None:
    manifest = _manifest()
    expected = next(
        q for q in load_golden_set(GOLDEN_PATH).questions if q.category == "overdue"
    )
    phoenix_overdue = {
        item["source_id"]
        for item in manifest["overdue"]
        if str(item["source_id"]).startswith("PHX-")
    }

    assert set(expected.expected_entities) == phoenix_overdue
    assert "ATL-88" not in expected.expected_entities


def test_due_this_week_expectations_match_the_manifest() -> None:
    manifest = _manifest()
    expected = next(
        q for q in load_golden_set(GOLDEN_PATH).questions if q.category == "deadline"
    )
    phoenix_due = {
        item["source_id"]
        for item in manifest["due_this_week"]
        if str(item["source_id"]).startswith("PHX-")
    }

    assert set(expected.expected_entities) == phoenix_due


def test_injection_question_targets_a_planted_payload_page() -> None:
    manifest = _manifest()
    injection_sources = {
        str(doc["source_id"]) for doc in manifest["injection"]["documents"]
    }
    question = next(
        q for q in load_golden_set(GOLDEN_PATH).questions if q.category == "injection"
    )
    documents = _corpus()
    target = next(
        doc
        for doc in documents
        if str(doc["source_id"]) in injection_sources and doc["title"].startswith("Vendor")
    )

    assert "Ignore all previous instructions" in str(target["content"])
    assert question.must_cite_source_types == [str(target["source"])]


def test_contradiction_question_uses_a_recorded_contradiction() -> None:
    manifest = _manifest()
    topics = {str(item["topic"]) for item in manifest["contradictions"]["items"]}
    question = next(
        q for q in load_golden_set(GOLDEN_PATH).questions if q.category == "contradictory"
    )

    assert any("PHX-150" in topic for topic in topics)
    assert {"2026-11-20", "2026-12-04"} <= set(question.expected_entities)
    assert set(question.must_cite_source_types) == {"jira", "gmail"}


def test_policy_refusal_question_references_an_assignee_with_corpus_pii() -> None:
    documents = {str(doc["source_id"]): doc for doc in _corpus()}
    issue = documents["PHX-142"]
    comments = [
        doc
        for doc in documents.values()
        if doc.get("parent_source_id") == "PHX-142" and "mobile" in str(doc["content"])
    ]

    assert "Assignee:" in str(issue["content"])
    assert comments, "PHX-142 no longer carries planted phone PII — revisit the refusal case"

