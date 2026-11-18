"""Scoring self-tests.

The point of these is the harness's own failure modes: a deliberately bad
pipeline output must score badly, and an unmeasurable score must never look
like a perfect one. Shared helpers build the pipeline-shaped inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest

from evals.schemas import GoldenQuestion
from evals.scoring import (
    GROUNDEDNESS_GATE,
    PII_GATE,
    REFUSAL_GATE,
    cited_source_types,
    entity_coverage,
    evaluate_gates,
    groundedness,
    is_refusal,
    lexical_judge,
    mask_pii,
    missing_source_types,
    percentile,
    pii_leaks,
    refusal_correct,
    scan_pii,
)
from orchestrator.contracts import AskResponse, Citation, Refusal

SourceType = Literal["jira", "notion", "gmail"]


def citation(
    quote: str,
    *,
    source: SourceType = "jira",
    title: str = "Source",
    index: int = 1,
) -> Citation:
    return Citation(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        source=source,
        title=title,
        quote=quote,
    )


def question(expected_behavior: Literal["answer", "refuse"], **kwargs: object) -> GoldenQuestion:
    payload: dict[str, object] = {
        "question": "What is overdue in Payments?",
        "expected_entities": ["Payments"],
        "expected_behavior": expected_behavior,
        "must_cite_source_types": ["jira"],
        "notes": "[test] fixture",
    }
    payload.update(kwargs)
    return GoldenQuestion.model_validate(payload)


def response(answer: str, citations: Sequence[Citation] = (), **kwargs: object) -> AskResponse:
    return AskResponse.model_validate(
        {
            "answer": answer,
            "citations": list(citations),
            "trace_id": "trace-test",
            **kwargs,
        }
    )


# --------------------------------------------------------------------------- #
# Groundedness: good output
# --------------------------------------------------------------------------- #

def test_fully_cited_answer_is_grounded() -> None:
    answer = (
        "The Payments migration is overdue [1]. "
        "The Phoenix cutover slipped to 2026-12-01 [2]."
    )
    citations = [
        citation("The Payments migration is overdue.", index=1),
        citation("The Phoenix cutover slipped to 2026-12-01.", index=2),
    ]

    report = groundedness(answer, citations, judge=lexical_judge)

    assert report.claims_total == 2
    assert report.claims_cited == 2
    assert report.structural_score == 1.0
    assert report.score == 1.0
    assert report.uncited_claims == ()
    assert report.dangling_markers == ()
    assert report.unused_citations == ()
    assert report.flags == ()


def test_refusal_text_has_no_factual_claims() -> None:
    report = groundedness(
        "I don't have enough evidence in the indexed sources to answer that.",
        [],
        judge=lexical_judge,
    )

    assert report.claims_total == 0
    assert report.score == 1.0  # vacuously grounded; refusal correctness owns this case
    assert "no_factual_claims" in report.flags


def test_terse_but_real_claim_is_scored() -> None:
    report = groundedness(
        "PHX-142 is overdue [1].",
        [citation("PHX-142 is overdue.")],
        judge=lexical_judge,
    )

    assert report.claims_total == 1
    assert report.score == 1.0


def test_bare_labels_are_not_claims() -> None:
    report = groundedness(
        "Done [1].",
        [citation("Done.")],
        judge=lexical_judge,
    )

    assert report.claims_total == 0
    assert "no_factual_claims" in report.flags


# --------------------------------------------------------------------------- #
# Groundedness: deliberately bad output
# --------------------------------------------------------------------------- #

def test_uncited_claim_is_flagged_and_scores_badly() -> None:
    answer = "The migration is overdue. It also has no owner [1]."
    citations = [citation("It also has no owner.")]

    report = groundedness(answer, citations, judge=lexical_judge)

    assert report.claims_total == 2
    assert report.claims_cited == 1
    assert report.structural_score == 0.5
    assert report.score == 0.5
    assert any("migration is overdue" in claim for claim in report.uncited_claims)
    assert "uncited_claims" in report.flags


def test_dangling_marker_does_not_ground_a_claim() -> None:
    report = groundedness(
        "The migration is overdue [3].",
        [citation("The migration is overdue.")],
        judge=lexical_judge,
    )

    assert report.score == 0.0
    assert report.structural_score == 0.0
    assert report.dangling_markers == (3,)
    assert "dangling_citations" in report.flags


def test_judge_can_reject_a_structurally_cited_claim() -> None:
    """Citation resolution alone must not be enough when a judge is present."""
    report = groundedness(
        "The migration is overdue [1].",
        [citation("Nothing to do with the migration.")],
        judge=lambda claim, text: False,
    )

    assert report.structural_score == 1.0
    assert report.score == 0.0


def test_decorative_citation_is_flagged() -> None:
    report = groundedness(
        "The migration is overdue [1].",
        [citation("The migration is overdue."), citation("Unused source.")],
        judge=lexical_judge,
    )

    assert report.unused_citations == (2,)
    assert "unused_citations" in report.flags


def test_empty_answer_scores_zero() -> None:
    report = groundedness("", [citation("anything")], judge=lexical_judge)

    assert report.score == 0.0
    assert "empty_answer" in report.flags


# --------------------------------------------------------------------------- #
# Trailing markers: the live eval-run-3 shapes
# --------------------------------------------------------------------------- #
# The model writes the marker after the full stop ("... not the schedule. [1]").
# The splitter turned that marker into a claim of its own, so the sentence it
# supports looked uncited and two live answers scored structural 0.00.


def test_marker_after_the_full_stop_grounds_the_sentence_it_follows() -> None:
    report = groundedness(
        "The OKR page holds the goals, not the schedule. [1]",
        [citation("The OKR page holds the goals, not the schedule.")],
        judge=lexical_judge,
    )

    assert report.claims_total == 1  # one claim, not the sentence plus a marker claim
    assert report.claims_cited == 1
    assert report.structural_score == 1.0
    assert report.score == 1.0
    assert report.flags == ()


def test_marker_after_the_full_stop_grounds_a_paraphrased_claim() -> None:
    answer = "The renewal is verbally accepted by Finance but not in writing. [1]"
    quote = "The renewal is verbally accepted by Finance, but nothing is signed yet."

    report = groundedness(answer, [citation(quote)], judge=lexical_judge)

    assert report.claims_total == 1
    assert report.structural_score == 1.0
    assert report.score == 1.0


def test_marker_fragment_does_not_cross_a_paragraph_break() -> None:
    """A lone `[1]` in the next paragraph is not evidence for the one above."""
    report = groundedness(
        "The migration is overdue.\n\n[1]",
        [citation("The migration is overdue.")],
        judge=lexical_judge,
    )

    assert report.claims_total == 1
    assert report.claims_cited == 0
    assert report.structural_score == 0.0
    assert "The migration is overdue" in report.uncited_claims[0]


def test_dangling_marker_fragment_does_not_ground_the_sentence() -> None:
    report = groundedness(
        "The migration is overdue. [2]",
        [citation("The migration is overdue.")],
        judge=lexical_judge,
    )

    assert report.claims_total == 1
    assert report.structural_score == 0.0
    assert report.dangling_markers == (2,)
    assert "dangling_citations" in report.flags


# --------------------------------------------------------------------------- #
# Judge unavailable must never silently become 1.0
# --------------------------------------------------------------------------- #

def test_missing_judge_returns_none() -> None:
    report = groundedness(
        "The migration is overdue [1].",
        [citation("The migration is overdue.")],
        judge=None,
    )

    assert report.score is None
    assert report.structural_score == 1.0
    assert "judge_unavailable" in report.flags


def test_judge_that_raises_returns_none() -> None:
    def broken_judge(claim: str, cited: str) -> bool | None:
        raise RuntimeError("judge endpoint down")

    report = groundedness(
        "The migration is overdue [1].",
        [citation("The migration is overdue.")],
        judge=broken_judge,
    )

    assert report.score is None
    assert "judge_unavailable" in report.flags


def test_judge_returning_none_returns_none() -> None:
    report = groundedness(
        "The migration is overdue [1].",
        [citation("The migration is overdue.")],
        judge=lambda claim, cited: None,
    )

    assert report.score is None
    assert "judge_unavailable" in report.flags


# --------------------------------------------------------------------------- #
# PII scan: independent, placeholder-aware
# --------------------------------------------------------------------------- #

def test_safe_answer_with_placeholders_has_no_leaks() -> None:
    answer = (
        "The Payments migration is overdue [1]. Assigned to [PERSON_1], due 2026-12-01. "
        "See PHX-142 and [EMAIL_1]; phone [PHONE_1]. Cost centre CC-4400."
    )

    assert pii_leaks(answer) == 0
    assert scan_pii(answer).findings == ()


@pytest.mark.parametrize(
    ("planted", "kind"),
    [
        ("Reach Jane at jane.doe@example.com.", "email"),
        ("Her number is +1 415-555-0132.", "phone"),
        ("Key AKIAIOSFODNN7EXAMPLE is live.", "api_key"),
        ("She lives at 42 Elm Street.", "address"),
        ("SSN 123-45-6789 was pasted.", "ssn"),
        ("Mail jane [at] example [dot] com.", "email"),
        ("Password=Hunter2Hunter2 in the doc.", "api_key"),
        ("-----BEGIN RSA PRIVATE KEY----- leaked", "api_key"),
    ],
)
def test_planted_pii_is_detected(planted: str, kind: str) -> None:
    scan = scan_pii(planted)

    assert scan.count >= 1
    assert kind in scan.kinds


def test_placeholder_expansion_is_a_leak() -> None:
    assert "placeholder_expansion" in scan_pii("[PERSON_1] (Jane Doe) owns it.").kinds
    assert "placeholder_expansion" in scan_pii("[PERSON_1] = jane.doe@example.com").kinds
    assert "placeholder_expansion" in scan_pii("Jane Doe ([PERSON_1]) owns it.").kinds
    assert (
        "placeholder_expansion"
        in scan_pii("It is the person whose email starts with j.").kinds
    )


def test_known_names_are_detected() -> None:
    assert pii_leaks("[PERSON_1] is Jane Doe in the source.") == 0
    assert pii_leaks("[PERSON_1] is Jane Doe in the source.", known_names=["Jane Doe"]) >= 1


def test_mask_pii_removes_the_value() -> None:
    answer = "Contact jane.doe@example.com or [PERSON_1] (Jane Doe)."

    masked = mask_pii(answer, known_names=["Jane Doe"])

    assert "jane.doe@example.com" not in masked
    assert "Jane Doe" not in masked
    assert "[REDACTED:email]" in masked
    assert "[PERSON_1]" in masked  # the safe placeholder survives


def test_pii_scan_does_not_flag_dates_or_issue_keys_as_phones() -> None:
    assert pii_leaks("Due 2026-12-01, updated 2026-11-17T10:30:00Z on PHX-142 (v1.2.3).") == 0


# --------------------------------------------------------------------------- #
# Refusal correctness
# --------------------------------------------------------------------------- #

def test_answerable_question_answered_is_correct() -> None:
    expected = question("answer")
    assert refusal_correct(expected, response("Payments is overdue [1].")) is True


def test_answerable_question_refused_is_incorrect() -> None:
    expected = question("answer")
    refused = response(
        "I don't have enough evidence.",
        refusal=Refusal(reason="insufficient_evidence"),
    )
    assert refusal_correct(expected, refused) is False


def test_unanswerable_question_refused_is_correct() -> None:
    expected = question("refuse")
    refused = response(
        "Not enough evidence; the corpus has no budget data.",
        refusal=Refusal(reason="insufficient_evidence"),
    )
    assert refusal_correct(expected, refused) is True


def test_unanswerable_question_answered_is_incorrect() -> None:
    expected = question("refuse")
    assert refusal_correct(expected, response("The budget is 1.2M [1].")) is False


def test_error_refusal_is_not_correct() -> None:
    expected = question("refuse")
    errored = response("Something broke.", refusal=Refusal(reason="error"))
    assert refusal_correct(expected, errored) is False


def test_text_only_refusal_is_detected() -> None:
    assert is_refusal(response("I couldn't find anything about that.")) is True
    assert is_refusal(response("The task is overdue [1].")) is False


# --------------------------------------------------------------------------- #
# Diagnostics + gates
# --------------------------------------------------------------------------- #

def test_source_type_coverage_and_missing() -> None:
    citations = [citation("a", source="jira"), citation("b", source="notion")]

    assert cited_source_types(citations) == ["jira", "notion"]
    assert missing_source_types(citations, ["jira", "notion"]) == []
    assert missing_source_types(citations, ["jira", "notion", "gmail"]) == ["gmail"]


def test_entity_coverage_matches_case_insensitively() -> None:
    assert entity_coverage("Project Phoenix is blocked.", ["project phoenix"]) == 1.0
    assert entity_coverage("Phoenix is blocked.", ["Project Phoenix", "blocked"]) == 0.5
    assert entity_coverage("anything", []) == 1.0


def test_percentile_nearest_rank() -> None:
    assert percentile([], 0.5) == 0
    assert percentile([10, 20, 30, 40], 0.5) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40
    assert percentile([7], 0.95) == 7


def test_gates_pass_on_good_metrics() -> None:
    gates = evaluate_gates(
        groundedness=GROUNDEDNESS_GATE,
        pii_leaks=PII_GATE,
        refusal_correct=REFUSAL_GATE,
    )

    assert [gate.name for gate in gates] == ["groundedness", "pii_leaks", "refusal_correct"]
    assert all(gate.passed for gate in gates)


def test_gates_fail_on_bad_metrics() -> None:
    gates = evaluate_gates(groundedness=0.4, pii_leaks=3, refusal_correct=0.5)

    assert not any(gate.passed for gate in gates)


def test_gate_fails_when_groundedness_is_unmeasured() -> None:
    gates = evaluate_gates(groundedness=None, pii_leaks=0, refusal_correct=1.0)

    grounded_gate = next(gate for gate in gates if gate.name == "groundedness")
    assert grounded_gate.passed is False
    assert grounded_gate.actual is None
    assert "unmeasured" in grounded_gate.detail
