"""Harness self-tests: a bad pipeline must fail the gate, a good one must pass.

The fake pipeline is an in-process ASGI app, so the suite exercises the real
HTTP path (``POST /ask``, ``AskRequest`` validation, ``AskResponse`` parsing)
without a running server.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from evals.run import (
    default_roster_path,
    exit_code_for,
    load_pii_roster,
    main,
    render_markdown,
    run_suite,
)
from evals.schemas import EvalResult, GoldenQuestion, load_golden_set
from evals.scoring import JudgeFn, lexical_judge
from orchestrator.contracts import AskRequest, AskResponse, Citation, Refusal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden.json"

SourceType = Literal["jira", "notion", "gmail"]


def _golden_questions() -> list[GoldenQuestion]:
    return list(load_golden_set(GOLDEN_PATH).questions)


def _trace_id(prefix: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"trace-{prefix}-{digest}"


def _source_for(question: GoldenQuestion, index: int) -> SourceType:
    required = question.must_cite_source_types
    if not required:
        return "jira"
    value = required[(index - 1) % len(required)]
    assert value in ("jira", "notion", "gmail")
    return cast(SourceType, value)


def _good_response(question: GoldenQuestion) -> AskResponse:
    """A correct, cited answer for any golden question."""
    if question.expected_behavior == "refuse":
        return AskResponse(
            answer="I don't have enough evidence in the indexed sources to answer that.",
            refusal=Refusal(reason="insufficient_evidence", detail="out of corpus"),
            trace_id=_trace_id("good", question.question),
        )

    claims: list[str] = []
    citations: list[Citation] = []
    entities = question.expected_entities or [question.question]
    for index, entity in enumerate(entities, start=1):
        statement = f"{entity} is documented in the indexed work sources"
        claims.append(f"{statement} [{index}].")
        citations.append(
            Citation(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                source=_source_for(question, index),
                title=entity,
                quote=f"{statement}.",
            )
        )
    return AskResponse(
        answer=" ".join(claims),
        citations=citations,
        trace_id=_trace_id("good", question.question),
    )


BAD_ANSWER = (
    "The Payments migration is overdue and assigned to John Smith at "
    "john.smith@example.com. Call +1 415-555-0132 for the latest status."
)


def _bad_response(question: GoldenQuestion) -> AskResponse:
    """Deliberately bad: uncited claims, planted PII, and no refusals."""
    return AskResponse(
        answer=BAD_ANSWER,
        citations=[],
        trace_id=_trace_id("bad", question.question),
    )


def _app(responses: Mapping[str, AskResponse]) -> FastAPI:
    app = FastAPI()

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        if req.question not in responses:
            raise HTTPException(status_code=500, detail="unknown question")
        return responses[req.question]

    return app


def _broken_app() -> FastAPI:
    app = FastAPI()

    @app.post("/ask")
    async def ask(req: AskRequest) -> AskResponse:
        raise HTTPException(status_code=503, detail="pipeline unavailable")

    return app


async def _run(
    app: FastAPI,
    questions: Sequence[GoldenQuestion],
    *,
    judge: JudgeFn | None = lexical_judge,
    known_names: Sequence[str] = (),
) -> EvalResult:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://eval.test") as client:
        return await run_suite(
            questions,
            base_url="http://eval.test",
            client=client,
            judge=judge,
            judge_name="lexical" if judge is not None else "none",
            known_names=known_names,
        )


async def test_good_pipeline_passes_every_gate() -> None:
    questions = _golden_questions()
    app = _app({q.question: _good_response(q) for q in questions})

    result = await _run(app, questions)

    assert result.gate_passed is True
    assert result.groundedness == pytest.approx(1.0)
    assert result.pii_leaks == 0
    assert result.refusal_correct == pytest.approx(1.0)
    assert result.errors == 0
    assert all(gate.passed for gate in result.gates)
    assert exit_code_for(result) == 0


async def test_bad_pipeline_fails_every_gate() -> None:
    questions = _golden_questions()
    app = _app({q.question: _bad_response(q) for q in questions})

    result = await _run(app, questions)

    assert result.gate_passed is False
    assert result.pii_leaks > 0
    assert result.groundedness is not None
    assert result.groundedness < 0.9
    assert result.refusal_correct < 0.9
    assert not any(gate.passed for gate in result.gates)
    assert exit_code_for(result) == 1
    assert {gate.name for gate in result.gates if not gate.passed} == {
        "groundedness",
        "pii_leaks",
        "refusal_correct",
    }


async def test_unjudged_run_cannot_pass_the_groundedness_gate() -> None:
    questions = _golden_questions()
    app = _app({q.question: _good_response(q) for q in questions})

    result = await _run(app, questions, judge=None)

    assert result.groundedness is None
    assert result.gate_passed is False
    assert "judge_unavailable" in result.flags
    grounded_gate = next(gate for gate in result.gates if gate.name == "groundedness")
    assert grounded_gate.passed is False
    assert grounded_gate.actual is None
    assert exit_code_for(result) == 1


async def test_unreachable_pipeline_is_a_harness_failure() -> None:
    questions = _golden_questions()

    result = await _run(_broken_app(), questions)

    assert result.errors == len(questions)
    assert exit_code_for(result) == 2
    assert "question_errors" in result.flags


async def test_markdown_report_has_gates_and_per_question_rows() -> None:
    questions = _golden_questions()
    app = _app({q.question: _good_response(q) for q in questions})

    result = await _run(app, questions)
    markdown = render_markdown(result, base_url="http://eval.test", corpus="fixtures")

    assert markdown.startswith("# Eval report")
    assert "**Gate: PASS**" in markdown
    assert "## Gates" in markdown
    assert "## Per question" in markdown
    assert "groundedness" in markdown
    assert markdown.count("\n|") >= len(questions)


async def test_bad_report_masks_planted_pii() -> None:
    questions = _golden_questions()
    app = _app({q.question: _bad_response(q) for q in questions})

    result = await _run(app, questions)
    markdown = render_markdown(result, base_url="http://eval.test")

    assert "**Gate: FAIL**" in markdown
    assert "john.smith@example.com" not in markdown
    assert "415-555-0132" not in markdown
    assert "[REDACTED:email]" in markdown
    assert "## Diagnostics" in markdown


async def test_markdown_warns_when_no_judge_is_configured() -> None:
    questions = _golden_questions()
    app = _app({q.question: _good_response(q) for q in questions})

    result = await _run(app, questions, judge=None)
    markdown = render_markdown(result, base_url="http://eval.test")

    assert "**Gate: FAIL**" in markdown
    assert "Groundedness is unmeasured" in markdown


async def test_request_errors_are_reported_per_question() -> None:
    """A pipeline that 500s on one question still reports the rest."""
    questions = _golden_questions()
    responses = {q.question: _good_response(q) for q in questions}
    missing = questions[0].question
    app = _app({key: value for key, value in responses.items() if key != missing})

    result = await _run(app, questions)

    assert result.errors == 1
    failed = next(report for report in result.per_question if report.actual_behavior == "error")
    assert "request_failed" in failed.flags
    assert failed.trace_id is None


def test_main_rejects_a_stale_golden_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    code = main(["--golden", str(path)])

    assert code == 2
    assert "schema version mismatch" in capsys.readouterr().err


def test_pii_roster_is_loaded_from_the_corpus_manifest() -> None:
    manifest = default_roster_path("fixtures")
    assert manifest is not None
    roster = load_pii_roster(manifest)

    assert "Priya Raghunathan" in roster
    assert len(roster) >= 5


async def test_fixture_roster_names_are_flagged_as_leaks() -> None:
    questions = _golden_questions()[:1]
    leaked = AskResponse(
        answer="Priya Raghunathan owns the escalation for this work [1].",
        citations=[
            Citation(
                chunk_id="chunk-1",
                document_id="doc-1",
                source="jira",
                title="PHX-142",
                quote="Priya Raghunathan owns the escalation.",
            )
        ],
        trace_id="trace-roster",
    )
    app = _app({questions[0].question: leaked})
    roster = load_pii_roster(default_roster_path("fixtures"))  # type: ignore[arg-type]

    without_roster = await _run(app, questions)
    with_roster = await _run(app, questions, known_names=roster)

    assert without_roster.pii_leaks == 0  # regex heuristics cannot know the name
    assert with_roster.pii_leaks >= 1
    assert with_roster.gate_passed is False
