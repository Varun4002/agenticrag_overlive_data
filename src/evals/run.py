"""Golden-question eval harness.

Runs a suite over the live HTTP ``/ask`` endpoint (async ``httpx``), so it can
score the pipeline before in-process wiring exists, then prints a markdown
report and exits non-zero when a success-criteria gate fails.

Usage::

    python -m evals.run --suite golden --base-url http://127.0.0.1:8000 \
        --judge lexical --user-id priya.raghunathan@northwind.example

``--user-id`` is not cosmetic: retrieval enforces each document's ``acl.users``
in SQL, and the default ``eval-user`` is in no corpus document — every question
would be refused with an empty evidence pack. The report flags that outcome as
``acl_identity_sees_nothing`` instead of presenting it as a system failure.

Exit codes: ``0`` all gates pass, ``1`` gate failure, ``2`` harness failure
(bad golden file, no responses at all). Success criteria: groundedness ≥ 0.90,
pii_leaks == 0, refusal correctness ≥ 0.90 (``docs/01-vision-scope.md``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
from pydantic import ValidationError

from evals.schemas import (
    EvalResult,
    GoldenQuestion,
    GoldenSetError,
    QuestionReport,
    load_golden_set,
)
from evals.scoring import (
    JudgeFn,
    cited_source_types,
    entity_coverage,
    evaluate_gates,
    groundedness,
    is_refusal,
    lexical_judge,
    mask_pii,
    missing_source_types,
    percentile,
    refusal_correct,
    scan_pii,
)
from orchestrator.contracts import AskRequest, AskResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Local 8B models take 30-60 s per question over a 12-chunk pack, and the
# orchestrator gives itself 120 s (`Settings.orchestrator_budget_s`). The client
# timeout must exceed that budget, or the harness aborts requests the pipeline
# would have answered.
DEFAULT_TIMEOUT_S = 180.0
EXCERPT_LIMIT = 320


def load_pii_roster(path: Path) -> list[str]:
    """Fixture-roster person names from a corpus manifest, for the leak scan.

    Regex heuristics cannot know fixture names; the manifest lists them under
    ``pii.kinds.person``. A missing/unreadable manifest yields an empty roster
    rather than failing the suite.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    kinds = data.get("pii", {}).get("kinds", {}) if isinstance(data, dict) else {}
    names = kinds.get("person", []) if isinstance(kinds, dict) else []
    return [name for name in names if isinstance(name, str)]


def default_roster_path(corpus: str) -> Path | None:
    """Locate a corpus manifest next to the fixture corpus, if one exists."""
    base = Path(corpus)
    for candidate in (base / "corpus" / "manifest.json", base / "manifest.json"):
        if candidate.exists():
            return candidate
    return None


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "\u2026"


def score_response(
    expected: GoldenQuestion,
    response: AskResponse,
    *,
    latency_ms: int,
    judge: JudgeFn | None = None,
    known_names: Sequence[str] = (),
) -> QuestionReport:
    """Score one pipeline response against its golden question."""
    grounded = groundedness(response.answer, response.citations, judge=judge)
    scan = scan_pii(response.answer, known_names=known_names)
    refused = is_refusal(response)
    missing = missing_source_types(response.citations, expected.must_cite_source_types)

    flags = list(dict.fromkeys([*grounded.flags, *scan.flags]))
    if missing and expected.expected_behavior == "answer" and not refused:
        flags.append("missing_required_sources")
    if response.degraded:
        flags.append("degraded_response")
    if not response.answer.strip() and response.refusal is None:
        flags.append("empty_answer")
    if refused and expected.expected_behavior == "answer":
        flags.append("refused_answerable_question")
    if not refused and expected.expected_behavior == "refuse":
        flags.append("answered_unanswerable_question")

    masked_answer = mask_pii(response.answer, known_names=known_names)
    return QuestionReport(
        question=expected.question,
        category=expected.category,
        expected_behavior=expected.expected_behavior,
        actual_behavior="refused" if refused else "answered",
        refusal_correct=refusal_correct(expected, response),
        groundedness=grounded.score,
        structural_groundedness=grounded.structural_score,
        pii_leaks=scan.count,
        pii_findings=[f"{finding.kind}@{finding.span[0]}" for finding in scan.findings],
        citation_count=len(response.citations),
        cited_source_types=cited_source_types(response.citations),
        missing_source_types=missing,
        entity_coverage=entity_coverage(response.answer, expected.expected_entities),
        latency_ms=latency_ms,
        trace_id=response.trace_id,
        answer_excerpt=_excerpt(masked_answer),
        uncited_claims=[
            _excerpt(mask_pii(claim, known_names=known_names), 160)
            for claim in grounded.uncited_claims
        ],
        flags=flags,
    )


def _error_report(
    expected: GoldenQuestion, *, latency_ms: int, exc: BaseException
) -> QuestionReport:
    return QuestionReport(
        question=expected.question,
        category=expected.category,
        expected_behavior=expected.expected_behavior,
        actual_behavior="error",
        refusal_correct=False,
        latency_ms=latency_ms,
        error=f"{type(exc).__name__}: {exc}",
        flags=["request_failed"],
    )


async def _ask_and_score(
    client: httpx.AsyncClient,
    base_url: str,
    expected: GoldenQuestion,
    *,
    session_id: str,
    user_id: str,
    timeout_s: float,
    judge: JudgeFn | None,
    known_names: Sequence[str],
) -> QuestionReport:
    url = base_url.rstrip("/") + "/ask"
    payload = AskRequest(
        question=expected.question, session_id=session_id, user_id=user_id
    ).model_dump()
    started = time.perf_counter()
    try:
        http_response = await client.post(url, json=payload, timeout=timeout_s)
        http_response.raise_for_status()
        parsed = AskResponse.model_validate(http_response.json())
    except (httpx.HTTPError, ValidationError, ValueError) as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _error_report(expected, latency_ms=latency_ms, exc=exc)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return score_response(
        expected, parsed, latency_ms=latency_ms, judge=judge, known_names=known_names
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


async def run_suite(
    questions: Sequence[GoldenQuestion],
    *,
    base_url: str = DEFAULT_BASE_URL,
    client: httpx.AsyncClient | None = None,
    judge: JudgeFn | None = None,
    judge_name: str = "none",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    session_id: str = "eval-suite",
    user_id: str = "eval-user",
    suite: str = "golden",
    known_names: Sequence[str] = (),
    progress: bool = False,
) -> EvalResult:
    """Run every question sequentially and roll the scores up.

    Sequential on purpose: deterministic ordering, no self-inflicted load, and
    no concurrency-induced rate limits. Pass ``client`` to score an in-process
    or fake transport (tests use an ASGI transport), and ``progress=True`` to
    print one status line per question on stderr — a live-model suite takes
    minutes, and silence looks like a hang.
    """
    owns_client = client is None
    active = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        reports: list[QuestionReport] = []
        total = len(questions)
        for index, question in enumerate(questions, start=1):
            report = await _ask_and_score(
                active,
                base_url,
                question,
                session_id=session_id,
                user_id=user_id,
                timeout_s=timeout_s,
                judge=judge,
                known_names=known_names,
            )
            reports.append(report)
            if progress:
                print(
                    f"[{index}/{total}] {report.actual_behavior}"
                    f" grounded={_format_score(report.groundedness)}"
                    f" citations={report.citation_count}"
                    f" leaks={report.pii_leaks}"
                    f" {report.latency_ms / 1000:.1f}s"
                    f" :: {report.question[:70]}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if owns_client:
            await active.aclose()

    answerable = [
        report
        for report in reports
        if report.expected_behavior == "answer" and report.actual_behavior != "error"
    ]
    grounded_values: list[float] = []
    for report in answerable:
        if report.groundedness is not None:
            grounded_values.append(report.groundedness)
    groundedness_agg = _mean(grounded_values)
    structural = _mean([report.structural_groundedness for report in answerable])
    pii_total = sum(report.pii_leaks for report in reports)
    refusal_total = _mean([1.0 if report.refusal_correct else 0.0 for report in reports])
    coverage_val = _mean([report.entity_coverage for report in answerable])
    errors = sum(1 for report in reports if report.actual_behavior == "error")
    latencies = [report.latency_ms for report in reports]

    flags: list[str] = []
    if groundedness_agg is None:
        flags.append("judge_unavailable")
        if not answerable:
            flags.append("no_answerable_questions")
    if errors:
        flags.append("question_errors")
    if any("degraded_response" in report.flags for report in reports):
        flags.append("degraded_responses")
    if any(report.missing_source_types for report in reports):
        flags.append("missing_required_sources")
    # "Everything refused with an empty retrieval" is almost never a pipeline
    # result: it means the ACL identity sees no document at all. That is a
    # harness-configuration problem, and it must not read like "the system
    # refused 10/10 questions".
    if reports and all(
        report.actual_behavior == "refused"
        and "retrieval returned no evidence chunks" in report.answer_excerpt
        for report in reports
    ):
        flags.append("acl_identity_sees_nothing")

    gates = evaluate_gates(
        groundedness=groundedness_agg,
        pii_leaks=pii_total,
        refusal_correct=refusal_total or 0.0,
    )
    return EvalResult(
        groundedness=groundedness_agg,
        pii_leaks=pii_total,
        refusal_correct=refusal_total or 0.0,
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        per_question=reports,
        suite=suite,
        judge=judge_name,
        pii_roster_names=len(known_names),
        question_count=len(reports),
        errors=errors,
        groundedness_structural=structural or 0.0,
        entity_coverage=coverage_val or 0.0,
        gates=gates,
        gate_passed=all(gate.passed for gate in gates),
        flags=flags,
        user_id=user_id,
    )


def exit_code_for(result: EvalResult) -> int:
    """Map a run to a CLI exit code (0 pass, 1 gate fail, 2 harness failure)."""
    if result.question_count == 0:
        return 2
    if result.errors == result.question_count:
        return 2
    return 0 if result.gate_passed else 1


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(
    result: EvalResult,
    *,
    base_url: str = "",
    corpus: str = "",
) -> str:
    """Render an EvalResult as a readable markdown report."""
    lines: list[str] = []
    verdict = "PASS" if result.gate_passed else "FAIL"
    lines.append(f"# Eval report — {result.suite}")
    lines.append("")
    lines.append(f"- **Gate: {verdict}**")
    lines.append(f"- generated: `{result.generated_at.isoformat(timespec='seconds')}`")
    lines.append(f"- endpoint: `{base_url or 'n/a'}`")
    lines.append(f"- questions: {result.question_count} ({result.errors} request errors)")
    lines.append(f"- judge: `{result.judge}`")
    if result.pii_roster_names:
        lines.append(f"- pii roster: {result.pii_roster_names} fixture names")
    if corpus:
        present = "present" if Path(corpus).exists() else "absent"
        lines.append(f"- corpus path: `{corpus}` ({present})")
    lines.append(f"- asked as user: `{result.user_id}` (retrieval enforces this identity's ACL)")
    if "acl_identity_sees_nothing" in result.flags:
        lines.append("")
        lines.append(
            "- **Every question was refused because retrieval returned nothing.**"
            f" The ACL identity `{result.user_id}` is not in any document's"
            " `acl.users`, so the SQL ACL predicate correctly filtered out the"
            " whole corpus. This is a harness configuration problem, not a"
            " pipeline result. Re-run with `--user-id <a user listed in the"
            " corpus ACL>` — the dashboard's Overview tab shows them under"
            " “Known ACL users”."
        )
    if result.judge == "none":
        lines.append(
            "- **Groundedness is unmeasured** without a judge. Wire an LLM judge or pass "
            "`--judge lexical` for a labelled deterministic proxy; the gate will not pass "
            "on an unmeasured score."
        )
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Result | Detail |")
    lines.append("|---|---|---|---|---|")
    for gate in result.gates:
        actual = "n/a" if gate.actual is None else gate.actual
        gate_result = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"| {_cell(gate.name)} | {_cell(gate.threshold)} | {_cell(actual)} "
            f"| {gate_result} | {_cell(gate.detail)} |"
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| groundedness (judged) | {_format_score(result.groundedness)} |")
    lines.append(
        f"| groundedness (structural) | {_format_score(result.groundedness_structural)} |"
    )
    lines.append(f"| pii leaks | {result.pii_leaks} |")
    lines.append(f"| behavior correctness | {_format_score(result.refusal_correct)} |")
    lines.append(f"| expected-entity coverage | {_format_score(result.entity_coverage)} |")
    lines.append(f"| latency p50 / p95 (ms) | {result.latency_p50_ms} / {result.latency_p95_ms} |")
    if result.flags:
        lines.append(f"| flags | {_cell(', '.join(result.flags))} |")
    lines.append("")

    lines.append("## Per question")
    lines.append("")
    lines.append(
        "| # | Category | Question | Expected | Actual | Grounded | PII | Sources | "
        "Latency ms | Trace |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for index, report in enumerate(result.per_question, start=1):
        sources = ",".join(report.cited_source_types) or "-"
        if report.missing_source_types:
            sources += f" (missing {','.join(report.missing_source_types)})"
        trace = report.trace_id or "-"
        if report.trace_id and len(report.trace_id) > 12:
            trace = report.trace_id[:12]
        lines.append(
            f"| {index} | {_cell(report.category)} | {_cell(_excerpt(report.question, 70))} "
            f"| {report.expected_behavior} | {report.actual_behavior} "
            f"| {_format_score(report.groundedness)} | {report.pii_leaks} "
            f"| {_cell(sources)} | {report.latency_ms} | {_cell(trace)} |"
        )
    lines.append("")

    problems = [
        report
        for report in result.per_question
        if report.flags or report.error or report.uncited_claims
    ]
    if problems:
        lines.append("## Diagnostics")
        lines.append("")
        for index, report in enumerate(result.per_question, start=1):
            if not (report.flags or report.error or report.uncited_claims):
                continue
            lines.append(f"### {index}. {_excerpt(report.question, 120)}")
            if report.error:
                lines.append(f"- error: `{report.error}`")
            if report.flags:
                lines.append(f"- flags: `{', '.join(report.flags)}`")
            if report.answer_excerpt:
                lines.append(f"- answer: {report.answer_excerpt}")
            for claim in report.uncited_claims:
                lines.append(f"- uncited claim: {claim}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run a golden-question suite against the /ask endpoint.",
    )
    parser.add_argument("--suite", default="golden", help="suite name (default: golden)")
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_PATH),
        help=f"golden JSON file (default: {DEFAULT_GOLDEN_PATH})",
    )
    parser.add_argument(
        "--corpus",
        default="fixtures",
        help="fixture corpus path, recorded in the report (default: fixtures)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"pipeline base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--judge",
        choices=("none", "lexical"),
        default="none",
        help="groundedness judge: none (unmeasured) or lexical (deterministic proxy)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"per-question HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument("--session-id", default="eval-suite", help="session id sent to /ask")
    parser.add_argument(
        "--user-id",
        default="eval-user",
        help=(
            "user id sent to /ask; retrieval enforces documents' acl.users, so "
            "this must be an identity the corpus actually grants (e.g. a "
            "fixture ACL user — see the dashboard's Known ACL users)"
        ),
    )
    parser.add_argument("--json", dest="json_path", default=None, help="write EvalResult JSON here")
    parser.add_argument(
        "--pii-roster",
        default=None,
        help="corpus manifest with pii.kinds.person names (default: auto-detect in --corpus)",
    )
    return parser.parse_args(argv)


def _resolve_judge(name: str) -> JudgeFn | None:
    if name == "none":
        return None
    if name == "lexical":
        return lexical_judge
    raise ValueError(f"unknown judge {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint; returns the process exit code."""
    args = _parse_args(argv)
    try:
        golden = load_golden_set(args.golden)
    except GoldenSetError as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 2

    judge = _resolve_judge(args.judge)
    roster_path = Path(args.pii_roster) if args.pii_roster else default_roster_path(args.corpus)
    known_names = load_pii_roster(roster_path) if roster_path is not None else []
    result = asyncio.run(
        run_suite(
            golden.questions,
            base_url=args.base_url,
            judge=judge,
            judge_name=args.judge,
            timeout_s=args.timeout,
            session_id=args.session_id,
            user_id=args.user_id,
            suite=args.suite,
            known_names=known_names,
            progress=True,
        )
    )
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=False),
            encoding="utf-8",
        )
    print(render_markdown(result, base_url=args.base_url, corpus=args.corpus), end="")
    return exit_code_for(result)


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
