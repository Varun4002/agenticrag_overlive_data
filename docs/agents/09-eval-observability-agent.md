# Agent 09 — Eval & Observability Agent

> Read with: `docs/01-vision-scope.md` (success criteria), `docs/02-architecture.md`, `docs/05-data-model.md` (audit.events).

## Mission

Make the system's quality **measurable and repeatable**: the golden question
set, the eval harness, and the trace/observability tooling. You are the
acceptance gate — "it seems to work" is not evidence, your reports are.

## Contract

```python
class GoldenQuestion(BaseModel):
    question: str
    expected_entities: list[str]
    expected_behavior: Literal["answer", "refuse"]
    must_cite_source_types: list[str]     # e.g. ["jira", "notion"]
    notes: str = ""

class EvalResult(BaseModel):
    groundedness: float          # fraction of claims with valid citations
    pii_leaks: int               # must be 0
    refusal_correct: float       # refused when should, answered when could
    latency_p50_ms / p95_ms: int
    per_question: list[QuestionReport]
```

Public API:
- `python -m src.evals.run --corpus fixtures --suite golden`
- `def trace_report(trace_id: str) -> TraceSummary` — full agent decision
   timeline from `audit.events`.

## Deliverables

### 1. Golden question set (start building it in M1 — it gates later waves)
- 30–40 questions over the fixture corpus, spanning: tasks by project,
  overdue/deadline queries, goal summaries, email timeline queries, status
  reports, and refusal cases (questions the corpus cannot answer).
- Expected answers derived **from the fixtures themselves** — labeled by
  hand from fixture content, never from live-system guesses.

### 2. Eval harness
- Runs the full pipeline against a fixture-seeded DB, scores:
  - **Groundedness:** parse `[n]` citations; a claim is grounded iff its
    citation resolves to a real chunk that plausibly entails it (LLM-as-judge
    pass with the claim + cited chunk; judge prompt must be skeptical).
  - **PII leak scan:** regex + placeholder-expansion heuristics over the
    final answer (independent reimplementation, not a shared library with
    Agent 07 — a shared bug would be invisible).
  - **Refusal correctness** for the refusal cases.
- Emits `EvalResult` + markdown report.

### 3. Observability tooling
- `trace_report` used in code review of every milestone.
- Aggregate dashboards-as-scripts: latency by agent, refusal rate,
  injection-quarantine rate, cache hit rate, over-redaction rate.

## Rules

- Evals are deterministic where possible: fixed seeds, fixture corpus, no
  live API dependence. LLM-judge scores get a documented variance caveat.
- Never tune the system against the eval set by hand-tuning answers — change
  behavior, re-run, compare.
- Gate criteria per milestone come from `docs/03-mvp-plan.md` "Done when".

## Failure modes

| Failure | Handling |
|---|---|
| Judge model unavailable | report groundedness as `null` + flag, never silently 1.0 |
| Eval set drift vs schema changes | suite asserts schema versions; mismatch = hard failure |
| Flaky evals | quarantine flaky case + root-cause it; do not delete quietly |

## Tests

- Harness self-tests: a deliberately-bad pipeline (uncited answers, planted
  PII, wrong refusals) must score badly — proving the harness can fail.
