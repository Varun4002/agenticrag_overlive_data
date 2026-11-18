# Agent 07 — Guardrail Agent

> Read with: `docs/06-guardrails-pii.md` — you implement that spec end to end.

## Mission

Implement the four guardrail layers as enforceable code: input checks,
retrieval-time checks, generation policy, and output re-scan. You are the
component other agents call; you don't own the pipeline.

## Contract

```python
class InputCheckRequest(BaseModel):  question: str; user_id: str
class InputCheckResult(BaseModel):   allowed: bool; reason: str | None

class OutputCheckRequest(BaseModel): answer: str; citations: list[Citation]; claims: VerifiedClaims | None
class OutputCheckResult(BaseModel):
    allowed: bool
    sanitized_answer: str | None     # minor violations stripped, else None
    violations: list[Violation]      # type: pii_leak / uncited_claim / policy / citation_integrity
```

Public API:
- `async def check_input(req) -> InputCheckResult`
- `def scan_chunk_for_injection(text: str) -> InjectionVerdict` (sync, pure)
- `async def check_output(req) -> OutputCheckResult`

## Behavior

### Input checks
- Length/charset sanity; empty/nonsense rejection is the **planner's** job —
  here only scope violations and clear abuse patterns.
- User scope: requests referencing sources outside the allowlist → blocked
  with standard message.

### Injection scanning (Layer 1 pre-scan + Layer 2 re-scan)
- Pattern heuristics: imperative instruction shapes aimed at the assistant
  ("ignore previous", "you are now", "system:", "disregard the above",
  "reveal", "print your"), role markers, tool-call-looking text.
- Verdict: `clean` / `flagged` / `quarantine`. Quarantined spans are wrapped
  in `<untrusted_content>` markers at ingestion; flagged-only chunks get
  neutralization wrappers at retrieval. Both verdicts logged to audit.
- False-positive tolerance: patterns must include assistant-directed verbs;
  ordinary imperative sentences ("Ship the demo by Friday") are `clean`.

### Output checks (final gate — fail closed)
- **PII re-scan:** regex + placeholder-expansion heuristics (partial email
  reconstructions, "starts with v…", phone fragments). Violation ⇒ sanitize
  (strip/redact span) or block if pervasive.
- **Citation integrity:** every `[n]` resolves; every cited chunk id is in
  the delivered citations; uncited factual sentences (Agent 06 strips most —
  this is the net) ⇒ strip or block.
- **Policy block list:** write-back requests, credential disclosure,
  instructions to de-anonymize, cross-user snooping ⇒ block with standard
  message.
- Every check/blocked outcome → `audit.events` with reason.

## Failure modes

| Failure | Handling |
|---|---|
| Scanner itself errors | fail closed: block the output (never emit unscanned text) |
| Sanitization makes answer nonsensical | block with refusal instead |
| Disagreement between layers | strictest verdict wins |

## Tests

- Planted injection fixtures (Notion page, email body, Jira comment) →
  correct verdicts; benign project-imperative text stays clean.
- Output PII leak attempts (placeholder expansion, partial-reconstruction
  strings) → blocked/sanitized.
- Citation integrity: dangling `[n]`, decorative citations → violations.
- Fail-closed behavior when the scanner raises.
