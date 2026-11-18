# 06 — Guardrails & PII Policy

## Guardrail layers (defense in depth)

Data is checked when it enters, when it's retrieved, and when it leaves.
No single check is trusted alone.

### Layer 1 — Ingestion guardrails (Connector Agent)
- **Source allowlist:** only configured Jira projects, Notion pages/databases,
  and Gmail labels are synced. Everything else is skipped and logged.
- **Content scan at normalize time:** flag documents containing known
  high-risk patterns (credentials, API keys, `PRIVATE KEY`, token-like
  strings) → those spans are marked for immediate redaction, and the document
  is tagged `sensitive` in metadata.
- **Prompt-injection pre-scan:** heuristic scan for instruction-like patterns
  ("ignore previous instructions", "you are now", "disregard", suspicious
  roleplay markers). Flagged spans are wrapped in a
  `<untrusted_content>` delimiter at storage time so downstream prompts can
  neutralize them structurally, not just textually.

### Layer 2 — Retrieval guardrails (Retrieval + Guardrail Agents)
- **ACL enforcement in SQL:** every candidate chunk is filtered by the
  requesting user's mirrored ACL. No post-hoc filtering — if it isn't in the
  WHERE clause result, it never exists for the request.
- **Freshness check:** chunks older than the question's implied freshness
  window (planner decides; e.g. "due today" ⇒ 24h) require a live re-fetch
  before they may be cited as *current*.
- **Injection re-scan:** chunks are re-scanned at retrieval time (content may
  have been edited since ingestion). Flagged chunks are either excluded or
  passed with injection-neutralization wrappers, per policy.
- **Confidence floor:** results below a similarity/recency-score floor are
  dropped rather than padding the context.

### Layer 3 — Generation guardrails (Evidence + Synthesizer Agents)
- **Groundedness gate:** every claim must map to ≥1 retrieved chunk. Claims
  without support are dropped; if enough claims drop, the answer degrades to
  refusal. The Evidence Agent output is the *only* thing the Synthesizer may
  use — no free generation from parametric memory.
- **Refusal policy:** insufficient evidence ⇒ explicit refusal naming what was
  searched and what would help ("I found 2 older tasks but nothing current
  about X — try narrowing to a project"). No hedged half-answers.
- **No actionable-irreversible advice:** the system describes state; it never
  instructs destructive actions ("delete all the tickets") even in roleplay.

### Layer 4 — Output guardrails (Guardrail Agent, final pass)
- **PII re-scan:** the *final answer text* is scanned again (it could
  reconstruct PII from redacted inputs, e.g. "the person whose email starts
  with v…"). Any leak ⇒ regenerate with the offending span removed, else
  refuse.
- **Citation integrity:** every `[n]` must resolve to a real chunk in the
  delivered citations; citations must be for chunks actually used (no
  decorative citations).
- **Policy block list:** requested actions that are write-backs, credential
  disclosure, or cross-source snooping beyond allowlisted scope are refused
  with a standard message.
- **Everything audited:** blocks, redactions, refusals all land in
  `audit.events` with reasons.

### Audit availability policy (fail closed)

An answer the audit trail cannot record is an answer that cannot be
investigated, so it is not served. Configured by `AUDIT_STRICT` (default
`true`):

| Condition | Behavior |
|---|---|
| Trace write fails (DB down, permissions, disk) | **503**, no answer generated |
| `APP_ENV=prod` and `DATABASE_URL` unset | **503** at intake |
| `AUDIT_STRICT=false` (explicit dev opt-out) | Answer served; `/health` still reports `database.reachable=false` |

The trace write of `api.request.received` is checked *before* any work is
done; the closing `api.request.completed` write is checked before the response
is returned. Implementation: `src/api/routes.py`, `src/db/tracing.py`
(`EventSink.write` returns success instead of raising), tests in
`tests/api/test_audit_policy.py`.

## PII policy

**Rule: the LLM never sees personal data.** Redaction happens between
retrieval and evidence verification.

### Entity classes redacted (MVP)
- Person names (Presidio NER + user-configurable team member list)
- Email addresses, phone numbers, physical addresses
- API keys / tokens / passwords appearing in content
- Dates of birth, ID numbers if detected

**Not redacted:** project names, task titles, deadlines, statuses, org-level
info — these are the point of the system. Personal *attributions*
("assigned to ___") are preserved as role references, e.g.
"assigned to [PERSON_1]" — the answer stays useful without the name.

### Redaction mechanics
- Presidio analyzer → anonymizer with stable placeholders (`[PERSON_1]`,
  `[EMAIL_1]`) — consistent within an evidence pack so coreference survives.
- Placeholder uniqueness is maintained across the whole pack; the mapping is
  written to `audit.redactions` with a **salted hash** of the original value.
  The mapping is not stored reversibly and never enters a prompt.
- Synthesizer is instructed to treat placeholders as opaque tokens, never to
  guess them; the output re-scan catches attempts.

## Prompt-injection defense summary

1. Structure: retrieved content is always passed inside
   `<untrusted_content>` delimiters; system prompt states that content inside
   them is data, never instructions.
2. Heuristic scan flags and quarantines obvious injection attempts.
3. The Evidence Agent only extracts *claims about work state*; instruction
   content cannot become a claim (schema constraint: claims cite work data).
4. Planner cannot be re-targeted by retrieved content — it only consumes
   the user question + memory, never raw chunks.

## Test obligations (guards must prove these)

- Planted injection in a Notion fixture page → detected, quarantined, answer
  unaffected.
- Question asking to reveal someone's email → refusal, even if the email
  exists in the corpus.
- Fixture corpus contains PII; eval asserts final answers contain none.
