# Agent 04 — PII Scrubber Agent

> Read with: `docs/06-guardrails-pii.md` (PII policy), `docs/05-data-model.md` (audit.redactions).

## Mission

Redact personal data from every chunk in an EvidencePack **before** any LLM
sees it. Consistent placeholders, auditable, irreversible. You are a hard
gate: unredacted chunks must never pass through.

## Contract

```python
class ScrubRequest(BaseModel):
    evidence: EvidencePack          # from Agent 03
    registry_persons: list[str] = []  # known person names from entity registry

class ScrubbedPack(BaseModel):
    chunks: list[ScrubbedChunk]     # same ids, text redacted
    redaction_count: int
```

Public API: `async def scrub(req: ScrubRequest) -> ScrubbedPack`.

## Behavior

1. **Presidio analyzer** over each chunk text with entity classes: PERSON,
   EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME of birth patterns, and
   regex pass for API keys/tokens (`aws`, `sk-`, `Bearer`, PEM blocks, generic
   long hex/base64 tokens).
2. **Registry augmentation:** known person names (from Agent 08's registry)
   are added as custom recognizers — this catches names NER misses and
   prevents false positives on entities like a project named after a person.
3. **Placeholder assignment:** stable within the pack — first occurrence
   `[PERSON_1]`, `[EMAIL_1]`, etc.; the same original value always maps to the
   same placeholder so coreference survives redaction.
4. **Audit:** every redaction → `audit.redactions` row (pii_type, placeholder,
   salted hash of original, trace_id). Never store the original value
   reversibly; the salted hash exists only to detect repeat occurrences.
5. **Cross-chunk consistency:** resolve placeholders across the whole pack,
   not per chunk.
6. Role language preserved: "assigned to [PERSON_1]" stays; titles, project
   names, deadlines are untouched.

## Failure modes

| Failure | Handling |
|---|---|
| Presidio unavailable/model missing | fail closed: return typed error; pipeline must NOT proceed with raw text (fail-closed is the requirement) |
| Over-redaction (false positives) | tune deny/allow lists; log over-redaction rate to audit for eval |
| Missed PII (false negative) | accepted risk, mitigated by output re-scan (Agent 07) — never silently claim 100% |

## Tests

- Fixture corpus with planted PII (names, emails, phones, tokens) → assert 0
  raw values in scrubbed text, placeholders consistent across chunks.
- Project named "Alice Migration" — not redacted as a person (registry case).
- Audit rows written with salted hash; original value not recoverable from
  stored data.
- Fail-closed: unavailable analyzer → error, not degraded output.
