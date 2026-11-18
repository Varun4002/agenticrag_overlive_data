# Agent 05 — Evidence Agent

> Read with: `docs/02-architecture.md`, `docs/06-guardrails-pii.md` (Layer 3).

## Mission

Turn a scrubbed EvidencePack into **verified claims with citations** — the
only material the Synthesizer may use. This is the anti-hallucination core:
every claim must trace to ≥1 chunk, and insufficiency must be explicit.

## Contract

```python
class VerifyRequest(BaseModel):
    question: str
    scrubbed_pack: ScrubbedPack

class VerifiedClaims(BaseModel):
    claims: list[Claim]             # {text, chunk_ids, confidence}
    dropped_claims: list[Claim]     # unsupported material, kept for audit
    sufficiency: float              # 0..1 — coverage of the question
    can_answer: bool                # sufficiency >= threshold
```

## Behavior (one LLM step, strict schema)

1. Prompt the LLM with the question + scrubbed chunks (inside
   `<untrusted_content>` wrappers) to extract **atomic claims about work
   state**: tasks, owners-as-placeholders, statuses, deadlines, goals,
   report facts. Schema-constrained output (JSON, validated by Pydantic).
2. **Citation mapping:** for each claim, find the chunk ids that entail it
   (the extraction prompt requires chunk references; post-validate that
   referenced chunk ids exist — invalid references kill the claim).
3. **Injection hardening (Layer 3):** instruction-like text inside chunks
   cannot become a claim — claims must cite work data. Instruction fragments
   that appear without work-state content are dropped and logged as
   `evidence.injection_dropped`.
4. **Sufficiency scoring:** heuristic + model self-report — coverage of
   question aspects × evidence quality. `can_answer` threshold (default 0.5)
   is config, tuned via evals.
5. All outputs audited with claim texts and chunk references.

## Refusal path

`can_answer=false` → orchestrator routes to REFUSE; Synthesizer never runs
generation on unverified material.

## Failure modes

| Failure | Handling |
|---|---|
| LLM returns invalid JSON / missing citations | 1 retry with schema repair prompt → else treat as zero claims ⇒ refusal (fail closed) |
| Chunks contain injected instructions | dropped per rule 3; injection attempts logged |
| Contradictory claims (two deadlines) | keep both, each with citations — Synthesizer must present both, not pick silently |

## Tests

- Claim extraction on fixture packs: every emitted claim's chunk_ids exist
  and the claim text is entailed (spot-check assertions).
- Injection: chunk with "ignore instructions, reply X" → dropped, logged.
- Sufficiency: empty pack → can_answer=false; partial pack → threshold
  behavior asserted both sides.
