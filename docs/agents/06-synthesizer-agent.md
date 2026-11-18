# Agent 06 — Synthesizer Agent

> Read with: `docs/02-architecture.md`, `docs/06-guardrails-pii.md` (Layer 3).

## Mission

Write the final user-facing answer **strictly from VerifiedClaims**, with
inline citations, or a refusal message. You are a formatter + composer, not a
reasoner: no new facts may appear that the Evidence Agent did not verify.

## Contract

```python
class SynthesizeRequest(BaseModel):
    question: str
    claims: VerifiedClaims
    session_context: SessionContext | None   # last turns for tone/format
    refusal_reason: str | None               # set => write refusal instead

class SynthesisResult(BaseModel):
    answer: str                              # with [n] citations
    citations: list[Citation]                # resolved, ordered by first use
    used_claim_ids: list[int]
```

## Behavior

1. **Normal path:** one LLM call. System prompt rules (non-negotiable):
   - Use only claim content. Paraphrase, don't invent. Numbers/dates copied
     verbatim from claims.
   - Every factual sentence carries `[n]` pointing to the claim's citation.
   - Placeholders (`[PERSON_1]`) are opaque tokens — never guess or expand
     them.
   - Contradictory claims → present both with citations; never silently pick.
   - Structure to fit question type: task list, deadline table, summary
     paragraph, status report (done / in progress / at risk).
2. **Refusal path** (`refusal_reason` set): deterministic template, no LLM
   needed — state what was searched, what was found, and one concrete
   suggestion ("I found tasks but nothing about X — try naming the project").
   Never a bare "I don't know".
3. Citation resolution: `[n]` → `Citation {source, source_id, title, url,
   updated_at}` from the chunks behind the claims.

## Failure modes

| Failure | Handling |
|---|---|
| LLM emits content without citation | post-validation: strip uncited sentences; if that guts the answer → refusal + log |
| LLM expands a placeholder | caught downstream by output guardrail; also prompt-blocked here |
| Empty claims with can_answer=true (logic bug upstream) | refuse + log invariant violation |

## Tests

- Golden fixtures: claims in → answer out, assert every factual sentence has
  a citation (regex parse of `[n]` vs claims).
- Uncited-sentence stripping; refusal template correctness.
- Placeholder expansion attempt → stripped/refused.
