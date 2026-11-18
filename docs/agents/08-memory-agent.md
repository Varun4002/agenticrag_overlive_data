# Agent 08 — Memory Agent

> Read with: `docs/07-memory-design.md` (your spec), `docs/05-data-model.md` (mem.* tables).

## Mission

Implement the three memory tiers: session scratchpad, answer cache with
staleness rules, and the entity registry. You make re-access cheap **without
ever serving stale or poisoned memory**.

## Contract

```python
class MemoryCheckRequest(BaseModel):
    question: str; user_id: str; session_id: str; entity_filters: list[EntityRef]

class MemoryCheckResult(BaseModel):
    cached_answer: CachedAnswer | None   # only if verifiably fresh
    stale_hint: StaleHint | None         # prior relevant query, too old to serve
    resolved_entities: list[EntityRef]
    session_context: SessionContext      # last-N turns

class CacheWriteRequest(BaseModel):
    question: str; user_id: str
    answer: str; citations: list[Citation]
    evidence_hashes: list[str]
    source_versions: dict[str, datetime]
```

Public API:
- `async def recall(req) -> MemoryCheckResult` (read path)
- `async def remember(req) -> None` (write path — called after successful answer)
- `async def record_turn(session_id, role, content, entity_hits)` 
- Registry builder: `async def update_registry()` — background job.

## Behavior

### Recall
1. Session turns loaded (last 10, oldest summarized if needed).
2. Cache lookup keyed on normalized question + entity fingerprint + user.
   **Staleness gate (non-negotiable):** serve only if every source watermark
   in `source_versions` is still ≥ current watermark for that source AND
   every `evidence_hashes` entry still matches the chunk's `content_hash`.
   Stale → demoted to `stale_hint` (sources/entities worth re-querying),
   never served as an answer.
3. Entity resolution: aliases GIN lookup; confidence ≥ 0.5 auto-resolves;
   ambiguous (multiple ≥ 0.5 hits) → return conflict, let planner ask or
   fall back to plain retrieval.

### Remember
- Write the cache row only after the output guardrail passed. A blocked
  answer is never cached.

### Registry builder (background job)
- Scan ingested docs for candidates: Jira project keys, Notion page titles,
  recurring task names, assignee names.
- Cluster similar candidates (trgm similarity); one LLM pass to confirm
  merges; write with confidence scores.
- User corrections (feedback API, post-MVP surface) → high-confidence rows
  that outrank inferred ones.

## Hard rules

- **Only designated jobs write memory.** No free-form writes. This is what
  makes memory poisoning via prompt injection impossible — retrieved content
  never creates memory.
- No raw PII stored. Person names in the registry are entity references used
  by the PII scrubber; store the registry entries themselves, not document
  excerpts containing personal data.
- Memory is derivable state — any table must be rebuildable from
  `core.documents` + `audit.*`.

## Failure modes

| Failure | Handling |
|---|---|
| Registry merge wrong (two entities merged) | confidence floor + conflict handling; corrections outrank inference |
| Cache stampede on identical fresh question | short in-flight lock so one computation serves all |
| Watermark clock skew across sources | compare against per-source `core.sync_state` only, never wall clock |

## Tests

- Staleness: change a chunk content hash after caching → recall demotes to
  hint, does not serve the answer.
- User isolation: user B never gets user A's personal cached answer.
- Registry: alias resolution, conflict case, confidence floor.
- No free-form write path exists (API surface test).
