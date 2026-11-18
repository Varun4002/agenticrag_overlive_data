# Agent 03 — Retrieval Agent

> Read with: `docs/02-architecture.md`, `docs/05-data-model.md`, `docs/06-guardrails-pii.md` (Layer 2).

## Mission

Given a planned query (entity filters, freshness needs, sources), produce an
**EvidencePack**: the best-ranked, ACL-clean candidate chunks with their
metadata. You do not call the LLM.

## Contract

```python
class RetrieveRequest(BaseModel):
    query_text: str
    entity_filters: list[EntityRef] = []     # resolved entities from planner
    sources: list[str] | None = None         # restrict to jira/notion/gmail
    freshness_window: timedelta | None = None
    top_k: int = 12

class EvidencePack(BaseModel):
    chunks: list[EvidenceChunk]              # id, text, meta, score, doc url, updated_at
    recency_rank: list[UUID] | None          # chunks flagged as possibly stale
    dropped_below_floor: int
```

## Behavior

1. Build the SQL query: hybrid score = cosine similarity (pgvector ANN) +
   `ts_rank_cd` full-text (pg_trgm/FTS), combined via reciprocal rank fusion
   (k=60 — simple, robust, no tuning dependency).
2. **ACL filter is a WHERE clause**, never post-filtering (Layer 2 guardrail).
   Source/scope filters from the plan likewise.
3. Recency boost: decay by document `updated_at` age (half-life ~30 days;
   deadline/date-bearing docs get a date-proximity bonus when the query
   mentions dates).
4. Freshness flagging: chunks older than `freshness_window` get returned but
   flagged — the planner may trigger a live fetch before citing them as current.
5. Score floor: drop the tail below floor; log the count.
6. Order output by final score; annotate each chunk with its citation
   coordinates (issue key / page title + heading / email subject + date).

## Failure modes

| Failure | Handling |
|---|---|
| Zero results | return empty pack (orchestrator handles refusal path) — never pad with low scores |
| Embedding of query fails | typed error; fallback to pure FTS retrieval, flagged in trace |
| Stale watermark vs requested freshness | return pack + `recency_rank`; planner decides on live fetch |

## Tests

- Ranking: fixture corpus, assert known-relevant chunk ranks top-3 for its
  query.
- ACL: chunk invisible without matching acl row — enforced in SQL (test by
  direct SQL inspection, not just result absence).
- Recency: two equally similar chunks with different ages — newer wins.
- Freshness flagging and floor-dropping behavior.
