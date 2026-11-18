# 07 — Memory Design

> **Status note (2026-11-18).** Only Tier 2 exists, and it is **Redis**, not the
> `mem.answer_cache` Postgres table (the table stays in the schema, unused).
> Keys are `sha256(normalized question | user_id | data version)`, the TTL is
> 900 s, and a version counter is bumped on every sync/backfill/seed write —
> ruling #14 in `docs/08-decisions.md`, code in `src/memory/answer_cache.py`,
> state of the component in `src/memory/HANDOFF.md`. **Live-verified
> (2026-11-18)** against the running API: cold question 9.102 s, identical repeat
> 0.095 s with `memory.cache_hit` in the trace (`intake → plan →
> memory.cache_hit → completed`, status `cache_hit`); a different `user_id`
> missed and ran the full pipeline (4.19 s); Redis keys are `answer_cache:` + 64
> hex chars and no value contains the question text or a person name;
> `POST /admin/cache/invalidate` returned
> `{invalidated: true, cleared: 2, version: 1}`. Still unverified: a mid-request
> Redis outage, concurrent requests, TTL expiry. The per-source watermark and
> `content_hash` freshness rule described below is **not implemented**: staleness
> is the version counter plus the TTL. Session scratchpad (Tier 1) and entity
> registry (Tier 3) are unbuilt, as are the cache stampede lock and the
> correction path.

Memory exists for one purpose: **re-access without re-paying full cost** —
latency (caching), token cost (skipping re-planning), and consistency (same
entity named the same way across turns and systems).

Three tiers, all in Postgres, all cheap to inspect in the audit trail.

## Tier 1 — Working memory (session)

- `mem.session_turns`: last N turns of the current conversation.
- Used by: planner (pronoun resolution — "what about *its* blockers?",
  "compare to *last week's* report"), synthesizer (tone/format continuity).
- Scope: session only, no long-term persistence. Bounded (last 10 turns
  injected; older summarized in one line by a cheap LLM call if needed).

## Tier 2 — Episodic memory (answer cache)

- `mem.answer_cache`: question → verified answer + citations + evidence
  hashes + per-source watermarks.
- **Staleness rule (this is the important part):** a cached answer is valid
  only while (a) no source watermark has advanced past the recorded ones,
  and (b) every evidence chunk's `content_hash` still matches. Stale entries
  are never served; they are re-computed or downgraded to "hints for the
  planner" (e.g., which sources/entities were relevant).
- Keying: normalized question text + resolved-entity fingerprint (so
  "PHX deadlines" and "Project Phoenix deadlines this week" can share an
  entry when freshness allows).
- Personal queries ("tasks assigned to me") are keyed per user and never
  served to another user.

## Tier 3 — Semantic memory (entity registry)

- `mem.entity_registry`: canonical names + aliases + source refs for
  projects, tasks, teams, pages.
- Built by a background job that scans ingested documents for candidate
  entities (Jira project keys, Notion page titles, repeated task names) and
  clusters them; matching LLM pass confirms merges with a confidence score.
- Used by:
  - **Planner** — resolves "Phoenix" → `PHX` (Jira) + Notion page + Gmail
    label before retrieval, so one query hits the right filters in all
    three systems.
  - **Answer cache** — entity fingerprints for stable cache keys.
  - **PII scrubber** — person entities registered here help distinguish
    "person name" from "project named after a person", cutting Presidio
    false positives on things like a task called "Alice migration".
- Corrections: the user can fix a wrong resolution; the correction is stored
  as a high-confidence registry row (learned preference).

## What memory is explicitly NOT

- Not a place where raw PII is stored (scrubbed data only).
- Not a place the model can write free-form "facts" — only structured
  entries produced by designated jobs (registry builder, cache writer,
  session recorder). This prevents memory poisoning via prompt injection:
  nothing in retrieved content can create memory directly.
- Not user-facing truth — the cache and registry are derivable state; any
  table can be rebuilt from documents + audit log.

## Failure modes handled

| Failure | Handling |
|---|---|
| Cache serves outdated info | watermark + hash staleness rule above |
| Wrong entity merge ("PHX" also matches a task) | confidence floor 0.5; below floor → planner asks user or falls back to plain retrieval |
| Memory growth unbounded | TTL on session turns, LRU eviction on cache beyond N entries, registry dedupe job |
| Injection tries to persist instructions | only designated jobs write memory; no free-form writes (see above) |
