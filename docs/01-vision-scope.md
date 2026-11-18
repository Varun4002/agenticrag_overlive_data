# 01 — Vision & Scope

## What it is

An **agentic RAG system over live third-party work data** (Jira, Notion, Gmail).
Unlike static RAG over a fixed corpus, the source of truth is a live system:
data is pulled on demand and by scheduled sync, and every answer reflects the
current state of projects, tasks, deadlines, and reports.

## Core capabilities

1. **Live ingestion** — pull issues, pages, and emails from configured sources
   on a schedule, incrementally (only what changed since last sync).
2. **Question answering** over that data: "What are the open blockers on
   Project X?", "What deadlines land this week?", "Summarize the goals doc."
3. **Evidence-first answers** — every factual claim carries a citation back to
   the source item (issue key, page title + anchor, email subject + date).
   Unsupported claims are dropped; insufficient evidence produces an explicit
   refusal, not a guess.
4. **PII removal before the model sees data** — names, emails, phones, and
   similar personal data are redacted from retrieved content before synthesis.
   The LLM never sees raw personal data.
5. **Guardrails** — layered controls on what can be ingested, retrieved,
   generated, and remembered (see `06-guardrails-pii.md`). Includes defense
   against prompt injection hidden inside ingested documents.
6. **Memory for re-access** — session scratchpad, cached question→answer pairs
   (invalidated by data freshness), and a canonical entity registry mapping
   project names across systems ("Project Phoenix" ↔ `PHX` in Jira ↔ Notion
   page) so repeated/related queries skip full re-planning.

## Example queries the MVP must answer

- "What tasks are assigned to me in the Payments project and what's overdue?"
- "When is the next deadline for the onboarding revamp, and what's blocking it?"
- "What did the goals page say about Q1 OKRs — summarize it."
- "Any recent emails about the vendor contract? Give me the timeline."
- "Give me a status report on Project Phoenix: done, in progress, at risk."

## Users

- MVP: a single authenticated user (Varun's own workspace data), CLI + HTTP API.
- Post-MVP: small team, per-user ACL-mirrored retrieval.

## Non-goals (explicitly out of scope)

- **Write-back**: creating or updating Jira issues, Notion pages, or sending
  email. Read-only for everything.
- **Fine-tuning** any model.
- **Chat with the source systems** (no posting to Slack, no Outlook support).
- **Reversible de-anonymization** in the MVP: redaction is irreversible to the
  answer path. A restricted audit trail records redaction events for
  debugging, but the model cannot recover original values.
- **Multi-tenant SSO**, usage metering, web UI.

## Success criteria (system-level)

- ≥ 90% of golden-question answers pass groundedness eval (every claim cited).
- 0 PII leaks in the eval suite output scan.
- Answers about entities mentioned anywhere in the corpus with sufficient
  evidence; explicit "not enough evidence" otherwise (no hallucinated answers
  observed in eval).
- p95 end-to-end answer latency ≤ 15s for cached-index questions, ≤ 60s when a
  live fetch is triggered.
