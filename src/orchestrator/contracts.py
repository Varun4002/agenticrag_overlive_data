"""Frozen cross-agent contracts.

These Pydantic models are the interface between the API, the orchestrator, and
every downstream agent (see ``docs/agents/00-orchestrator.md`` and the interface
freeze rule in ``docs/04-orchestration.md``). Changing a field here requires
Lead sign-off — everything *behind* the contract is free to change.

M0 uses ``AskRequest``/``AskResponse`` with a placeholder answer; the state
machine that fills them for real arrives in M1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single piece of evidence an answer claim points at."""

    chunk_id: str
    document_id: str
    source: Literal["jira", "notion", "gmail"]
    title: str
    quote: str
    url: str | None = None


class Refusal(BaseModel):
    """Present when the system declines to answer."""

    reason: Literal[
        "insufficient_evidence",
        "out_of_scope",
        "guardrail_blocked",
        "error",
    ]
    detail: str | None = None


class AskRequest(BaseModel):
    """Incoming question."""

    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)


class AskResponse(BaseModel):
    """Answer, its citations, an optional refusal, and the trace to inspect."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refusal: Refusal | None = None
    trace_id: str
    # True when the orchestrator hit a budget limit and answered with
    # partial evidence (docs/agents/00-orchestrator.md failure table).
    degraded: bool = False
