"""Lead-owned pipeline contracts (internal data flow).

Frozen interfaces between the retrieval, PII, evidence, and synthesis agents.
Each agent owns its implementation; these shapes are the hand-off points, so
changing one requires Lead sign-off (docs/04-orchestration.md, "Interface
freeze rule").

Flow::

    EvidencePack  --scrub-->  ScrubbedPack  --verify-->  VerifiedClaims
      (agent 03)               (agent 04)                  (agent 05)
                                           --synthesize-->  answer + citations
                                                              (agent 06)

Rule that makes the chain safe: **no LLM sees an ``EvidenceChunk``.** The
scrubber is the only path from retrieval to evidence verification.

Freeze-comment amendments (Lead-approved, ruling #20 in ``docs/08-decisions.md``):
``SynthesisResult.refusal_reason`` was added as an additive field with a default,
so every existing construction stays valid.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.contracts import Citation

SourceName = Literal["jira", "notion", "gmail"]


class EvidenceChunk(BaseModel):
    """One retrieved chunk, before PII scrubbing. Never sent to an LLM."""

    chunk_id: str
    document_id: str
    text: str
    # Ranking signal from the retriever (fused hybrid score).
    score: float
    source: SourceName
    source_id: str
    title: str
    url: str | None = None
    updated_at: datetime
    # True when the chunk predates the plan's freshness window; the planner
    # decides whether to trigger a live fetch before citing it as current.
    stale: bool = False
    # Chunker metadata (heading path, section, labels, fields, ...).
    meta: dict[str, object] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    """Ordered candidates for one question, ACL-filtered at query time."""

    question: str
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    dropped_below_floor: int = 0
    # Set when retrieval fell back (e.g. embedding failure -> FTS-only).
    degraded_reason: str | None = None


class ScrubbedChunk(BaseModel):
    """A chunk after PII redaction. Placeholders are opaque, stable tokens."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source: SourceName
    source_id: str
    title: str
    url: str | None = None
    updated_at: datetime
    stale: bool = False
    meta: dict[str, object] = Field(default_factory=dict)
    redacted_types: list[str] = Field(default_factory=list)


class ScrubbedPack(BaseModel):
    """What the Evidence Agent and Synthesizer are allowed to see."""

    question: str
    chunks: list[ScrubbedChunk] = Field(default_factory=list)
    redaction_count: int = 0
    degraded_reason: str | None = None


class Claim(BaseModel):
    """One atomic assertion about work state, traceable to chunks."""

    claim_id: int
    text: str
    chunk_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class VerifiedClaims(BaseModel):
    """The only material the Synthesizer may use."""

    question: str
    claims: list[Claim] = Field(default_factory=list)
    dropped_claims: list[Claim] = Field(default_factory=list)
    sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    can_answer: bool = False
    # Why it cannot answer, when can_answer is False.
    insufficiency_reason: str | None = None


class SynthesisResult(BaseModel):
    """The user-facing answer, produced only from verified claims."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_claim_ids: list[int] = Field(default_factory=list)
    refused: bool = False
    # The deterministic reason when ``refused`` is true ("answer truncated by the
    # token limit", "the draft answer could not be tied back to the evidence",
    # "insufficient evidence: ..."), so the orchestrator can report an honest
    # typed refusal instead of a generic fallback.
    refusal_reason: str | None = None
