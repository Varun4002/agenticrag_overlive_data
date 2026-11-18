"""Builders for synthesis tests (no network, no database)."""

from __future__ import annotations

from datetime import UTC, datetime

from llm.base import CompletionRequest, CompletionResponse, Usage
from orchestrator.pipeline import Claim, ScrubbedChunk, ScrubbedPack, VerifiedClaims

FIXED_TIME = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def chunk(chunk_id: str, text: str, *, source_id: str = "PHX-150") -> ScrubbedChunk:
    return ScrubbedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=0.9,
        source="jira",
        source_id=source_id,
        title="Cut over billing webhooks",
        url=f"https://jira.example/browse/{source_id}",
        updated_at=FIXED_TIME,
    )


def pack(*chunks: ScrubbedChunk) -> ScrubbedPack:
    return ScrubbedPack(question="When is the PHX-150 cutover?", chunks=list(chunks))


def verified(*claims: Claim, can_answer: bool = True) -> VerifiedClaims:
    return VerifiedClaims(
        question="When is the PHX-150 cutover?",
        claims=list(claims),
        sufficiency=0.9 if can_answer else 0.1,
        can_answer=can_answer,
        insufficiency_reason=(
            None if can_answer else "question coverage 0.10 is below the 0.50 threshold"
        ),
    )


def claim(text: str, *chunk_ids: str, claim_id: int = 0) -> Claim:
    return Claim(claim_id=claim_id, text=text, chunk_ids=list(chunk_ids) or ["c1"])


class FakeClient:
    """Deterministic LLM client that replays one canned answer."""

    provider = "fake"

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.requests: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        return CompletionResponse(
            text=self._answer,
            model="fake",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=1,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True
