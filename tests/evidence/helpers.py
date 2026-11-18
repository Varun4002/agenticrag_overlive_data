"""Builders shared by evidence tests (no network, no database)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

from llm.base import CompletionRequest, CompletionResponse, Usage
from orchestrator.pipeline import ScrubbedChunk, ScrubbedPack

FIXED_TIME = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "PHX-150",
    title: str = "Cut over billing webhooks",
) -> ScrubbedChunk:
    """One scrubbed chunk with deterministic metadata."""
    return ScrubbedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=0.9,
        source="jira",
        source_id=source_id,
        title=title,
        updated_at=FIXED_TIME,
    )


def make_pack(question: str, *chunks: ScrubbedChunk) -> ScrubbedPack:
    """A scrubbed pack for ``question``."""
    return ScrubbedPack(question=question, chunks=list(chunks))


def claims_json(*claims: dict[str, object]) -> str:
    """Serialize a model-style ``{"claims": [...]}`` payload."""
    return json.dumps({"claims": list(claims)})


class FakeClient:
    """Deterministic ``LLMClient`` that replays canned responses."""

    provider = "fake"

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeClient ran out of canned responses")
        return CompletionResponse(
            text=self._responses.pop(0),
            model="fake",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=1,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True
