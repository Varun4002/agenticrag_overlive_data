"""Shared fakes for orchestrator tests: no database, no network, no model."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory.answer_cache import normalize_question
from orchestrator.contracts import AskResponse, Citation
from orchestrator.pipeline import (
    Claim,
    EvidenceChunk,
    EvidencePack,
    ScrubbedChunk,
    ScrubbedPack,
    SynthesisResult,
    VerifiedClaims,
)
from orchestrator.run import Stages

NOW = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def make_evidence_chunk(
    chunk_id: str = "c1",
    *,
    text: str = "PHX-150 · comment\nThe cutover moves to 2026-11-27.",
    document_id: str = "doc-1",
    source: str = "jira",
    source_id: str = "PHX-150",
    title: str = "PHX-150 — Cut over billing webhooks",
    score: float = 0.03,
    stale: bool = False,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title=title,
        url=f"https://example.atlassian.net/browse/{source_id}",
        updated_at=NOW,
        stale=stale,
        meta={"heading_path": ["Cutover"]},
    )


def make_pack(
    question: str = "When is the PHX-150 cutover?",
    *chunks: EvidenceChunk,
    degraded_reason: str | None = None,
) -> EvidencePack:
    return EvidencePack(
        question=question,
        chunks=list(chunks) or [make_evidence_chunk()],
        dropped_below_floor=0,
        degraded_reason=degraded_reason,
    )


def make_scrubbed(pack: EvidencePack, *, redactions: int = 0) -> ScrubbedPack:
    chunks = [
        ScrubbedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text.replace("Priya", "[PERSON_1]"),
            score=chunk.score,
            source=chunk.source,
            source_id=chunk.source_id,
            title=chunk.title,
            url=chunk.url,
            updated_at=chunk.updated_at,
            stale=chunk.stale,
            meta=chunk.meta,
            redacted_types=["person"] if redactions else [],
        )
        for chunk in pack.chunks
    ]
    return ScrubbedPack(
        question=pack.question,
        chunks=chunks,
        redaction_count=redactions,
        degraded_reason=pack.degraded_reason,
    )


def make_verified(
    *,
    can_answer: bool = True,
    claims: int = 2,
    sufficiency: float = 0.9,
    reason: str | None = None,
) -> VerifiedClaims:
    items = [
        Claim(
            claim_id=index + 1,
            text=f"claim {index + 1}",
            chunk_ids=["c1"],
            confidence=0.9,
        )
        for index in range(claims)
    ]
    return VerifiedClaims(
        question="When is the PHX-150 cutover?",
        claims=items if can_answer else [],
        dropped_claims=[] if can_answer else items,
        sufficiency=sufficiency if can_answer else 0.1,
        can_answer=can_answer,
        insufficiency_reason=None if can_answer else (reason or "not covered by evidence"),
    )


def make_synthesis(
    *,
    refused: bool = False,
    answer: str = "The cutover is 2026-11-27 [1].",
    citations: int = 1,
) -> SynthesisResult:
    cites = [
        Citation(
            chunk_id="c1",
            document_id="doc-1",
            source="jira",  # type: ignore[arg-type]
            title="PHX-150 — Cut over billing webhooks",
            quote="The cutover moves to 2026-11-27.",
            url="https://example.atlassian.net/browse/PHX-150",
        )
        for _ in range(citations)
    ]
    return SynthesisResult(
        answer=answer,
        citations=[] if refused else cites,
        used_claim_ids=[] if refused else [0],
        refused=refused,
    )


@dataclass
class FakeStages:
    """Records call order/arguments and returns (or raises) canned results."""

    pack: EvidencePack = field(default_factory=make_pack)
    scrubbed: ScrubbedPack | None = None
    verified: VerifiedClaims = field(default_factory=make_verified)
    synthesis: SynthesisResult = field(default_factory=make_synthesis)
    retrieve_error: BaseException | None = None
    scrub_error: BaseException | None = None
    verify_error: BaseException | None = None
    synthesize_error: BaseException | None = None
    retrieve_delay_s: float = 0.0
    verify_delay_s: float = 0.0
    synthesize_delay_s: float = 0.0
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scrubbed is None:
            self.scrubbed = make_scrubbed(self.pack)

    def order(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def args(self, name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for call_name, args, kwargs in self.calls:
            if call_name == name:
                return args, kwargs
        raise AssertionError(f"{name} was not called; order={self.order()}")

    def as_stages(self) -> Stages:
        async def retrieve(*args: Any, **kwargs: Any) -> EvidencePack:
            self.calls.append(("retrieve", args, kwargs))
            if self.retrieve_delay_s:
                await asyncio.sleep(self.retrieve_delay_s)
            if self.retrieve_error is not None:
                raise self.retrieve_error
            return self.pack

        async def scrub(*args: Any, **kwargs: Any) -> ScrubbedPack:
            self.calls.append(("scrub", args, kwargs))
            if self.scrub_error is not None:
                raise self.scrub_error
            assert self.scrubbed is not None
            return self.scrubbed

        async def verify(*args: Any, **kwargs: Any) -> VerifiedClaims:
            self.calls.append(("verify", args, kwargs))
            if self.verify_delay_s:
                await asyncio.sleep(self.verify_delay_s)
            if self.verify_error is not None:
                raise self.verify_error
            return self.verified

        async def synthesize(*args: Any, **kwargs: Any) -> SynthesisResult:
            self.calls.append(("synthesize", args, kwargs))
            if self.synthesize_delay_s:
                await asyncio.sleep(self.synthesize_delay_s)
            if self.synthesize_error is not None:
                raise self.synthesize_error
            return self.synthesis

        return Stages(
            retrieve=retrieve, scrub=scrub, verify=verify, synthesize=synthesize
        )


@dataclass
class CachePut:
    """One recorded ``put``: exactly what the orchestrator asked to store."""

    question: str
    user_id: str
    version: int
    response: AskResponse
    ttl_s: int


@dataclass
class FakeAnswerCache:
    """In-memory answer cache: dict + version counter + TTL bookkeeping.

    Normalizes the question the same way ``memory.answer_cache`` does, so a
    casing/whitespace variant of a cached question is a hit — that is part of
    the contract under test, not a convenience.
    """

    responses: dict[tuple[str, str, int], AskResponse] = field(default_factory=dict)
    puts: list[CachePut] = field(default_factory=list)
    gets: list[tuple[str, str, int]] = field(default_factory=list)
    version_value: int = 0
    get_error: BaseException | None = None
    put_error: BaseException | None = None
    version_error: BaseException | None = None
    closed: bool = False

    def seed(
        self,
        question: str,
        user_id: str,
        response: AskResponse,
        *,
        version: int | None = None,
    ) -> None:
        """Preload an entry as if an earlier request had answered and cached it."""
        self.responses[
            (
                normalize_question(question),
                user_id,
                self.version_value if version is None else version,
            )
        ] = response

    def normalized_puts(self) -> list[tuple[str, str, int]]:
        """The (normalized question, user, version) each ``put`` wrote."""
        return [(normalize_question(p.question), p.user_id, p.version) for p in self.puts]

    async def get(self, question: str, user_id: str, version: int) -> AskResponse | None:
        if self.get_error is not None:
            raise self.get_error
        self.gets.append((normalize_question(question), user_id, version))
        return self.responses.get((normalize_question(question), user_id, version))

    async def put(
        self,
        question: str,
        user_id: str,
        version: int,
        response: AskResponse,
        ttl_s: int,
    ) -> None:
        self.puts.append(
            CachePut(
                question=question,
                user_id=user_id,
                version=version,
                response=response,
                ttl_s=ttl_s,
            )
        )
        if self.put_error is not None:
            raise self.put_error
        self.responses[(normalize_question(question), user_id, version)] = response

    async def version(self) -> int:
        if self.version_error is not None:
            raise self.version_error
        return self.version_value

    async def bump_version(self) -> int:
        self.version_value += 1
        return self.version_value

    async def clear(self) -> int:
        removed = len(self.responses)
        self.responses.clear()
        return removed

    async def aclose(self) -> None:
        self.closed = True

