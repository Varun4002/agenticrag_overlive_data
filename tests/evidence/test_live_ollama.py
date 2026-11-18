"""Opt-in live check against the local model. Skipped unless LIVE_LLM=1."""

from __future__ import annotations

import os

import pytest
from helpers import make_chunk, make_pack

from evals.fixtures import FixtureCorpus
from evidence.service import verify

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVE_LLM"),
    reason="live model check: set LIVE_LLM=1 and have Ollama running",
)


async def test_live_extraction_cites_only_provided_chunks(corpus: FixtureCorpus) -> None:
    document = corpus.get("jira", "PHX-150")
    pack = make_pack(
        "When is the PHX-150 cutover?",
        make_chunk("live-1", document.content[:1500], source_id=document.source_id),
    )

    result = await verify("When is the PHX-150 cutover?", pack, trace_id="live-trace")

    provided = {chunk.chunk_id for chunk in pack.chunks}
    assert result.claims, "the local model produced no claims for a real document"
    assert all(set(claim.chunk_ids) <= provided for claim in result.claims)
    assert 0.0 <= result.sufficiency <= 1.0
