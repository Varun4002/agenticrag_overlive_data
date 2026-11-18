"""Builders shared by the PII tests (no pytest fixtures here).

``tests/pii`` is added to ``sys.path`` by pytest, so tests import this module
directly — the same pattern as ``tests/connectors/jira_fixtures.py``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.pipeline import EvidenceChunk, EvidencePack

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "fixtures" / "corpus"

FIXED_NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

#: Row ids the audit tests reference; seeded into the scratch database.
DOCUMENT_ID = uuid.UUID(int=42)
CHUNK_ID = uuid.UUID(int=43)


def load_manifest() -> dict[str, Any]:
    return json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))


def make_chunk(
    text: str,
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    title: str = "Phoenix notes",
    source: str = "notion",
    source_id: str = "src-1",
) -> EvidenceChunk:
    source_name: Any = source
    return EvidenceChunk(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=document_id or str(uuid.uuid4()),
        text=text,
        score=0.5,
        source=source_name,
        source_id=source_id,
        title=title,
        updated_at=FIXED_NOW,
    )


def make_pack(*texts: str, question: str = "What is due this week?") -> EvidencePack:
    return EvidencePack(question=question, chunks=[make_chunk(text) for text in texts])


def corpus_chunks() -> list[EvidenceChunk]:
    """Every corpus document body as one evidence chunk, with stable ids."""
    chunks: list[EvidenceChunk] = []
    sources = {
        "jira_phoenix.json": "jira",
        "jira_atlas.json": "jira",
        "notion_pages.json": "notion",
        "gmail_threads.json": "gmail",
    }
    for filename, source in sources.items():
        payload = json.loads((CORPUS_DIR / filename).read_text(encoding="utf-8"))

        def walk(node: object, *, source: str = source) -> None:
            if isinstance(node, dict):
                content = node.get("content")
                if isinstance(content, str):
                    chunks.append(
                        make_chunk(
                            content,
                            chunk_id=str(uuid.UUID(int=len(chunks) + 1)),
                            document_id=str(uuid.UUID(int=len(chunks) + 1000)),
                            title=str(node.get("title") or ""),
                            source=source,
                            source_id=str(node.get("id") or node.get("source_id") or ""),
                        )
                    )
                else:
                    for value in node.values():
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(payload)
    return chunks


def corpus_pack() -> EvidencePack:
    return EvidencePack(
        question="What is due this week in Project Phoenix?", chunks=corpus_chunks()
    )
