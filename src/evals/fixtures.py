"""Synthetic fixture corpus — loader and DB seeder.

The corpus is the shared substrate for the indexer, retrieval, PII, evidence,
guardrail, and eval agents: everything must be buildable and testable without
live Jira/Notion/Gmail credentials (``docs/05-data-model.md`` §"Migration & seed
strategy").

Layout (``fixtures/corpus/``)::

    manifest.json        projects, clock, PII index, injection index,
                         contradictions, false-positive names, overdue list
    jira_phoenix.json    Project Phoenix issues + comments
    jira_atlas.json      Project Atlas issues + comments (deliberately unrelated)
    notion_pages.json    goal/status/vendor/directory/migration pages + sections
    gmail_threads.json   email threads, quoted history kept on purpose

Document shape (column-for-column with ``core.documents``)::

    source, source_id, parent_source_id, doc_type, title, content,
    url, author, created_at, updated_at, acl, labels

``content_hash`` is *not* in the files: the loader computes ``sha256(content)``
so the files cannot drift from the hash. ``is_current`` and ``fetched_at`` are
assigned at seed time.

Determinism: fixed dates (no relative dates), no randomness, canonical
``(source, source_id)`` ordering on load, and deterministic document ids at seed
time (``uuid5`` of source + source_id), so re-seeding is idempotent and chunk
ids stay stable across runs.

The corpus is fictional throughout: invented people at ``*.example`` domains,
555-range phone numbers, and an obviously fake API key.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.models import Document as DocumentRow
from indexing.tokens import sha256_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "corpus"
MANIFEST_NAME = "manifest.json"

#: Namespace for deterministic document ids: same fixture → same row id.
DOCUMENT_ID_NAMESPACE = uuid.UUID("0b7d3f61-2c4a-4e58-9f13-7a5d8e2c6b90")

Source = Literal["jira", "notion", "gmail"]
DocType = Literal["issue", "issue_comment", "page", "page_section", "email"]

#: doc_type is bound to its source: a Jira row can never be a page.
ALLOWED_DOC_TYPES: Mapping[str, frozenset[str]] = {
    "jira": frozenset({"issue", "issue_comment"}),
    "notion": frozenset({"page", "page_section"}),
    "gmail": frozenset({"email"}),
}


class FixtureCorpusError(ValueError):
    """The corpus on disk is missing, malformed, or internally inconsistent."""


class FixtureDocument(BaseModel):
    """One canonical document from the corpus (schema-validated)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Source
    source_id: str
    parent_source_id: str | None = None
    doc_type: DocType
    title: str
    content: str
    url: str | None = None
    author: str | None = None
    created_at: datetime
    updated_at: datetime
    acl: dict[str, list[str]] = {}
    labels: list[str] = []
    #: sha256 of ``content``, filled by the loader.
    content_hash: str = ""

    @model_validator(mode="after")
    def _validate(self) -> FixtureDocument:
        if self.doc_type not in ALLOWED_DOC_TYPES[self.source]:
            raise ValueError(
                f"{self.source}/{self.source_id}: doc_type {self.doc_type!r} is not "
                f"valid for source {self.source!r}"
            )
        if self.doc_type in {"issue_comment", "page_section"} and not self.parent_source_id:
            raise ValueError(
                f"{self.source}/{self.source_id}: {self.doc_type} requires parent_source_id"
            )
        if self.doc_type == "issue" and self.parent_source_id:
            raise ValueError(f"{self.source}/{self.source_id}: issue cannot have a parent")
        for name in ("created_at", "updated_at"):
            value: datetime = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"{self.source}/{self.source_id}: {name} must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError(f"{self.source}/{self.source_id}: updated_at precedes created_at")
        if not self.content.strip():
            raise ValueError(f"{self.source}/{self.source_id}: content is empty")
        return self.model_copy(update={"content_hash": sha256_text(self.content)})

    @property
    def key(self) -> str:
        """Stable identity for error messages and lookups: ``source/source_id``."""
        return f"{self.source}/{self.source_id}"


@dataclass(frozen=True, slots=True)
class FixtureCorpus:
    """A loaded, validated corpus plus its manifest."""

    documents: tuple[FixtureDocument, ...]
    manifest: dict[str, Any]
    path: Path

    def by_source(self, source: str) -> tuple[FixtureDocument, ...]:
        return tuple(doc for doc in self.documents if doc.source == source)

    def by_doc_type(self, doc_type: str) -> tuple[FixtureDocument, ...]:
        return tuple(doc for doc in self.documents if doc.doc_type == doc_type)

    def get(self, source: str, source_id: str) -> FixtureDocument:
        for doc in self.documents:
            if doc.source == source and doc.source_id == source_id:
                return doc
        raise FixtureCorpusError(f"no fixture document {source}/{source_id}")

    def pii_documents(self) -> tuple[FixtureDocument, ...]:
        """Documents the manifest marks as containing deliberate PII."""
        planted = self.manifest.get("pii", {}).get("planted_in", [])
        return tuple(
            self.get(item["source"], item["source_id"])
            for item in planted
            if self._has(item["source"], item["source_id"])
        )

    def injection_documents(self) -> tuple[FixtureDocument, ...]:
        """Documents carrying a planted prompt-injection payload."""
        planted = self.manifest.get("injection", {}).get("documents", [])
        return tuple(
            self.get(item["source"], item["source_id"])
            for item in planted
            if self._has(item["source"], item["source_id"])
        )

    def _has(self, source: str, source_id: str) -> bool:
        return any(doc.source == source and doc.source_id == source_id for doc in self.documents)

    @property
    def counts(self) -> dict[str, int]:
        """``{source:doc_type: count}`` — handy for assertions and reports."""
        counts: dict[str, int] = {}
        for doc in self.documents:
            key = f"{doc.source}:{doc.doc_type}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


def document_id_for(doc: FixtureDocument) -> uuid.UUID:
    """Deterministic row id so seeding twice never creates duplicates."""
    return uuid.uuid5(DOCUMENT_ID_NAMESPACE, f"{doc.source}:{doc.source_id}")


def load_fixture_corpus(path: Path | None = None) -> FixtureCorpus:
    """Load and validate every ``*.json`` corpus file.

    Documents come back in canonical ``(source, source_id, doc_type)`` order so
    callers see the same sequence on every machine. Raises
    :class:`FixtureCorpusError` when a file is missing, unreadable, has duplicate
    ids, or when the manifest references a document that does not exist.
    """
    root = path or FIXTURE_DIR
    if not root.is_dir():
        raise FixtureCorpusError(f"fixture corpus directory not found: {root}")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FixtureCorpusError(f"fixture manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureCorpusError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FixtureCorpusError(f"{manifest_path}: manifest must be a JSON object")

    documents: list[FixtureDocument] = []
    seen: dict[str, str] = {}
    for file_path in sorted(root.glob("*.json")):
        if file_path.name == MANIFEST_NAME:
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FixtureCorpusError(f"{file_path}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "documents" not in payload:
            raise FixtureCorpusError(f"{file_path}: expected an object with a 'documents' list")
        raw_documents = payload["documents"]
        if not isinstance(raw_documents, list):
            raise FixtureCorpusError(f"{file_path}: 'documents' must be a list")
        for index, raw in enumerate(raw_documents):
            try:
                document = FixtureDocument.model_validate(raw)
            except Exception as exc:  # pydantic.ValidationError, no need to leak the class
                raise FixtureCorpusError(
                    f"{file_path.name}[{index}]: invalid document: {exc}"
                ) from exc
            if document.key in seen:
                raise FixtureCorpusError(
                    f"{file_path.name}[{index}]: duplicate document {document.key} "
                    f"(already in {seen[document.key]})"
                )
            seen[document.key] = file_path.name
            documents.append(document)

    documents.sort(key=lambda doc: (doc.source, doc.source_id, doc.doc_type))
    corpus = FixtureCorpus(documents=tuple(documents), manifest=manifest, path=root)
    _validate_manifest(corpus)
    return corpus


def _validate_manifest(corpus: FixtureCorpus) -> None:
    """Every manifest reference must resolve; stale indexes are a corpus bug."""
    references: list[Mapping[str, Any]] = []
    references.extend(corpus.manifest.get("pii", {}).get("planted_in", []))
    references.extend(corpus.manifest.get("injection", {}).get("documents", []))
    references.extend(corpus.manifest.get("overdue", []))
    references.extend(corpus.manifest.get("due_this_week", []))
    for item in corpus.manifest.get("contradictions", {}).get("items", []):
        references.extend(item.get("sources", []))
    for item in corpus.manifest.get("false_positive_names", {}).get("items", []):
        references.extend(item.get("sources", []))

    missing = sorted(
        {
            f"{ref.get('source')}/{ref.get('source_id')}"
            for ref in references
            if not corpus._has(str(ref.get("source")), str(ref.get("source_id")))
        }
    )
    if missing:
        raise FixtureCorpusError(f"manifest references missing documents: {missing}")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeedResult:
    """What one seeding run wrote."""

    documents_seeded: int
    ids: tuple[uuid.UUID, ...]


async def seed_fixture_corpus(
    session_factory: async_sessionmaker[AsyncSession],
    corpus: FixtureCorpus | None = None,
    *,
    now: datetime | None = None,
) -> SeedResult:
    """Upsert the corpus into ``core.documents`` (idempotent).

    Documents get deterministic ids and ``content_hash`` values, so a second
    seed updates the same rows and leaves chunks attached. ``fetched_at`` is the
    only non-deterministic value (defaults to now).
    """
    resolved = corpus or load_fixture_corpus()
    fetched_at = now or datetime.now(UTC)
    rows = [
        {
            "id": document_id_for(doc),
            "source": doc.source,
            "source_id": doc.source_id,
            "parent_source_id": doc.parent_source_id,
            "doc_type": doc.doc_type,
            "title": doc.title,
            "content": doc.content,
            "url": doc.url,
            "author": doc.author,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "fetched_at": fetched_at,
            "acl": doc.acl,
            "content_hash": doc.content_hash,
            "is_current": True,
        }
        for doc in resolved.documents
    ]
    if not rows:
        return SeedResult(documents_seeded=0, ids=())

    statement = pg_insert(DocumentRow).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "parent_source_id": statement.excluded.parent_source_id,
            "doc_type": statement.excluded.doc_type,
            "title": statement.excluded.title,
            "content": statement.excluded.content,
            "url": statement.excluded.url,
            "author": statement.excluded.author,
            "created_at": statement.excluded.created_at,
            "updated_at": statement.excluded.updated_at,
            "fetched_at": statement.excluded.fetched_at,
            "acl": statement.excluded.acl,
            "content_hash": statement.excluded.content_hash,
            "is_current": True,
        },
    )
    async with session_factory() as session, session.begin():
        await session.execute(statement)
    return SeedResult(
        documents_seeded=len(rows), ids=tuple(document_id_for(doc) for doc in resolved.documents)
    )


def fixture_document_ids(corpus: FixtureCorpus | None = None) -> Sequence[uuid.UUID]:
    """Ids the corpus seeds, in canonical order (useful for targeted indexing)."""
    return [document_id_for(doc) for doc in (corpus or load_fixture_corpus()).documents]


def iter_documents(corpus: FixtureCorpus) -> Iterable[FixtureDocument]:
    """Typed iterator (keeps callers off ``corpus.documents`` internals)."""
    yield from corpus.documents


__all__ = [
    "ALLOWED_DOC_TYPES",
    "FIXTURE_DIR",
    "FixtureCorpus",
    "FixtureCorpusError",
    "FixtureDocument",
    "SeedResult",
    "document_id_for",
    "fixture_document_ids",
    "iter_documents",
    "load_fixture_corpus",
    "seed_fixture_corpus",
]
