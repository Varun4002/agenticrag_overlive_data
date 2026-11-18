"""Source payloads -> :class:`CanonicalDoc` -> ``core.documents``.

Two responsibilities live here because they are two halves of the same type:

1. **Normalization** — turn a vendor payload into the canonical document model
   (``docs/05-data-model.md``). ``CanonicalDoc`` is owned by this component; the
   other agents consume it through that model, never the raw vendor JSON.
2. **Persistence** — the ``core.documents`` upsert, written once against a small
   :class:`DocumentStore` port so the tricky decision logic (hash change
   detection, supersession, retirement) is testable without Postgres while the
   SQLAlchemy store implements the real thing.

Nothing here redacts PII: raw text is stored unredacted on purpose and scrubbed
at retrieval time (``docs/02-architecture.md``). Jira descriptions/comments are
ADF (Atlassian Document Format), so text extraction is part of normalization.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import AsyncIterator, Collection, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocType, Document, Source, SyncState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADF -> plain text
# ---------------------------------------------------------------------------
# Jira Cloud REST v3 returns rich text as ADF (a JSON tree), not markdown. We
# flatten it to plain text: retrieval and synthesis work on text, and keeping the
# tree would leak JSON noise into embeddings. Handles the node types that appear
# in real issue/comment bodies; unknown nodes render their children.
_TEXT_LINK_FMT = "{text} <{href}>"


def adf_to_text(node: Any) -> str:
    """Flatten an ADF node (or a plain string) to readable plain text."""
    return _collapse_blank_lines(_render(node))


def _render(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_render(child) for child in node)
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type")
    if not isinstance(node_type, str):
        return ""
    attrs = node.get("attrs")
    attrs = attrs if isinstance(attrs, dict) else {}
    children = _render(node.get("content"))

    if node_type == "text":
        text = str(node.get("text") or "")
        href = _link_target(node)
        return _TEXT_LINK_FMT.format(text=text, href=href) if href else text
    if node_type == "hardBreak":
        return "\n"
    if node_type == "mention":
        name = attrs.get("text") or attrs.get("displayName") or attrs.get("id") or ""
        return f"@{name}"
    if node_type == "emoji":
        return str(attrs.get("text") or "")
    if node_type in {"inlineCard", "blockCard"}:
        return str(attrs.get("url") or "")
    if node_type in {"media", "mediaSingle", "mediaGroup"}:
        return "[media]"
    if node_type == "codeBlock":
        return f"```\n{children.strip()}\n```\n"
    if node_type == "rule":
        return "---\n"
    if node_type == "listItem":
        return f"- {children.strip()}\n"
    if node_type in {"bulletList", "orderedList", "panel", "expand", "blockquote"}:
        return children if children.endswith("\n") else f"{children}\n"
    if node_type == "tableRow":
        return f"{children.rstrip()}\n"
    if node_type == "tableCell":
        return f"{children.strip()} | "
    if node_type == "heading":
        return f"{children.strip()}\n"
    if node_type in {"paragraph", "doc", "table", "tableHeader"}:
        return children if node_type == "doc" else f"{children.strip()}\n"
    return children


def _link_target(node: Mapping[str, Any]) -> str | None:
    marks = node.get("marks")
    if not isinstance(marks, list):
        return None
    for mark in marks:
        if isinstance(mark, dict) and mark.get("type") == "link":
            attrs = mark.get("attrs")
            if isinstance(attrs, dict) and isinstance(attrs.get("href"), str):
                return str(attrs["href"])
    return None


def _collapse_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Canonical document model
# ---------------------------------------------------------------------------
def compute_content_hash(title: str, content: str) -> str:
    """sha256 over the text a reader would see — the change-detection key.

    Covers title *and* content, so a status/assignee/due-date change (all of which
    are rendered into the issue content) also changes the hash.
    """
    digest = hashlib.sha256()
    digest.update(title.strip().encode("utf-8"))
    digest.update(b"\n\n")
    digest.update(content.strip().encode("utf-8"))
    return digest.hexdigest()


class Acl(BaseModel):
    """Mirrored source permissions, stored as ``core.documents.acl`` jsonb.

    Identity conventions (documented for the retrieval agent, which filters on
    these values):

    - ``users``: ``account:<source account id>`` and/or ``email:<lowercased>``
    - ``groups``: ``<source>:project:<KEY>`` (Jira), workspace id (Notion)
    - ``labels``: Gmail labels — unused by Jira

    Lists are sorted and de-duplicated so the jsonb value is deterministic (ACL
    drift detection compares it directly).
    """

    model_config = ConfigDict(extra="forbid")

    users: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)

    @field_validator("users", "groups", "labels")
    @classmethod
    def _canonical(cls, value: list[str]) -> list[str]:
        return sorted({item.strip() for item in value if item.strip()})

    def as_jsonb(self) -> dict[str, list[str]]:
        return {"users": list(self.users), "groups": list(self.groups), "labels": list(self.labels)}


class CanonicalDoc(BaseModel):
    """One canonical source item, 1:1 with a ``core.documents`` row."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    source_id: str = Field(min_length=1)
    parent_source_id: str | None = None
    doc_type: DocType
    title: str
    content: str
    url: str | None = None
    author: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    fetched_at: AwareDatetime
    acl: Acl = Field(default_factory=Acl)
    content_hash: str = ""
    is_current: bool = True

    @model_validator(mode="after")
    def _fill_content_hash(self) -> CanonicalDoc:
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.title, self.content)
        return self


# ---------------------------------------------------------------------------
# Jira normalization
# ---------------------------------------------------------------------------
def jira_issue_to_docs(
    issue: Mapping[str, Any],
    *,
    base_url: str,
    fetched_at: datetime,
    comments: Sequence[Mapping[str, Any]] = (),
) -> list[CanonicalDoc]:
    """Map one Jira issue (plus its comments) to canonical documents.

    Returns the issue document first, then one document per comment. Raises
    ``ValueError`` on a payload that cannot be normalized (the caller records an
    ``ErrorReport`` and moves on).
    """
    key = issue.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("issue payload has no 'key'")
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError(f"issue {key} has no 'fields' object")

    project = fields.get("project")
    project = project if isinstance(project, Mapping) else {}
    project_key = str(project.get("key") or key.split("-")[0])
    summary = _as_text(fields.get("summary")) or f"({key})"

    assignee = _person(fields.get("assignee"))
    reporter = _person(fields.get("reporter"))
    issue_acl = _jira_acl(project_key=project_key, people=[assignee, reporter])

    description = _as_text(fields.get("description"))
    content = _issue_content(
        key=key,
        fields=fields,
        project=project,
        summary=summary,
        assignee=assignee,
        reporter=reporter,
        description=description,
        url=_issue_url(base_url, key),
    )
    issue_doc = CanonicalDoc(
        source=Source.jira,
        source_id=key,
        parent_source_id=None,
        doc_type=DocType.issue,
        title=f"{key}: {summary}",
        content=content,
        url=_issue_url(base_url, key),
        author=reporter.display_name if reporter else None,
        created_at=_timestamp(fields.get("created"), fallback=fetched_at, what=f"{key} created"),
        updated_at=_timestamp(fields.get("updated"), fallback=fetched_at, what=f"{key} updated"),
        fetched_at=fetched_at,
        acl=issue_acl,
    )

    docs = [issue_doc]
    for comment in comments:
        docs.append(
            _jira_comment_doc(
                issue_key=key,
                summary=summary,
                comment=comment,
                base_url=base_url,
                fetched_at=fetched_at,
                issue_acl=issue_acl,
            )
        )
    return docs


def jira_comment_source_id(issue_key: str, comment_id: str) -> str:
    """Stable canonical id for a comment document."""
    return f"{issue_key}/comment/{comment_id}"


def _jira_comment_doc(
    *,
    issue_key: str,
    summary: str,
    comment: Mapping[str, Any],
    base_url: str,
    fetched_at: datetime,
    issue_acl: Acl,
) -> CanonicalDoc:
    comment_id = comment.get("id")
    if comment_id is None or not str(comment_id):
        raise ValueError(f"comment on {issue_key} has no 'id'")
    comment_id = str(comment_id)

    author = _person(comment.get("author"))
    body = _as_text(comment.get("body"))
    if not body:
        # An empty comment is legal in Jira ("attachment only"); keep a marker so
        # the document still exists and its hash is stable.
        body = "(empty comment)"

    url = f"{_issue_url(base_url, issue_key)}?focusedCommentId={comment_id}"
    content = _collapse_blank_lines(
        "\n".join(
            [
                f"Issue: {issue_key} — {summary}",
                f"Author: {_person_label(author)}",
                f"Created: {_iso_or_unknown(comment.get('created'))}",
                f"Updated: {_iso_or_unknown(comment.get('updated'))}",
                f"URL: {url}",
                "",
                body,
            ]
        )
    )
    comment_acl = Acl(
        users=[*issue_acl.users, *_person_identifiers(author)],
        groups=issue_acl.groups,
        labels=issue_acl.labels,
    )
    return CanonicalDoc(
        source=Source.jira,
        source_id=jira_comment_source_id(issue_key, comment_id),
        parent_source_id=issue_key,
        doc_type=DocType.issue_comment,
        title=f"{issue_key}: comment by {author.display_name if author else 'unknown'}",
        content=content,
        url=url,
        author=author.display_name if author else None,
        created_at=_timestamp(comment.get("created"), fallback=fetched_at, what="comment created"),
        updated_at=_timestamp(comment.get("updated"), fallback=fetched_at, what="comment updated"),
        fetched_at=fetched_at,
        acl=comment_acl,
    )


def _issue_content(
    *,
    key: str,
    fields: Mapping[str, Any],
    project: Mapping[str, Any],
    summary: str,
    assignee: Person | None,
    reporter: Person | None,
    description: str,
    url: str,
) -> str:
    """Render the issue as text.

    Every field the brief names (status, assignee, reporter, due date, priority,
    labels) is rendered as a ``Label: value`` line, which is what makes questions
    like "what is due this week" answerable from the document text alone. Lines
    with no value are omitted so a missing due date stays visibly missing.
    """
    project_name = str(project.get("name") or "")
    project_key = str(project.get("key") or key.split("-")[0])
    lines = [
        f"Summary: {summary}",
        f"Project: {project_name} ({project_key})" if project_name else f"Project: {project_key}",
    ]
    issue_type = _nested_name(fields.get("issuetype"))
    status = _nested_name(fields.get("status"))
    priority = _nested_name(fields.get("priority"))
    resolution = _nested_name(fields.get("resolution"))
    labels = fields.get("labels")
    labels_text = ", ".join(str(label) for label in labels) if isinstance(labels, list) else ""

    for label, value in (
        ("Type", issue_type),
        ("Status", status),
        ("Priority", priority),
        ("Resolution", resolution),
        ("Assignee", _person_label(assignee)),
        ("Reporter", _person_label(reporter)),
        ("Due", _as_text(fields.get("duedate"))),
        ("Labels", labels_text),
        ("Created", _iso_or_unknown(fields.get("created"))),
        ("Updated", _iso_or_unknown(fields.get("updated"))),
        ("URL", url),
    ):
        if value:
            lines.append(f"{label}: {value}")

    lines.append("")
    lines.append("Description:")
    lines.append(description or "(no description)")
    return _collapse_blank_lines("\n".join(lines))


@dataclass(frozen=True, slots=True)
class Person:
    """A source-visible person, kept whole because PII scrubbing happens later."""

    display_name: str | None = None
    email: str | None = None
    account_id: str | None = None


def _person(raw: Any) -> Person | None:
    if not isinstance(raw, Mapping):
        return None
    display_name = raw.get("displayName")
    email = raw.get("emailAddress")
    account_id = raw.get("accountId")
    person = Person(
        display_name=str(display_name) if display_name else None,
        email=str(email) if email else None,
        account_id=str(account_id) if account_id else None,
    )
    if person == Person():
        return None
    return person


def _person_identifiers(person: Person | None) -> list[str]:
    if person is None:
        return []
    identifiers = []
    if person.account_id:
        identifiers.append(f"account:{person.account_id}")
    if person.email:
        identifiers.append(f"email:{person.email.lower()}")
    return identifiers


def _person_label(person: Person | None) -> str:
    """Human-readable person rendering, e.g. ``Ada Lovelace <ada@example.com>``."""
    if person is None:
        return ""
    if person.display_name and person.email:
        return f"{person.display_name} <{person.email}>"
    return person.display_name or person.email or person.account_id or ""


def _jira_acl(*, project_key: str, people: Iterable[Person | None]) -> Acl:
    users: list[str] = []
    for person in people:
        users.extend(_person_identifiers(person))
    return Acl(users=users, groups=[f"jira:project:{project_key}"], labels=[])


def _nested_name(raw: Any) -> str:
    if isinstance(raw, Mapping):
        name = raw.get("name") or raw.get("value")
        return str(name) if name else ""
    return ""


def _as_text(raw: Any) -> str:
    """ADF object, plain string, or nothing -> text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return _collapse_blank_lines(raw)
    return adf_to_text(raw)


def _iso_or_unknown(raw: Any) -> str:
    return str(raw) if raw else ""


def _issue_url(base_url: str, key: str) -> str:
    return f"{base_url.rstrip('/')}/browse/{key}"


def _timestamp(raw: Any, *, fallback: datetime, what: str) -> datetime:
    """Parse a Jira timestamp, falling back to the fetch time.

    Falling back (instead of failing) matters because ``created``/``updated`` are
    not always granted by Jira field-level security; the item is still usable.
    """
    if isinstance(raw, str) and raw:
        try:
            return parse_jira_timestamp(raw)
        except ValueError:
            logger.warning("unparseable Jira timestamp for %s (%r); using fetch time", what, raw)
    return fallback


def parse_jira_timestamp(raw: str) -> datetime:
    """Parse Jira's ``2026-11-16T09:12:00.000+0000`` shape into an aware datetime."""
    text = raw.strip()
    # Python 3.11+ understands "+0000"; normalise just in case a proxy strips it.
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        normalised = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
        parsed = datetime.fromisoformat(normalised.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-qualified: {raw!r}")
    return parsed


# ---------------------------------------------------------------------------
# Persistence port
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Minimal snapshot of an existing ``core.documents`` row."""

    id: uuid.UUID
    source_id: str
    content_hash: str
    acl: dict[str, Any]
    is_current: bool


@dataclass(frozen=True, slots=True)
class SyncStateRecord:
    """Snapshot of a ``core.sync_state`` row."""

    source: str
    watermark: datetime | None = None
    last_run_at: datetime | None = None
    last_status: str | None = None
    error: str | None = None


class UpsertOutcome(StrEnum):
    """What an upsert did to the store."""

    created = "created"
    updated = "updated"
    unchanged = "unchanged"


@dataclass(frozen=True, slots=True)
class UpsertResult:
    outcome: UpsertOutcome
    document_id: uuid.UUID
    superseded_id: uuid.UUID | None = None


class DocumentStore(Protocol):
    """The persistence primitives the sync logic needs.

    Deliberately narrow and dumb: all *decisions* live in
    :func:`apply_document` / :func:`retire_missing`, so they are exercised once,
    against both a real SQL store and an in-memory one in tests.
    """

    async def find_current(self, source: Source, source_id: str) -> DocumentRecord | None: ...

    async def insert(self, doc: CanonicalDoc) -> uuid.UUID: ...

    async def retire(self, document_id: uuid.UUID) -> None:
        """Mark a document non-current, keeping its native ``source_id``.

        History rows share the native id with the current row; the partial unique
        index ``uq_documents_current_source_id`` only constrains current rows
        (migration 0003), so any number of retired revisions is allowed.
        """
        ...

    async def list_current_root_ids(self, source: Source) -> list[str]: ...

    async def list_current_child_ids(self, source: Source, parent_source_id: str) -> list[str]: ...

    async def load_sync_state(self, source: str) -> SyncStateRecord: ...

    async def save_sync_state(self, state: SyncStateRecord) -> None: ...

    def unit_of_work(self) -> AbstractAsyncContextManager[None]:
        """A savepoint around one item, so one failure cannot poison the run."""
        ...


async def apply_document(store: DocumentStore, doc: CanonicalDoc) -> UpsertResult:
    """Insert/update one canonical document, skipping unchanged content.

    Policy (``docs/agents/01-connector-agent.md``): unchanged ``content_hash`` is
    skipped; otherwise the outgoing row is kept for audit with
    ``is_current=false`` and the new content becomes the current row. Nothing is
    ever hard-deleted.
    """
    current = await store.find_current(doc.source, doc.source_id)
    if current is None:
        return UpsertResult(UpsertOutcome.created, await store.insert(doc))

    unchanged = current.content_hash == doc.content_hash and current.acl == doc.acl.as_jsonb()
    if unchanged:
        return UpsertResult(UpsertOutcome.unchanged, current.id)

    # The outgoing row keeps its native source_id: the partial unique index only
    # covers current rows, so history accumulates without collisions.
    await store.retire(current.id)
    new_id = await store.insert(doc)
    return UpsertResult(UpsertOutcome.updated, new_id, superseded_id=current.id)


async def retire_missing(
    store: DocumentStore,
    *,
    source: Source,
    seen: Collection[str],
    parent_source_id: str | None = None,
) -> list[str]:
    """Mark documents that vanished from the source ``is_current=false``.

    With ``parent_source_id`` set, only that parent's children (comments) are
    considered — used to catch a comment deleted from an issue we just synced.
    Without it, all *root* documents of the source are swept (full syncs only);
    retiring a root also retires its children.

    Returns the ``source_id`` values that were actually retired (roots and their
    children).

    An empty ``seen`` set is only honoured for a children sweep (an issue whose
    last comment was deleted); for a whole-source sweep it retires nothing,
    because a query that returns zero items is far more likely to be a bug than a
    source that lost every document at once.
    """
    if parent_source_id is None and not seen:
        logger.warning("%s: refusing to retire the whole scope on an empty seen-set", source.value)
        return []

    if parent_source_id is None:
        candidates = await store.list_current_root_ids(source)
    else:
        candidates = await store.list_current_child_ids(source, parent_source_id)

    retired: list[str] = []
    for source_id in sorted(set(candidates) - set(seen)):
        if await _retire_document(store, source, source_id):
            retired.append(source_id)
        if parent_source_id is None:
            # A deleted issue takes its comments with it.
            for child_id in await store.list_current_child_ids(source, source_id):
                if await _retire_document(store, source, child_id):
                    retired.append(child_id)
    if retired:
        logger.info(
            "%s: retired %d document(s) no longer present in source", source.value, len(retired)
        )
    return retired


async def _retire_document(store: DocumentStore, source: Source, source_id: str) -> bool:
    """Retire one document. Returns False when it was already retired/absent."""
    record = await store.find_current(source, source_id)
    if record is None:
        return False
    await store.retire(record.id)
    return True


# ---------------------------------------------------------------------------
# SQLAlchemy store
# ---------------------------------------------------------------------------
class SqlAlchemyDocumentStore:
    """``DocumentStore`` over ``core.documents`` / ``core.sync_state``.

    Owns no transaction: the caller commits. Each write is expected to be wrapped
    in :meth:`unit_of_work` (a SAVEPOINT) by the sync loop so one bad item rolls
    back alone.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def unit_of_work(self) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def _savepoint() -> AsyncIterator[None]:
            # SAVEPOINT: a failing item rolls back alone, the run's transaction
            # (and every earlier item) survives.
            async with self._session.begin_nested():
                yield None

        return _savepoint()

    async def find_current(self, source: Source, source_id: str) -> DocumentRecord | None:
        row = (
            await self._session.execute(
                select(Document).where(
                    Document.source == source,
                    Document.source_id == source_id,
                    Document.is_current.is_(True),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return DocumentRecord(
            id=row.id,
            source_id=row.source_id,
            content_hash=row.content_hash,
            acl=dict(row.acl or {}),
            is_current=row.is_current,
        )

    async def insert(self, doc: CanonicalDoc) -> uuid.UUID:
        row = Document(
            source=doc.source,
            source_id=doc.source_id,
            parent_source_id=doc.parent_source_id,
            doc_type=doc.doc_type,
            title=doc.title,
            content=doc.content,
            url=doc.url,
            author=doc.author,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            fetched_at=doc.fetched_at,
            acl=doc.acl.as_jsonb(),
            content_hash=doc.content_hash,
            is_current=True,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def retire(self, document_id: uuid.UUID) -> None:
        """Flip ``is_current`` to false; the row keeps its native ``source_id``.

        Only current rows are uniquely constrained (partial index), so retired
        revisions can pile up under the same id — nothing is renamed or deleted.
        """
        await self._session.execute(
            update(Document)
            # is_current guard: retiring twice is a no-op.
            .where(Document.id == document_id, Document.is_current.is_(True))
            .values(is_current=False)
        )
        await self._session.flush()

    async def list_current_root_ids(self, source: Source) -> list[str]:
        rows = await self._session.execute(
            select(Document.source_id).where(
                Document.source == source,
                Document.is_current.is_(True),
                Document.parent_source_id.is_(None),
            )
        )
        return list(rows.scalars())

    async def list_current_child_ids(self, source: Source, parent_source_id: str) -> list[str]:
        rows = await self._session.execute(
            select(Document.source_id).where(
                Document.source == source,
                Document.parent_source_id == parent_source_id,
                Document.is_current.is_(True),
            )
        )
        return list(rows.scalars())

    async def load_sync_state(self, source: str) -> SyncStateRecord:
        row = (
            await self._session.execute(select(SyncState).where(SyncState.source == source))
        ).scalar_one_or_none()
        if row is None:
            return SyncStateRecord(source=source)
        return SyncStateRecord(
            source=row.source,
            watermark=row.watermark,
            last_run_at=row.last_run_at,
            last_status=row.last_status,
            error=row.error,
        )

    async def save_sync_state(self, state: SyncStateRecord) -> None:
        statement = pg_insert(SyncState).values(
            source=state.source,
            watermark=state.watermark,
            last_run_at=state.last_run_at,
            last_status=state.last_status,
            error=state.error,
        )
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[SyncState.source],
                set_={
                    "watermark": statement.excluded.watermark,
                    "last_run_at": statement.excluded.last_run_at,
                    "last_status": statement.excluded.last_status,
                    "error": statement.excluded.error,
                },
            )
        )
        await self._session.flush()


class InMemoryDocumentStore:
    """Reference ``DocumentStore`` for unit tests — same semantics, no SQL.

    Provided in the package (not tests) so the semantics have one executable
    definition; ``tests/connectors`` uses it for the sync-logic suite and a
    Postgres-backed test proves the SQL store behaves the same.
    """

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, CanonicalDoc] = {}
        self.current: dict[tuple[Source, str], uuid.UUID] = {}
        self.sync_state: dict[str, SyncStateRecord] = {}

    def unit_of_work(self) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def _noop() -> AsyncIterator[None]:
            yield

        return _noop()

    async def find_current(self, source: Source, source_id: str) -> DocumentRecord | None:
        row_id = self.current.get((source, source_id))
        if row_id is None:
            return None
        doc = self.rows[row_id]
        return DocumentRecord(
            id=row_id,
            source_id=doc.source_id,
            content_hash=doc.content_hash,
            acl=doc.acl.as_jsonb(),
            is_current=True,
        )

    async def insert(self, doc: CanonicalDoc) -> uuid.UUID:
        row_id = uuid.uuid4()
        self.rows[row_id] = doc.model_copy(update={"is_current": True})
        self.current[(doc.source, doc.source_id)] = row_id
        return row_id

    async def retire(self, document_id: uuid.UUID) -> None:
        doc = self.rows[document_id]
        if not doc.is_current:  # retiring twice is a no-op
            return
        self.current.pop((doc.source, doc.source_id), None)
        self.rows[document_id] = doc.model_copy(update={"is_current": False})

    async def list_current_root_ids(self, source: Source) -> list[str]:
        return [
            doc.source_id
            for doc in self.rows.values()
            if doc.source == source and doc.is_current and doc.parent_source_id is None
        ]

    async def list_current_child_ids(self, source: Source, parent_source_id: str) -> list[str]:
        return [
            doc.source_id
            for doc in self.rows.values()
            if doc.source == source and doc.is_current and doc.parent_source_id == parent_source_id
        ]

    async def load_sync_state(self, source: str) -> SyncStateRecord:
        return self.sync_state.get(source, SyncStateRecord(source=source))

    async def save_sync_state(self, state: SyncStateRecord) -> None:
        self.sync_state[state.source] = state


__all__ = [
    "Acl",
    "CanonicalDoc",
    "DocumentRecord",
    "DocumentStore",
    "InMemoryDocumentStore",
    "SqlAlchemyDocumentStore",
    "SyncStateRecord",
    "UpsertOutcome",
    "UpsertResult",
    "adf_to_text",
    "apply_document",
    "compute_content_hash",
    "jira_comment_source_id",
    "jira_issue_to_docs",
    "parse_jira_timestamp",
    "retire_missing",
]
