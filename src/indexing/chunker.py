"""Deterministic, structure-aware chunking.

Turns one canonical document (``core.documents`` row) into retrievable
``core.chunks`` records. Pure functions + dataclasses: no database, no network,
no clock, so every boundary is reproducible and golden-testable.

Design rules (from ``docs/agents/02-indexer-agent.md``):

1. **Structure before size.** Jira field blocks, Notion headings, and an email
   body are separate structural units. Section boundaries are never merged,
   even when a section is shorter than the target size — a 40-token Notion
   section stays its own chunk because merging it would hide the heading it
   belongs to.
2. **Size within a unit.** Long text is packed into 300-500 token chunks with
   ~15% sentence/paragraph overlap, so a fact split across a boundary is still
   retrievable from both sides.
3. **Self-describing text.** Each chunk is prefixed with a one-line context
   breadcrumb (issue key/title + heading path). A short comment like "agreed,
   I'll pick it up" is useless to an embedding unless its chunk says *what*
   it agrees with. The breadcrumb is also what makes issue keys findable by
   full-text search.
4. **Quoted email text is dropped.** Gmail replies carry the entire thread
   below the new text; indexing that duplicates other documents and inflates
   the corpus.
5. **``content_hash`` per chunk** is ``sha256(text)``, so identical chunk text
   maps to one embedding-cache entry and unchanged chunks are detectable.

Token counts come from :func:`indexing.tokens.estimate_tokens` — deterministic
but approximate (see that module).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from indexing.tokens import estimate_tokens, sha256_text

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

#: Sentence boundary: terminal punctuation followed by whitespace and a capital
#: letter, digit, or opening quote/bracket. Deliberately conservative — it must
#: not split "v2.1" or "3.5x".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n+")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_JIRA_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/-]{0,30}):[ \t]*(.*)$")
_WHITESPACE_RE = re.compile(r"[ \t]+")

# Email quoting / signature markers. Any of these starts the "everything below
# is a duplicate of an earlier message" region.
_QUOTE_MARKERS = (
    re.compile(r"^-{2,}[ \t]*original message[ \t]*-{2,}$", re.IGNORECASE),
    re.compile(r"^-{5,}[ \t]*forwarded message[ \t]*-{5,}$", re.IGNORECASE),
    re.compile(r"^on\b.{0,200}\bwrote:[ \t]*$", re.IGNORECASE),
    re.compile(r"^>{1,} ?"),
    re.compile(r"^_{10,}$"),
)
_HEADER_START_RE = re.compile(r"^(from|sent|to|cc|bcc|date|subject):[ \t]", re.IGNORECASE)
_HEADER_SIBLING_RE = re.compile(r"^(sent|to|cc|bcc|date|subject):[ \t]", re.IGNORECASE)
_SIGNATURE_MARKERS = (
    re.compile(r"^--[ \t]?$"),
    re.compile(r"^sent from my \w+", re.IGNORECASE),
)

#: Field labels that make a Jira issue card worth splitting off as its own
#: chunk. Any label matching the generic ``Key: value`` shape is accepted, but
#: at least one of these must appear — otherwise prose containing colons (e.g.
#: "Note: x") would be misread as a field block.
JIRA_CARD_FIELDS = frozenset(
    {
        "key",
        "summary",
        "type",
        "issue type",
        "status",
        "priority",
        "assignee",
        "reporter",
        "due date",
        "created",
        "updated",
        "labels",
        "components",
        "project",
        "sprint",
        "story points",
        "resolution",
        "fix version",
        "epic link",
        "parent",
    }
)

REQUIRED_META_KEYS = (
    "source",
    "source_id",
    "doc_type",
    "title",
    "heading_path",
    "section",
    "part",
    "labels",
    "token_count",
    "content_hash",
)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """The slice of a ``core.documents`` row the chunker needs.

    Deliberately not an ORM model: the chunker must be usable (and testable)
    without a database. ``indexing.indexer`` builds this from the row.
    """

    source: str
    source_id: str
    doc_type: str
    title: str
    content: str
    document_id: str | None = None
    parent_source_id: str | None = None
    author: str | None = None
    url: str | None = None
    labels: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable slice, ready for ``core.chunks``."""

    seq: int
    text: str
    content_hash: str
    token_count: int
    meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Chunker:
    """Chunk sizing policy. Deterministic: no randomness, no time, no env."""

    #: Hard ceiling for one chunk (tokens, breadcrumb included).
    max_tokens: int = 500
    #: Target floor; a short trailing fragment is merged back when it fits.
    min_tokens: int = 300
    #: Share of a chunk repeated at the start of the next one.
    overlap_ratio: float = 0.15
    #: Breadcrumbs are truncated past this share of the budget so a pathological
    #: title cannot eat the whole chunk.
    max_context_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_tokens < 50:
            raise ValueError("max_tokens must be >= 50")
        if not 0 <= self.overlap_ratio < 0.5:
            raise ValueError("overlap_ratio must be in [0, 0.5)")

    @property
    def overlap_tokens(self) -> int:
        return round(self.max_tokens * self.overlap_ratio)

    def chunk(
        self,
        doc: SourceDocument,
        *,
        entity_ids: Sequence[str] = (),
    ) -> list[Chunk]:
        """Chunk one document. Returns ``[]`` for an empty document."""
        body = _normalize(doc.content)
        if not body.strip():
            return []

        sections = _sections_for(doc, body)
        chunks: list[Chunk] = []
        seq = 0
        for section in sections:
            prefix = self._context_line(doc, section.heading_path)
            for part, (body_text, token_count) in enumerate(self._pack_section(prefix, section)):
                text = f"{prefix}\n\n{body_text}" if prefix else body_text
                content_hash = sha256_text(text)
                meta = _meta_for(
                    doc, section, part=part, token_count=token_count, entity_ids=entity_ids
                )
                meta["content_hash"] = content_hash
                chunks.append(
                    Chunk(
                        seq=seq,
                        text=text,
                        content_hash=content_hash,
                        token_count=token_count,
                        meta=meta,
                    )
                )
                seq += 1
        return chunks

    # -- breadcrumb + packing --------------------------------------------
    def _context_line(self, doc: SourceDocument, heading_path: tuple[str, ...]) -> str:
        """Breadcrumb for one section, capped at a share of the token budget.

        A pathological title (or a deeply nested heading path) must not eat the
        whole chunk, so the line is truncated to ``max_context_ratio`` of
        ``max_tokens``. The truncated line is what goes into the chunk text, so
        token counts and the token budget stay truthful.
        """
        line = _context_line(doc, heading_path)
        cap = max(8, round(self.max_tokens * self.max_context_ratio))
        if estimate_tokens(line) > cap:
            line = _truncate_words(line, cap)
        return line

    def _pack_section(self, prefix: str, section: _Section) -> list[tuple[str, int]]:
        """Split one structural section into ``(text, token_count)`` pairs."""
        prefix_tokens = estimate_tokens(prefix) + 2 if prefix else 0
        budget = self.max_tokens - prefix_tokens

        units = _units(section.body, budget)
        bodies = _pack(units, budget=budget, overlap=max(0, min(self.overlap_tokens, budget // 2)))
        bodies = _merge_short_tail(bodies, min_tokens=self.min_tokens, budget=budget)

        out: list[tuple[str, int]] = []
        for body_text in bodies:
            if not body_text.strip():
                continue
            total = estimate_tokens(body_text) + prefix_tokens
            out.append((body_text, total))
        return out


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

DEFAULT_CHUNKER = Chunker()


def chunk_document(
    doc: SourceDocument,
    *,
    chunker: Chunker | None = None,
    entity_ids: Sequence[str] = (),
) -> list[Chunk]:
    """Chunk ``doc`` with the default (or supplied) policy."""
    return (chunker or DEFAULT_CHUNKER).chunk(doc, entity_ids=entity_ids)


# ---------------------------------------------------------------------------
# Structure extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Section:
    """One structural unit of a document, before size-based splitting."""

    body: str
    heading_path: tuple[str, ...] = ()
    section: str = "body"
    fields: tuple[tuple[str, str], ...] = ()
    labels: tuple[str, ...] = ()


def _sections_for(doc: SourceDocument, body: str) -> list[_Section]:
    """Dispatch on ``doc_type``; fall back to generic paragraphs."""
    doc_type = doc.doc_type
    if doc_type == "issue":
        return _jira_issue_sections(body)
    if doc_type == "issue_comment":
        return [_Section(body=body, section="comment")]
    if doc_type == "email":
        return _email_sections(body)
    if doc_type in {"page", "page_section"}:
        return _notion_sections(body)
    return [_Section(body=body)]


def _jira_issue_sections(body: str) -> list[_Section]:
    """Split a Jira issue into its field card and its description.

    A field block is a run of leading ``Label: value`` lines separated from the
    description by a blank line, containing at least one
    :data:`JIRA_CARD_FIELDS` label. Anything else is treated as description —
    the connector's normalizer is free to emit a different layout without
    breaking the indexer.
    """
    lines = body.split("\n")
    fields: list[tuple[str, str]] = []
    consumed = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            consumed = index + 1
            break
        match = _JIRA_FIELD_RE.match(stripped)
        if match is None:
            consumed = 0
            break
        fields.append((match.group(1).strip(), match.group(2).strip()))
        consumed = index + 1
    else:
        consumed = len(lines)

    recognized = {name.lower() for name, _ in fields} & JIRA_CARD_FIELDS
    if len(fields) < 2 or not recognized:
        return [_Section(body=body, section="description")]

    field_map = {name.lower(): value for name, value in fields}
    labels = tuple(
        part.strip()
        for part in field_map.get("labels", "").replace(";", ",").split(",")
        if part.strip()
    )
    description = "\n".join(lines[consumed:]).strip()
    sections = [
        _Section(
            body="\n".join(f"{name}: {value}" for name, value in fields if value),
            section="fields",
            fields=tuple(fields),
            labels=labels,
        )
    ]
    if description:
        sections.append(_Section(body=description, section="description"))
    return sections


def _notion_sections(body: str) -> list[_Section]:
    """Split markdown-ish Notion text at heading boundaries.

    Headings are ``#``..``######`` lines (what Notion's markdown export emits).
    Content before the first heading becomes a preamble section with an empty
    heading path. The heading stack is level-aware, so ``### X`` under ``## Y``
    yields ``("Y", "X")``.
    """
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [_Section(body=body)]

    sections: list[_Section] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(_Section(body=preamble, section="preamble"))

    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(name for _, name in stack)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        # The heading line is already in the breadcrumb (and in ``heading_path``
        # metadata), so the section body starts after it. Keeping the raw "##"
        # markers would only add FTS noise.
        after_heading = body[match.end() : end].strip()
        if not after_heading:
            # A heading with no content has no evidence in it; emitting a chunk
            # that only repeats the breadcrumb is noise.
            continue
        sections.append(_Section(body=after_heading, heading_path=path, section="heading"))
    return sections or [_Section(body=body)]


def _email_sections(body: str) -> list[_Section]:
    """Email body with quoted replies, forwarded chains, and signatures cut."""
    stripped = strip_email_quotes(body)
    return [_Section(body=stripped or body, section="email_body")]


def strip_email_quotes(body: str) -> str:
    """Remove quoted reply history and signature from an email body.

    Public because the PII/guardrail agents need the same notion of "the new
    text in this message" and duplicating the rule invites drift.
    """
    lines = body.split("\n")
    cut = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker.match(stripped) for marker in _QUOTE_MARKERS):
            cut = index
            break
        if _HEADER_START_RE.match(stripped) and any(
            _HEADER_SIBLING_RE.match(candidate.strip())
            for candidate in lines[index + 1 : index + 8]
        ):
            # Gmail's "From: … / Sent: … / To: …" header block.
            cut = index
            break

    kept = lines[:cut]
    for index, line in enumerate(kept):
        if any(marker.match(line.strip()) for marker in _SIGNATURE_MARKERS):
            kept = kept[:index]
            break
    return _normalize("\n".join(kept)).strip()


# ---------------------------------------------------------------------------
# Size-based splitting
# ---------------------------------------------------------------------------


def _pack(units: list[str], *, budget: int, overlap: int) -> list[str]:
    """Greedily pack units into chunks of at most ``budget`` tokens.

    Each new chunk starts with the tail of the previous one (~``overlap``
    tokens), so a sentence straddling a boundary is retrievable from both
    sides. The carried tail is capped at half the budget and can never be the
    whole previous chunk (that would emit duplicate chunks).
    """
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and tokens + unit_tokens + 2 > budget:
            chunks.append("\n\n".join(current))
            carry = _tail_units(current, overlap)
            current = list(carry)
            tokens = sum(estimate_tokens(item) + 2 for item in current)
            while current and tokens + unit_tokens + 2 > budget:
                tokens -= estimate_tokens(current.pop(0)) + 2
        current.append(unit)
        tokens += unit_tokens + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _tail_units(units: list[str], overlap: int) -> list[str]:
    """Trailing whole units (or trailing sentences of one unit) within budget."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if total + unit_tokens > overlap:
            if not tail:
                sentences = _sentences(unit)
                picked: list[str] = []
                for sentence in reversed(sentences):
                    sentence_tokens = estimate_tokens(sentence)
                    if total + sentence_tokens > overlap:
                        break
                    picked.append(sentence)
                    total += sentence_tokens
                tail = list(reversed(picked))
            break
        tail.insert(0, unit)
        total += unit_tokens
    if len(tail) + 1 > len(units):
        # Overlap must be a strict suffix: never repeat the entire previous chunk.
        tail = tail[1:]
    return tail


def _merge_short_tail(chunks: list[str], *, min_tokens: int, budget: int) -> list[str]:
    """Fold a tiny trailing chunk back in when the merge fits the budget."""
    if len(chunks) < 2:
        return chunks
    last = chunks[-1]
    if estimate_tokens(last) >= min_tokens:
        return chunks
    combined = f"{chunks[-2]}\n\n{last}"
    if estimate_tokens(combined) > budget:
        return chunks
    return [*chunks[:-2], combined]


def _units(text: str, budget: int) -> list[str]:
    """Paragraphs → sentences → hard word windows, each at most ``budget``."""
    units: list[str] = []
    for paragraph in _paragraphs(text):
        if estimate_tokens(paragraph) <= budget:
            units.append(paragraph)
            continue
        for sentence in _sentences(paragraph):
            if estimate_tokens(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, budget))
    return units


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in _PARAGRAPH_SPLIT_RE.split(text) if block.strip()]


def _sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def _hard_split(text: str, budget: int) -> list[str]:
    """Last resort for a single unpunctuated run (logs, base64, long tables)."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word) + 1
        if current and tokens + word_tokens > budget:
            pieces.append(" ".join(current))
            current = []
            tokens = 0
        current.append(word)
        tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _context_line(doc: SourceDocument, heading_path: tuple[str, ...]) -> str:
    """One-line breadcrumb: what this chunk is, before its body."""
    parts: list[str] = []
    if doc.source == "jira":
        if doc.doc_type == "issue_comment":
            # A comment's own source_id is opaque (a Jira comment id); the issue
            # key is what makes it findable and citable.
            parts.append(f"{doc.parent_source_id or doc.source_id} · comment")
            if doc.title.strip():
                parts.append(doc.title.strip())
        else:
            label = doc.title.strip() or doc.source_id
            if doc.source_id and doc.source_id.lower() not in label.lower():
                label = f"{doc.source_id} · {label}"
            parts.append(label)
    elif doc.source == "gmail":
        parts.append(f"Email · {doc.title.strip() or doc.source_id}")
    else:
        parts.append(doc.title.strip() or doc.source_id)

    for segment in heading_path:
        cleaned = segment.strip()
        if cleaned and cleaned.lower() not in {part.lower() for part in parts}:
            parts.append(cleaned)
    return " · ".join(part for part in parts if part)


def _meta_for(
    doc: SourceDocument,
    section: _Section,
    *,
    part: int,
    token_count: int,
    entity_ids: Sequence[str],
) -> dict[str, Any]:
    labels = sorted({*doc.labels, *section.labels})
    meta: dict[str, Any] = {
        "source": doc.source,
        "source_id": doc.source_id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "heading_path": list(section.heading_path),
        "section": section.section,
        "part": part,
        "labels": labels,
        "token_count": token_count,
        "content_hash": "",
        # Filled by the memory agent once the registry resolves matches.
        "entity_ids": list(entity_ids),
    }
    if doc.document_id:
        meta["document_id"] = doc.document_id
    if doc.parent_source_id:
        meta["parent_source_id"] = doc.parent_source_id
    if doc.author:
        meta["author"] = doc.author
    if doc.url:
        meta["url"] = doc.url
    if doc.created_at is not None:
        meta["created_at"] = doc.created_at.isoformat()
    if doc.updated_at is not None:
        meta["updated_at"] = doc.updated_at.isoformat()
    if section.fields:
        meta["fields"] = {name: value for name, value in section.fields}
    return meta


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """CRLF → LF, strip trailing spaces, cap blank runs, trim."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _truncate_words(text: str, max_tokens: int) -> str:
    words = text.split()
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    for word in words:
        if estimate_tokens(" ".join([*kept, word])) > max_tokens:
            break
        kept.append(word)
    return f"{' '.join(kept)}…" if kept else text[: max(1, max_tokens * 4)]
