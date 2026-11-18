"""Gmail connector over IMAP (read-only, app password).

Scope rules, and nothing else:

- **Label-scoped only.** A message is in scope if it lives in an allowlisted label
  mailbox; nothing outside the allowlist is ever selected, so nothing outside it
  can become a document (fail closed, see :func:`is_label_allowed`).
- **Read-only.** Every mailbox is opened with ``EXAMINE`` and bodies are fetched
  with ``BODY.PEEK[]``, so a sync never marks a message as read and never
  mutates, moves, or sends anything.
- **One document per message**, addressed by an id that is stable across syncs
  (``X-GM-MSGID``, else the RFC 822 ``Message-ID``, else a content hash).
- ``fetch_targeted`` for the planner's on-demand freshness fetches.

Why IMAP instead of the Gmail REST API: the deployment has an app password and no
OAuth client. IMAP gives the same read-only, label-scoped read access with the
standard library, so no new dependency and no token-refresh machinery.

Two deliberate deviations from the Jira adapter's behaviour, both documented at
the point of use: there is **no deletion sweep** (IMAP exposes no reliable
deleted-signal; mail is an append-mostly source) and there is **no HTTP rate
budget** (the budget is the per-label message cap).

Credentials come from the environment (``connectors.settings``) and are never
logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import imaplib
import logging
import re
from collections.abc import Collection, Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from email import message_from_bytes
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from email.message import Message
from email.policy import default as default_policy
from email.utils import parseaddr
from html import unescape
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from connectors.base import (
    ConnectorAuthError,
    ConnectorBadRequestError,
    ConnectorConfigError,
    ConnectorError,
    ConnectorNotFoundError,
    ConnectorResponseError,
    ConnectorServerError,
)
from connectors.contracts import ErrorReport, SyncRequest, SyncResult, result_timestamp
from connectors.normalizer import (
    Acl,
    CanonicalDoc,
    DocumentStore,
    SyncStateRecord,
    UpsertOutcome,
    apply_document,
)
from connectors.settings import ConnectorSettings, GmailCredentials, get_connector_settings
from db.models import DocType, Source
from indexing.chunker import strip_email_quotes

logger = logging.getLogger(__name__)

# MVP scope: the inbox plus whatever project labels the operator allowlists.
DEFAULT_ALLOWED_LABELS: tuple[str, ...] = ("INBOX",)

#: Gmail's "everything, regardless of label" mailbox — the only reliable place to
#: find a message whose label set we do not know (targeted fetches).
ALL_MAIL_MAILBOX = "[Gmail]/All Mail"

#: Socket timeout for one IMAP command. Not a setting: nobody tunes this, and a
#: hung mailbox should surface as a failure rather than a frozen sync.
IMAP_TIMEOUT_S = 30.0

GMAIL_WEB_URL = "https://mail.google.com/mail/u/0/#all/"

#: IMAP date literals must use English month abbreviations; ``strftime("%b")``
#: follows the process locale, so the table is explicit.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_MESSAGE_ID_RE = re.compile(r"<([^<>]+)>")
_INTERNALDATE_VALUE_RE = re.compile(
    r"^(\d{1,2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2}) ([+-])(\d{2})(\d{2})$"
)
# Untagged FETCH metadata: ``17 (X-GM-MSGID 123 UID 4 INTERNALDATE "..." BODY[] {n}``.
_UID_RE = re.compile(rb"\bUID (\d+)")
_GM_MSGID_RE = re.compile(rb"\bX-GM-MSGID (\d+)")
_INTERNALDATE_RE = re.compile(rb'INTERNALDATE "([^"]*)"')

_BLOCK_TAG_RE = re.compile(r"(?i)</?(?:p|div|br|tr|li|h[1-6]|blockquote)[^>]*>")
_DROP_ELEMENT_RE = re.compile(r"(?is)<(?:script|style|head)\b.*?</(?:script|style|head)>")
_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,;:!?])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Label scoping
# ---------------------------------------------------------------------------
def is_label_allowed(labels: Iterable[str], allowed: Collection[str]) -> bool:
    """True when the message carries at least one allowlisted label.

    Case-insensitive because Gmail label names are user-chosen; whitespace is
    ignored for the same reason. An empty ``allowed`` set allows nothing (fail
    closed).
    """
    allowset = {label.strip().casefold() for label in allowed if label.strip()}
    if not allowset:
        return False
    return any(label.strip().casefold() in allowset for label in labels)


def gmail_message_is_in_scope(message: dict[str, Any], allowed: Collection[str]) -> bool:
    """Scope test for a raw ``users.messages`` payload (labelIds only, no bodies)."""
    labels = message.get("labelIds")
    if not isinstance(labels, list):
        return False
    return is_label_allowed([str(label) for label in labels], allowed)


# ---------------------------------------------------------------------------
# IMAP seam
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FetchedMessage:
    """One fetched message: the raw RFC 822 bytes plus the metadata fetched with it."""

    uid: str
    raw: bytes
    #: IMAP ``INTERNALDATE`` — the server's receive time. Naive on servers that
    #: omit the zone; the connector coerces those to UTC.
    internaldate: datetime | None = None
    #: Gmail's immutable per-message id, when the server offers the extension.
    gm_msgid: str | None = None


class ImapConnection(Protocol):
    """The narrow slice of IMAP this connector needs.

    Production is :class:`ThreadedImapConnection` (``imaplib`` on a worker
    thread); tests inject an in-memory fake. Every method may raise a
    :class:`ConnectorError` subclass — never a raw ``imaplib`` error.
    """

    async def select(self, mailbox: str) -> bool:
        """Open ``mailbox`` read-only. False when it does not exist server-side."""
        ...

    async def uid_search(self, criteria: str) -> list[str]:
        """``UID SEARCH``: matching uids, ascending (i.e. oldest first)."""
        ...

    async def uid_fetch(self, uids: Sequence[str]) -> list[FetchedMessage]:
        """``UID FETCH`` bodies plus metadata, one entry per message."""
        ...

    async def logout(self) -> None:
        """Close the session. Safe to call when never logged in."""
        ...


class ThreadedImapConnection:
    """``ImapConnection`` over ``imaplib.IMAP4_SSL``.

    Every blocking call runs in a worker thread (``asyncio.to_thread``). The
    client is created lazily on first use: constructing a connector must not open
    a socket. The app password is read from its ``SecretStr`` at connect time and
    appears in no log line or exception message.
    """

    def __init__(
        self, credentials: GmailCredentials, *, timeout_s: float = IMAP_TIMEOUT_S
    ) -> None:
        self._credentials = credentials
        self._timeout_s = timeout_s
        self._client: imaplib.IMAP4_SSL | None = None
        # imaplib sessions are stateful (one selected mailbox per connection) and
        # not thread-safe, so one command at a time per connection.
        self._lock = asyncio.Lock()

    async def select(self, mailbox: str) -> bool:
        async with self._lock:
            client = await self._connect()
            return await asyncio.to_thread(self._select_sync, client, mailbox)

    async def uid_search(self, criteria: str) -> list[str]:
        async with self._lock:
            client = await self._connect()
            return await asyncio.to_thread(self._search_sync, client, criteria)

    async def uid_fetch(self, uids: Sequence[str]) -> list[FetchedMessage]:
        if not uids:
            return []
        async with self._lock:
            client = await self._connect()
            items = await asyncio.to_thread(self._fetch_sync, client, ",".join(uids))
        return _parse_fetch(items)

    async def logout(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await asyncio.to_thread(client.logout)
        except (imaplib.IMAP4.error, OSError):
            logger.debug("gmail: IMAP logout failed; closing the socket instead")
            with suppress(imaplib.IMAP4.error, OSError):
                await asyncio.to_thread(client.shutdown)

    # --- blocking half (runs in a worker thread) -----------------------
    async def _connect(self) -> imaplib.IMAP4_SSL:
        if self._client is None:
            self._client = await asyncio.to_thread(self._open)
        return self._client

    def _open(self) -> imaplib.IMAP4_SSL:
        credentials = self._credentials
        try:
            client = imaplib.IMAP4_SSL(
                credentials.imap_host, credentials.imap_port, timeout=self._timeout_s
            )
            client.login(credentials.address, credentials.app_password.get_secret_value())
        except imaplib.IMAP4.error as exc:
            # imaplib raises with the server's rejection text, which never
            # contains the password; it is capped like every other error message.
            raise ConnectorAuthError(
                "Gmail rejected the app password (IMAP login failed): "
                f"{_short(exc)} — check GMAIL_ADDRESS/GMAIL_APP_PASSWORD",
                source="gmail",
            ) from exc
        except OSError as exc:
            raise ConnectorServerError(
                f"could not reach IMAP host {credentials.imap_host}:{credentials.imap_port}: "
                f"{_short(exc)}",
                source="gmail",
            ) from exc
        return client

    def _select_sync(self, client: imaplib.IMAP4_SSL, mailbox: str) -> bool:
        try:
            # readonly=True sends EXAMINE, which cannot alter the mailbox.
            status, _ = client.select(_imap_quote(mailbox), readonly=True)
        except imaplib.IMAP4.error:
            return False
        return status == "OK"

    def _search_sync(self, client: imaplib.IMAP4_SSL, criteria: str) -> list[str]:
        try:
            # One argument, so imaplib emits exactly ``UID SEARCH <criteria>``
            # (no CHARSET: every criterion this module builds is ASCII).
            status, data = client.uid("SEARCH", criteria)
        except imaplib.IMAP4.error as exc:
            raise ConnectorResponseError(
                f"UID SEARCH failed: {_short(exc)}", source="gmail"
            ) from exc
        if status != "OK":
            raise ConnectorResponseError(f"UID SEARCH failed ({status})", source="gmail")
        uids: list[str] = []
        for item in data:
            if isinstance(item, bytes):
                uids.extend(part.decode("ascii", "replace") for part in item.split())
        return uids

    def _fetch_sync(
        self, client: imaplib.IMAP4_SSL, uid_set: str
    ) -> list[tuple[bytes, bytes]]:
        try:
            status, data = client.uid(
                "FETCH", uid_set, "(BODY.PEEK[] X-GM-MSGID INTERNALDATE)"
            )
            if status != "OK":
                # A server without the Gmail extension rejects X-GM-MSGID. Retry
                # once without it; the connector then identifies messages by
                # their Message-ID header.
                logger.info("gmail: X-GM-MSGID fetch rejected; retrying without it")
                status, data = client.uid("FETCH", uid_set, "(BODY.PEEK[] INTERNALDATE)")
        except imaplib.IMAP4.error as exc:
            raise ConnectorResponseError(
                f"UID FETCH failed: {_short(exc)}", source="gmail"
            ) from exc
        if status != "OK":
            raise ConnectorResponseError(f"UID FETCH failed ({status})", source="gmail")

        items: list[tuple[bytes, bytes]] = []
        for item in data:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], bytes)
                and isinstance(item[1], bytes)
            ):
                items.append((item[0], item[1]))
        return items


class GmailConnector:
    """Gmail messages (allowlisted labels only) -> ``core.documents``."""

    source = "gmail"

    def __init__(
        self,
        store: DocumentStore | None = None,
        settings: ConnectorSettings | None = None,
        *,
        connection: ImapConnection | None = None,
    ) -> None:
        config = settings or get_connector_settings()
        # Validated at construction, like the Jira adapter: a misconfigured source
        # fails where it is built, not halfway through a sync.
        self._credentials = config.require_gmail()
        self._config = config
        self._store = store
        self._connection = connection
        # An injected connection belongs to the caller; only a connection we open
        # ourselves is closed by ``aclose``.
        self._owns_connection = connection is None

    # --- scope ----------------------------------------------------------
    @property
    def labels(self) -> tuple[str, ...]:
        """The label mailboxes this connector may read, in configured order.

        Every candidate has to pass :func:`is_label_allowed` against the same
        allowlist, which drops blank entries and collapses case variants. Nothing
        else is ever selected, so a non-allowlisted mailbox cannot produce a
        document even if its messages would look like mail we want.
        """
        allowed = self._config.allowed_gmail_labels
        return tuple(
            label.strip() for label in allowed if is_label_allowed([label], allowed)
        )

    # --- sync -----------------------------------------------------------
    async def sync(self, req: SyncRequest) -> SyncResult:
        """Pull the allowlisted label mailboxes; collect per-item failures."""
        store = self._store
        if store is None:
            raise ConnectorConfigError(
                "GmailConnector.sync needs a DocumentStore; construct it with store=..."
            )

        started_at = datetime.now(UTC)
        state = await store.load_sync_state(self.source)
        full = req.full or state.watermark is None
        since = None if full else state.watermark

        counts = {outcome: 0 for outcome in UpsertOutcome}
        errors: list[ErrorReport] = []
        # Dedupe across mailboxes: Gmail exposes one message under every label it
        # carries, so a message in two allowlisted labels would otherwise be
        # written twice — and its differing ACL would supersede it on every run.
        processed: set[str] = set()
        skipped = 0
        watermark = state.watermark
        earliest_failure: datetime | None = None

        logger.info(
            "gmail sync starting (full=%s, labels=%s, since=%s)",
            full,
            ",".join(self.labels),
            None if full else since,
        )

        connection = self._connection_for_use()
        try:
            for label in self.labels:
                if not await connection.select(label):
                    # A label that was renamed or deleted is not an error: the
                    # allowlist is the operator's intent, not a guarantee.
                    logger.info("gmail: no mailbox for label %r server-side; skipping", label)
                    continue

                uids = await connection.uid_search(_search_criteria(since))
                if len(uids) > self._config.gmail_max_messages:
                    # IMAP returns uids ascending and Gmail allocates them
                    # monotonically, so the tail is the newest mail. Capping from
                    # the end is the documented choice: a cap that kept the
                    # *oldest* messages would never reach today's mail.
                    logger.info(
                        "gmail: %s matched %d messages; syncing the newest %d",
                        label,
                        len(uids),
                        self._config.gmail_max_messages,
                    )
                    uids = uids[-self._config.gmail_max_messages :]

                page_size = self._config.gmail_page_size
                for start in range(0, len(uids), page_size):
                    for message in await connection.uid_fetch(uids[start : start + page_size]):
                        observed = _aware(message.internaldate, fallback=started_at)
                        try:
                            doc = self._to_doc(message, label=label, fetched_at=started_at)
                        except (ValueError, KeyError, TypeError, ValidationError) as exc:
                            errors.append(
                                _error(
                                    source_id=message.uid,
                                    stage="normalize",
                                    error_type=type(exc).__name__,
                                    message=_short(exc),
                                )
                            )
                            earliest_failure = _earliest(earliest_failure, observed)
                            continue

                        if doc.source_id in processed:
                            skipped += 1
                            continue

                        try:
                            # SAVEPOINT per message: one bad row cannot poison the run.
                            async with store.unit_of_work():
                                record = await apply_document(store, doc)
                        except SQLAlchemyError as exc:
                            errors.append(
                                _error(
                                    source_id=doc.source_id,
                                    stage="persist",
                                    error_type=type(exc).__name__,
                                    message=_short(exc),
                                )
                            )
                            earliest_failure = _earliest(earliest_failure, doc.updated_at)
                            continue

                        counts[record.outcome] += 1
                        processed.add(doc.source_id)
                        watermark = _latest(watermark, doc.updated_at)
                    # Yield between pages: the fetch ran in a worker thread, but a
                    # large mailbox must not monopolise the event loop while its
                    # messages are parsed.
                    await asyncio.sleep(0)

        except ConnectorAuthError as exc:
            await self._record_failure(store, state, started_at, exc)
            raise
        except ConnectorError as exc:
            errors.append(
                _error(
                    stage="fetch",
                    error_type=type(exc).__name__,
                    message=_short(exc),
                    retryable=exc.retryable,
                )
            )
            logger.warning("gmail sync interrupted: %s", exc)

        finished_at = datetime.now(UTC)
        new_watermark = _resolve_watermark(
            watermark=watermark, failed_at=earliest_failure, fallback=started_at
        )
        status = "ok" if not errors else "partial"
        await store.save_sync_state(
            _state_after_run(self.source, new_watermark, started_at, status, errors)
        )

        result = SyncResult(
            source=self.source,
            documents_upserted=counts[UpsertOutcome.created] + counts[UpsertOutcome.updated],
            watermark=result_timestamp(new_watermark),
            errors=errors,
            documents_unchanged=counts[UpsertOutcome.unchanged],
            # No deletion sweep: IMAP has no reliable "deleted" signal (a message
            # can be archived, relabelled, or moved between runs), so mail is
            # treated as append-mostly. Stale docs age out via ``is_current``
            # only when a future adapter can prove absence.
            documents_retired=0,
            documents_skipped=skipped,
            full=full,
            started_at=result_timestamp(started_at),
            finished_at=result_timestamp(finished_at),
        )
        logger.info(
            "gmail sync finished (full=%s, upserted=%d, unchanged=%d, skipped=%d, errors=%d)",
            full,
            result.documents_upserted,
            result.documents_unchanged,
            result.documents_skipped,
            len(result.errors),
        )
        return result

    # --- normalization ---------------------------------------------------
    def _to_doc(
        self, message: FetchedMessage, *, label: str, fetched_at: datetime
    ) -> CanonicalDoc:
        """One canonical document per message.

        Everything a reader would see (names, addresses, quoted history) is kept
        verbatim: PII is scrubbed at retrieval time, and pre-redacting here would
        change ``content_hash`` per redaction run and make every message look
        modified.
        """
        parsed = message_from_bytes(message.raw, policy=default_policy)
        sender = _header(parsed, "From")
        recipient = _header(parsed, "To")
        subject = _header(parsed, "Subject") or "(no subject)"
        date_header = _header(parsed, "Date")
        message_id = _message_id(_header(parsed, "Message-ID"))
        in_reply_to = _message_id(_header(parsed, "In-Reply-To"))
        received = _aware(message.internaldate, fallback=fetched_at)

        # Quoted replies and signatures are the same text as the message they
        # quote; the chunker's public stripper is reused so both components share
        # one definition of "the new text in this message".
        body = strip_email_quotes(_body_text(parsed))

        display_name, address = parseaddr(sender)
        return CanonicalDoc(
            source=Source.gmail,
            source_id=_source_id(
                message,
                message_id=message_id,
                sender=sender,
                subject=subject,
                date=date_header,
            ),
            # A reply keeps its parent's Message-ID, so a thread is walkable.
            parent_source_id=in_reply_to,
            doc_type=DocType.email,
            title=subject,
            content=_email_content(
                sender=sender,
                recipient=recipient,
                date=date_header or received.isoformat(),
                subject=subject,
                body=body,
            ),
            url=_gmail_url(message_id),
            author=display_name or address or None,
            created_at=received,
            updated_at=received,
            fetched_at=fetched_at,
            # Single-user MVP: the account owner is the only reader of their own
            # mail. Widening this needs the M4 identity layer (per-user mailboxes
            # and a delegation model), not a larger list here.
            acl=Acl(
                users=[f"email:{self._credentials.address.lower()}"],
                groups=[f"gmail:label:{label}"],
                labels=[label],
            ),
        )

    # --- targeted fetches -----------------------------------------------
    async def fetch_targeted(
        self,
        *,
        source_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[CanonicalDoc]:
        """Fetch one message on demand. Writes nothing.

        ``source_id`` is an ``X-GM-MSGID`` (what the connector stores as the
        document id when Gmail offers it). ``query`` is not implemented: IMAP
        search syntax is not what the planner speaks, and nothing needs it yet.
        """
        if source_id is None:
            raise ConnectorBadRequestError(
                "Gmail targeted fetch needs source_id (an X-GM-MSGID); "
                "query= is not implemented — IMAP search syntax is not a planner-facing "
                "query language yet. No request was made.",
                source=self.source,
            )
        msgid = _validate_msgid(source_id)

        connection = self._connection_for_use()
        if not await connection.select(ALL_MAIL_MAILBOX):
            raise ConnectorNotFoundError(
                f"mailbox {ALL_MAIL_MAILBOX} is not available on this account; "
                f"cannot look up message {msgid}",
                source=self.source,
            )
        uids = await connection.uid_search(f"X-GM-MSGID {msgid}")
        if not uids:
            raise ConnectorNotFoundError(
                f"no message with X-GM-MSGID {msgid}", source=self.source
            )

        fetched_at = datetime.now(UTC)
        messages = await connection.uid_fetch(uids[:1])
        if not messages:
            raise ConnectorNotFoundError(
                f"message {msgid} could not be fetched", source=self.source
            )
        # All Mail contains the message whatever its labels. The document's ACL
        # names the first allowlisted label, which is also what a sync assigns to
        # a message in several allowlisted mailboxes (first label wins), and this
        # path writes nothing, so a mismatch is only visible in the return value.
        label = self.labels[0] if self.labels else ALL_MAIL_MAILBOX
        return [self._to_doc(messages[0], label=label, fetched_at=fetched_at)]

    # --- lifecycle -------------------------------------------------------
    async def aclose(self) -> None:
        """Log out of IMAP when this connector opened the connection."""
        connection = self._connection
        if connection is None or not self._owns_connection:
            return
        self._owns_connection = False
        await connection.logout()

    def _connection_for_use(self) -> ImapConnection:
        if self._connection is None:
            self._connection = ThreadedImapConnection(self._credentials)
            self._owns_connection = True
        return self._connection

    async def _record_failure(
        self,
        store: DocumentStore,
        state: SyncStateRecord,
        started_at: datetime,
        exc: ConnectorError,
    ) -> None:
        """Persist a terminal failure so the next run's status is explainable."""
        await store.save_sync_state(
            _state_after_run(
                self.source,
                state.watermark,
                started_at,
                "error",
                [_error(stage="fetch", error_type=type(exc).__name__, message=_short(exc))],
            )
        )

    def __repr__(self) -> str:  # never renders the app password
        return f"GmailConnector(source='gmail', labels={self.labels!r})"


# ---------------------------------------------------------------------------
# Message -> document helpers
# ---------------------------------------------------------------------------
def _source_id(
    message: FetchedMessage,
    *,
    message_id: str | None,
    sender: str,
    subject: str,
    date: str,
) -> str:
    """Stable identity for one message, in order of decreasing authority.

    ``X-GM-MSGID`` is Gmail's immutable per-message id: it survives label moves,
    archive/restore, and re-syncs, which is exactly what supersession and chunk
    reuse depend on. Falling back to the RFC 822 ``Message-ID`` keeps the id
    stable for servers without the Gmail extension. The last resort hashes the
    from/subject/date triple because it is stable for a message that never
    changes — an edited body would (correctly) look like a new message rather
    than silently rewriting an id.
    """
    if message.gm_msgid:
        return message.gm_msgid
    if message_id:
        return message_id
    digest = hashlib.sha256()
    for part in (sender, subject, date):
        digest.update(part.strip().casefold().encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _gmail_url(message_id: str | None) -> str | None:
    """Deep link to the message, when it has a Message-ID to address it by."""
    if not message_id:
        return None
    return f"{GMAIL_WEB_URL}{message_id}"


def _email_content(*, sender: str, recipient: str, date: str, subject: str, body: str) -> str:
    """Header block + new text — the shape the chunker's email splitter expects."""
    lines = [
        f"{name}: {value}"
        for name, value in (
            ("From", sender),
            ("To", recipient),
            ("Date", date),
            ("Subject", subject),
        )
        if value.strip()
    ]
    if body.strip():
        lines.extend(["", body.strip()])
    return "\n".join(lines)


def _header(message: Message, name: str) -> str:
    """Decoded, un-folded header value (RFC 2047 words resolved)."""
    raw = message.get(name)
    if raw is None:
        return ""
    try:
        decoded = str(make_header(decode_header(str(raw))))
    except (HeaderParseError, LookupError, UnicodeDecodeError, ValueError):
        decoded = str(raw)
    return _INLINE_SPACE_RE.sub(" ", decoded.replace("\r", " ").replace("\n", " ")).strip()


def _message_id(raw: str) -> str | None:
    """``<abc@host>`` -> ``abc@host``: the id form the source_id/url convention uses."""
    text = raw.strip()
    if not text:
        return None
    match = _MESSAGE_ID_RE.search(text)
    return match.group(1).strip() if match else text


def _body_text(message: Message) -> str:
    """New-message text: ``text/plain`` if present, else ``text/html`` as text."""
    plain: list[str] = []
    html: list[str] = []
    for part in _leaf_parts(message):
        ctype = part.get_content_type()
        if ctype not in {"text/plain", "text/html"}:
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except LookupError:  # unknown charset label
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/plain":
            plain.append(text)
        else:
            html.append(_html_to_text(text))
    return "\n\n".join(plain or html)


def _leaf_parts(message: Message) -> Iterator[Message]:
    """Leaf parts of a message, without descending into attached/forwarded messages.

    A forwarded mail is its own document in the mailbox; inlining it here would
    duplicate that document's text under the forwarder's id.
    """
    if message.get_content_type() == "message/rfc822":
        return
    if message.is_multipart():
        children = message.get_payload()
        if not isinstance(children, list):
            return
        for child in children:
            if isinstance(child, Message):
                yield from _leaf_parts(child)
        return
    if message.get_content_disposition() == "attachment" or message.get_filename():
        return
    yield message


def _html_to_text(raw: str) -> str:
    """Tags -> text. Deliberately crude: enough to index an html-only mail."""
    text = _DROP_ELEMENT_RE.sub(" ", raw)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Sync-state / timestamp helpers
# ---------------------------------------------------------------------------
def _search_criteria(since: datetime | None) -> str:
    """Incremental search from the watermark; a full sync searches everything.

    IMAP date literals are day-granular and ``SINCE`` is inclusive, so the whole
    boundary day is re-fetched and then skipped by ``content_hash`` — the same
    lossless-rather-than-off-by-a-day trade-off the Jira adapter makes with
    minute precision.
    """
    if since is None:
        return "ALL"
    return f"SINCE {_imap_date(since)}"


def _imap_date(value: datetime) -> str:
    """``2026-11-17T...`` -> ``17-Nov-2026`` (IMAP's required date literal)."""
    utc = value.astimezone(UTC)
    return f"{utc.day:02d}-{_MONTHS[utc.month - 1]}-{utc.year:04d}"


def _aware(value: datetime | None, *, fallback: datetime) -> datetime:
    """IMAP INTERNALDATE, coerced to an aware timestamp.

    ``CanonicalDoc`` rejects naive timestamps (``AwareDatetime``), and a naive
    value from a server that omits the zone offset would otherwise poison the
    watermark comparison, so it is read as UTC here rather than at the model.
    """
    if value is None:
        return fallback
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _parse_internaldate(value: str | None) -> datetime | None:
    """``17-Nov-2026 12:34:56 +0000`` -> aware datetime (None when unparseable)."""
    if not value:
        return None
    match = _INTERNALDATE_VALUE_RE.match(value.strip())
    if match is None:
        return None
    day, month_name, year, hour, minute, second, sign, off_h, off_m = match.groups()
    try:
        month = _MONTHS.index(month_name.capitalize()) + 1
        offset = timedelta(hours=int(off_h), minutes=int(off_m))
        if sign == "-":
            offset = -offset
        return datetime(
            int(year),
            month,
            int(day),
            int(hour),
            int(minute),
            int(second),
            tzinfo=timezone(offset),
        )
    except (ValueError, IndexError):
        return None


def _resolve_watermark(
    *, watermark: datetime | None, failed_at: datetime | None, fallback: datetime
) -> datetime:
    """New cursor.

    When a message failed, the cursor must not move past it or its day would be
    skipped forever: clamp to just before it. IMAP search is day-granular, so the
    clamp re-reads that whole day on the next run anyway.
    """
    candidate = watermark or fallback
    if failed_at is not None:
        clamped = failed_at - timedelta(seconds=1)
        if clamped < candidate:
            candidate = clamped
    return candidate


def _state_after_run(
    source: str,
    watermark: datetime | None,
    started_at: datetime,
    status: str,
    errors: list[ErrorReport],
) -> SyncStateRecord:
    return SyncStateRecord(
        source=source,
        watermark=watermark,
        last_run_at=started_at,
        last_status=status,
        # First error only: the full list lives in SyncResult. Keeps the column short.
        error=errors[0].message if errors else None,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _error(
    *,
    stage: str,
    error_type: str,
    message: str,
    source_id: str | None = None,
    retryable: bool = False,
) -> ErrorReport:
    return ErrorReport(
        source="gmail",
        source_id=source_id,
        stage=stage,
        error_type=error_type,
        message=message,
        retryable=retryable,
    )


def _short(exc: BaseException) -> str:
    """One-line, length-capped message: source bodies must not flood the report."""
    text = str(exc).replace("\n", " ").strip()
    return text[:300] or type(exc).__name__


def _latest(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate > current:
        return candidate
    return current


def _earliest(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate < current:
        return candidate
    return current


def _imap_quote(value: str) -> str:
    """``imaplib`` sends strings verbatim, so a mailbox name with a space needs this."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _validate_msgid(source_id: str) -> str:
    normalised = source_id.strip()
    if not normalised.isdigit():
        raise ConnectorConfigError(
            f"invalid Gmail message id: {source_id!r} (expected an X-GM-MSGID: digits only)"
        )
    return normalised


def _parse_fetch(items: Sequence[tuple[bytes, bytes]]) -> list[FetchedMessage]:
    """IMAP FETCH metadata + body literals -> :class:`FetchedMessage`."""
    messages: list[FetchedMessage] = []
    for meta, raw in items:
        uid = _first_group(_UID_RE, meta)
        if uid is None:
            # Without a uid the message cannot be addressed; skipping beats
            # guessing an identity that would churn on the next sync.
            logger.warning("gmail: FETCH response carried no UID; ignoring that message")
            continue
        messages.append(
            FetchedMessage(
                uid=uid,
                raw=raw,
                internaldate=_parse_internaldate(_first_group(_INTERNALDATE_RE, meta)),
                gm_msgid=_first_group(_GM_MSGID_RE, meta),
            )
        )
    return messages


def _first_group(pattern: re.Pattern[bytes], data: bytes) -> str | None:
    match = pattern.search(data)
    if match is None:
        return None
    return match.group(1).decode("ascii", "replace")


__all__ = [
    "ALL_MAIL_MAILBOX",
    "DEFAULT_ALLOWED_LABELS",
    "FetchedMessage",
    "GmailConnector",
    "ImapConnection",
    "ThreadedImapConnection",
    "gmail_message_is_in_scope",
    "is_label_allowed",
]
