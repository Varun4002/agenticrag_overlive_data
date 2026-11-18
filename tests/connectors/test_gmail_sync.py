"""Gmail sync behaviour over a fake IMAP server.

Everything runs offline: ``FakeImap`` (``gmail_fixtures``) implements the
connector's ``ImapConnection`` protocol over hand-built RFC 822 messages, and
``InMemoryDocumentStore`` implements the same persistence port the SQLAlchemy
store does. No test in this module opens a socket.
"""

from __future__ import annotations

import imaplib
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from gmail_fixtures import (
    APP_PASSWORD,
    OWNER,
    FakeImap,
    build_message,
    default_inbox,
    default_mailboxes,
    make_gmail_settings,
)

from connectors.base import (
    ConnectorAuthError,
    ConnectorBadRequestError,
    ConnectorConfigError,
)
from connectors.contracts import SyncRequest, SyncResult
from connectors.gmail import FetchedMessage, GmailConnector, ThreadedImapConnection, _parse_fetch
from connectors.normalizer import CanonicalDoc, InMemoryDocumentStore
from db.models import DocType, Source


def connector_for(
    fake: FakeImap,
    store: InMemoryDocumentStore,
    **settings_overrides: object,
) -> GmailConnector:
    return GmailConnector(
        store, make_gmail_settings(**settings_overrides), connection=fake
    )


def current_docs(store: InMemoryDocumentStore) -> dict[str, CanonicalDoc]:
    """Current documents keyed by ``source_id``."""
    return {doc.source_id: doc for doc in store.rows.values() if doc.is_current}


async def sync_full(
    fake: FakeImap, store: InMemoryDocumentStore, **settings_overrides: object
) -> SyncResult:
    connector = connector_for(fake, store, **settings_overrides)
    try:
        return await connector.sync(SyncRequest(source="gmail", full=True))
    finally:
        await connector.aclose()


# --- full sync ------------------------------------------------------------
async def test_full_sync_upserts_one_doc_per_message(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    try:
        result = await connector.sync(SyncRequest(source="gmail"))
    finally:
        await connector.aclose()

    # INBOX (5 messages) + project-phoenix (1); SENT is not allowlisted.
    assert result.documents_upserted == 6
    assert result.documents_unchanged == 0
    assert result.documents_retired == 0
    assert result.documents_skipped == 0
    assert result.errors == []
    assert result.full is True
    # Newest INTERNALDATE seen, with the naive one read as UTC.
    assert result.watermark == datetime(2026, 11, 17, 15, 0, tzinfo=UTC)

    docs = current_docs(store)
    assert set(docs) == {"9001", "9002", "html@example.com", "9004", "9005", "9101"}
    assert all(doc.source is Source.gmail for doc in docs.values())
    assert all(doc.doc_type is DocType.email for doc in docs.values())
    assert store.sync_state["gmail"].last_status == "ok"


async def test_message_becomes_a_canonical_document(store: InMemoryDocumentStore) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    doc = current_docs(store)["9001"]
    assert doc.title == "Kickoff notes"
    assert doc.author == "Ada Lovelace"
    assert doc.url == "https://mail.google.com/mail/u/0/#all/kickoff@example.com"
    assert doc.created_at == datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
    assert doc.updated_at == datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
    # Header block first, then the body — what the chunker's email splitter expects.
    assert doc.content.startswith("From: Ada Lovelace <ada@example.com>")
    assert "To: Varun <owner@example.com>" in doc.content
    assert "Date: Tue, 17 Nov 2026 09:00:00 +0000" in doc.content
    assert "Subject: Kickoff notes" in doc.content
    assert "Kickoff is Monday at ten." in doc.content
    # Full name + address are kept verbatim: PII scrubbing happens downstream.
    assert "ada@example.com" in doc.content

    # Single-user MVP: the account owner is the only reader.
    assert doc.acl.users == [f"email:{OWNER}"]
    assert doc.acl.groups == ["gmail:label:INBOX"]
    assert doc.acl.labels == ["INBOX"]


async def test_reply_keeps_its_thread_parent(store: InMemoryDocumentStore) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    reply = current_docs(store)["9002"]
    assert reply.parent_source_id == "kickoff@example.com"


async def test_message_without_x_gm_msgid_falls_back_to_message_id(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    doc = current_docs(store)["html@example.com"]
    assert doc.source_id == "html@example.com"


async def test_message_without_subject_gets_a_placeholder_title(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    assert current_docs(store)["9004"].title == "(no subject)"


async def test_quoted_reply_text_is_stripped(store: InMemoryDocumentStore) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    content = current_docs(store)["9001"].content
    assert "Kickoff is Monday at ten." in content
    # The quoted chain and the "wrote:" marker are the same text as another
    # message; indexing them would duplicate the corpus.
    assert "earlier text that must not be indexed" not in content
    assert "wrote:" not in content


async def test_html_only_message_is_converted_to_text(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    content = current_docs(store)["html@example.com"].content
    body = content.split("\n\n", 1)[1]
    assert "Shipped v2 of the reader." in body
    assert "Second paragraph." in body
    assert "<" not in body  # no tags left in the extracted text


async def test_naive_internaldate_is_coerced_to_utc(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)

    doc = current_docs(store)["9005"]
    assert doc.created_at == datetime(2026, 11, 17, 15, 0, tzinfo=UTC)
    assert doc.created_at.utcoffset() == timedelta(0)
    assert doc.updated_at == doc.created_at


# --- scoping --------------------------------------------------------------
async def test_out_of_allowlist_mailbox_is_never_read(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    result = await sync_full(fake, store, gmail_labels="INBOX")

    assert "SENT" not in fake.selects
    assert "9201" not in current_docs(store)
    assert result.documents_skipped == 0
    assert all(
        doc.acl.labels == ["INBOX"] for doc in current_docs(store).values()
    )


async def test_missing_label_mailbox_is_skipped_without_error(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    result = await sync_full(fake, store, gmail_labels="INBOX,project-atlas")

    assert "project-atlas" in fake.selects  # looked for ...
    assert result.errors == []  # ... and its absence is not an error
    assert result.documents_upserted == 5  # only INBOX's messages
    assert all(
        doc.acl.labels != ["project-atlas"] for doc in current_docs(store).values()
    )


async def test_message_in_two_allowlisted_labels_is_written_once(
    store: InMemoryDocumentStore,
) -> None:
    """Gmail exposes one message under every label it carries.

    Writing it twice would rewrite the row on every run, because the two copies
    disagree about ``acl.labels``.
    """
    shared = build_message(
        uid="401",
        gm_msgid="9301",
        subject="Labelled twice",
        message_id="<twice@example.com>",
        internaldate=datetime(2026, 11, 16, 9, 0, tzinfo=UTC),
    )
    fake = (
        FakeImap()
        .with_mailbox("INBOX", shared)
        .with_mailbox("project-phoenix", shared)
        .with_mailbox("[Gmail]/All Mail", shared)
    )

    first = await sync_full(fake, store)
    assert first.documents_upserted == 1
    assert first.documents_skipped == 1
    assert len(store.rows) == 1

    second = await sync_full(fake, store)
    assert second.documents_upserted == 0
    assert second.documents_unchanged == 1
    assert len(store.rows) == 1  # no supersession churn


# --- incremental ----------------------------------------------------------
async def test_incremental_sync_searches_from_the_watermark(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    try:
        await connector.sync(SyncRequest(source="gmail"))
    finally:
        await connector.aclose()

    fake.add_message(
        "INBOX",
        build_message(
            uid="106",
            gm_msgid="9006",
            subject="Later mail",
            message_id="<later@example.com>",
            body="Arrived after the first sync.",
            internaldate=datetime(2026, 11, 18, 8, 0, tzinfo=UTC),
        ),
    )

    connector = connector_for(fake, store)
    try:
        result = await connector.sync(SyncRequest(source="gmail"))
    finally:
        await connector.aclose()

    assert result.full is False
    assert "SINCE 17-Nov-2026" in fake.searched_criteria()
    assert result.documents_upserted == 1
    # The boundary day is re-searched (SINCE is inclusive) and skipped by hash.
    assert result.documents_unchanged == 3
    assert result.watermark == datetime(2026, 11, 18, 8, 0, tzinfo=UTC)
    assert "9006" in current_docs(store)
    assert len(store.rows) == 7


async def test_full_sync_has_no_cursor(store: InMemoryDocumentStore) -> None:
    fake = default_mailboxes()
    await sync_full(fake, store)
    await sync_full(fake, store)

    assert fake.searched_criteria() == ["ALL", "ALL", "ALL", "ALL"]


async def test_max_messages_cap_keeps_the_newest(store: InMemoryDocumentStore) -> None:
    fake = default_mailboxes()
    result = await sync_full(
        fake, store, gmail_labels="INBOX", gmail_max_messages=2, gmail_page_size=1
    )

    assert result.documents_upserted == 2
    assert set(current_docs(store)) == {"9004", "9005"}  # the two newest uids
    assert fake.fetches == [("INBOX", ("104",)), ("INBOX", ("105",))]


# --- failure handling -----------------------------------------------------
async def test_auth_failure_raises_and_records_state(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes().fail_auth()
    connector = connector_for(fake, store)
    try:
        with pytest.raises(ConnectorAuthError):
            await connector.sync(SyncRequest(source="gmail"))
    finally:
        await connector.aclose()

    state = store.sync_state["gmail"]
    assert state.last_status == "error"
    assert "app password" in (state.error or "")
    assert store.rows == {}


async def test_fetch_failure_returns_a_partial_result(
    store: InMemoryDocumentStore,
) -> None:
    class BreakingImap(FakeImap):
        async def uid_fetch(self, uids: Sequence[str]) -> list[FetchedMessage]:
            raise ConnectorBadRequestError("malformed uid set", source="gmail")

    fake = BreakingImap().with_mailbox("INBOX", *default_inbox())
    result = await sync_full(fake, store, gmail_labels="INBOX")

    assert [error.stage for error in result.errors] == ["fetch"]
    assert result.errors[0].error_type == "ConnectorBadRequestError"
    assert result.documents_upserted == 0
    assert store.sync_state["gmail"].last_status == "partial"


async def test_unconfigured_connector_raises_config_error(
    store: InMemoryDocumentStore,
) -> None:
    with pytest.raises(ConnectorConfigError, match="GMAIL_ADDRESS"):
        GmailConnector(store, make_gmail_settings(gmail_address=""))
    with pytest.raises(ConnectorConfigError, match="GMAIL_APP_PASSWORD"):
        GmailConnector(store, make_gmail_settings(gmail_app_password=""))


async def test_app_password_never_appears_in_logs_or_results(
    store: InMemoryDocumentStore, caplog: pytest.LogCaptureFixture
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    caplog.set_level(logging.DEBUG)
    try:
        result = await connector.sync(SyncRequest(source="gmail"))
        await connector.fetch_targeted(source_id="9001")
    finally:
        await connector.aclose()

    assert APP_PASSWORD not in caplog.text
    assert APP_PASSWORD not in repr(connector)
    assert APP_PASSWORD not in result.model_dump_json()


# --- targeted fetches -----------------------------------------------------
async def test_fetch_targeted_returns_the_doc_and_writes_nothing(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    try:
        docs = await connector.fetch_targeted(source_id="9001")
    finally:
        await connector.aclose()

    assert [doc.source_id for doc in docs] == ["9001"]
    assert docs[0].title == "Kickoff notes"
    assert docs[0].doc_type is DocType.email
    assert fake.selects[-1] == "[Gmail]/All Mail"
    assert fake.searched_criteria()[-1] == "X-GM-MSGID 9001"
    assert store.rows == {}  # a targeted fetch never writes


async def test_fetch_targeted_query_is_not_implemented(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    try:
        with pytest.raises(ConnectorBadRequestError, match="query"):
            await connector.fetch_targeted(query="from:ada@example.com")
    finally:
        await connector.aclose()

    assert fake.selects == []  # refused before touching the server


async def test_fetch_targeted_rejects_a_non_numeric_msgid(
    store: InMemoryDocumentStore,
) -> None:
    fake = default_mailboxes()
    connector = connector_for(fake, store)
    try:
        with pytest.raises(ConnectorConfigError, match="invalid Gmail message id"):
            await connector.fetch_targeted(source_id='9001" OR SINCE 1-Jan-2000')
    finally:
        await connector.aclose()

    assert fake.searches == []


async def test_labels_property_is_fail_closed(store: InMemoryDocumentStore) -> None:
    connector = GmailConnector(
        store,
        make_gmail_settings(gmail_labels=" INBOX , ,project-phoenix "),
        connection=default_mailboxes(),
    )
    assert connector.labels == ("INBOX", "project-phoenix")
    await connector.aclose()


# --- imaplib response parsing --------------------------------------------
# The fake speaks the connector's protocol, so these two tests are the only
# offline coverage of the wire shape the production wrapper has to parse. They
# use hand-built metadata (the layout Gmail documents for an untagged FETCH), not
# a recording — a live login is out of scope (no credentials on this machine).
def test_imap_fetch_response_is_parsed_into_messages() -> None:
    meta = (
        b'17 (X-GM-MSGID 1278455344230334865 UID 17 RFC822.SIZE 14 '
        b'INTERNALDATE "17-Nov-2026 09:30:00 +0000" BODY[] {14}'
    )
    messages = _parse_fetch([(meta, b"Subject: hi\r\n")])

    assert messages == [
        FetchedMessage(
            uid="17",
            raw=b"Subject: hi\r\n",
            internaldate=datetime(2026, 11, 17, 9, 30, tzinfo=UTC),
            gm_msgid="1278455344230334865",
        )
    ]


def test_imap_fetch_without_gmail_extension_is_parsed() -> None:
    """A server without X-GM-MSGID: no id in the metadata, so Message-ID is used."""
    meta = b'2 (UID 2 INTERNALDATE "17-Nov-2026 09:30:00 +0000" BODY[] {5}'
    messages = _parse_fetch([(meta, b"hello")])

    assert [(message.uid, message.gm_msgid) for message in messages] == [("2", None)]


def test_unparseable_internaldate_is_reported_as_missing() -> None:
    meta = b'3 (UID 3 INTERNALDATE "not-a-date" BODY[] {2}'
    assert _parse_fetch([(meta, b"hi")])[0].internaldate is None


class _StubImapClient:
    """Duck-typed ``imaplib`` client: records commands, returns canned responses."""

    def __init__(
        self,
        *,
        select_response: object = ("OK", [b"1"]),
        uid_responses: Sequence[object] = (),
    ) -> None:
        self.select_response = select_response
        self.uid_responses = list(uid_responses)
        self.select_args: list[tuple[str, bool]] = []
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def select(self, mailbox: str, readonly: bool = False) -> object:
        self.select_args.append((mailbox, readonly))
        if isinstance(self.select_response, Exception):
            raise self.select_response
        return self.select_response

    def uid(self, command: str, *args: str) -> object:
        self.calls.append((command, args))
        response = self.uid_responses.pop(0) if self.uid_responses else ("OK", [])
        if isinstance(response, Exception):
            raise response
        return response


def test_imap_select_is_read_only_and_quotes_the_mailbox() -> None:
    connection = ThreadedImapConnection(make_gmail_settings().require_gmail())
    client = _StubImapClient()

    assert connection._select_sync(client, "[Gmail]/All Mail") is True
    # Quoted (imaplib does not), and EXAMINE rather than SELECT.
    assert client.select_args == [('"[Gmail]/All Mail"', True)]


def test_missing_mailbox_selects_as_false() -> None:
    connection = ThreadedImapConnection(make_gmail_settings().require_gmail())
    client = _StubImapClient(select_response=imaplib.IMAP4.error("no such mailbox"))

    assert connection._select_sync(client, "project-atlas") is False


def test_imap_uid_search_parses_the_uid_list() -> None:
    connection = ThreadedImapConnection(make_gmail_settings().require_gmail())
    client = _StubImapClient(uid_responses=[("OK", [b"12 34 56"])])

    assert connection._search_sync(client, "SINCE 17-Nov-2026") == ["12", "34", "56"]
    # One argument, so imaplib emits ``UID SEARCH SINCE 17-Nov-2026``.
    assert client.calls == [("SEARCH", ("SINCE 17-Nov-2026",))]


def test_imap_uid_fetch_retries_without_the_gmail_extension() -> None:
    connection = ThreadedImapConnection(make_gmail_settings().require_gmail())
    body = b'1 (UID 1 INTERNALDATE "17-Nov-2026 09:30:00 +0000" BODY[] {5}'
    client = _StubImapClient(
        uid_responses=[("BAD", [b"Unknown command"]), ("OK", [(body, b"hello")])]
    )

    items = connection._fetch_sync(client, "1")

    assert [args for _, args in client.calls] == [
        ("1", "(BODY.PEEK[] X-GM-MSGID INTERNALDATE)"),
        ("1", "(BODY.PEEK[] INTERNALDATE)"),
    ]
    assert _parse_fetch(items)[0].raw == b"hello"
