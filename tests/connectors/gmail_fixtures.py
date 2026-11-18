"""Test helpers for the Gmail connector: an in-memory IMAP server.

Kept out of ``conftest.py`` so test modules can import them directly
(``from gmail_fixtures import FakeImap, default_mailboxes``) — pytest puts this
directory on ``sys.path`` because the suite has no ``__init__.py``.

No network anywhere: ``FakeImap`` implements the connector's ``ImapConnection``
protocol over hand-built RFC 822 messages, and records every command so a test
can assert exactly what was asked of the server.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from connectors.base import ConnectorAuthError
from connectors.gmail import FetchedMessage
from connectors.settings import ConnectorSettings

# Fixed clock so watermark/`fetched_at` values are deterministic.
FIXED_NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
OWNER = "owner@example.com"
# Obviously fake: no real credential ever appears in this repository.
APP_PASSWORD = "fake-app-password-not-real"

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def make_gmail_settings(**overrides: Any) -> ConnectorSettings:
    """Connector settings for tests — fake credentials, never real ones."""
    values: dict[str, Any] = {
        "gmail_address": OWNER,
        "gmail_app_password": APP_PASSWORD,
        "gmail_labels": "INBOX,project-phoenix",
        "gmail_page_size": 2,
    }
    values.update(overrides)
    return ConnectorSettings(**values)


@dataclass(frozen=True, slots=True)
class FakeMessage:
    """One message in a fake mailbox."""

    uid: str
    raw: bytes
    internaldate: datetime | None = FIXED_NOW
    gm_msgid: str | None = None

    def as_fetched(self) -> FetchedMessage:
        return FetchedMessage(
            uid=self.uid,
            raw=self.raw,
            internaldate=self.internaldate,
            gm_msgid=self.gm_msgid,
        )


def build_message(
    *,
    uid: str,
    subject: str | None = "Status update",
    sender: str = "Ada Lovelace <ada@example.com>",
    recipient: str = f"Varun <{OWNER}>",
    date: str = "Tue, 17 Nov 2026 09:00:00 +0000",
    body: str = "Plain body.",
    content_type: str = "text/plain",
    message_id: str | None = "<msg-1@example.com>",
    in_reply_to: str | None = None,
    gm_msgid: str | None = None,
    internaldate: datetime | None = FIXED_NOW,
) -> FakeMessage:
    """Build a message's raw RFC 822 bytes plus its IMAP metadata."""
    headers = [f"From: {sender}", f"To: {recipient}", f"Date: {date}", "MIME-Version: 1.0"]
    if subject is not None:
        headers.append(f"Subject: {subject}")
    if message_id is not None:
        headers.append(f"Message-ID: {message_id}")
    if in_reply_to is not None:
        headers.append(f"In-Reply-To: {in_reply_to}")
    headers.append(f"Content-Type: {content_type}; charset=utf-8")
    raw = ("\r\n".join(headers) + "\r\n\r\n" + body).encode("utf-8")
    return FakeMessage(uid=uid, raw=raw, internaldate=internaldate, gm_msgid=gm_msgid)


class FakeImap:
    """A fake Gmail account: label mailboxes, uid search/fetch, and a command log."""

    def __init__(self) -> None:
        self.mailboxes: dict[str, list[FakeMessage]] = {}
        self.selects: list[str] = []
        self.searches: list[tuple[str, str]] = []
        self.fetches: list[tuple[str, tuple[str, ...]]] = []
        self.selected: str | None = None
        self.logged_out = False
        self.auth_failure: str | None = None

    # --- configuration -------------------------------------------------
    def with_mailbox(self, name: str, *messages: FakeMessage) -> FakeImap:
        self.mailboxes[name] = list(messages)
        return self

    def add_message(self, mailbox: str, message: FakeMessage) -> FakeImap:
        self.mailboxes.setdefault(mailbox, []).append(message)
        return self

    def fail_auth(self, detail: str = "[AUTHENTICATIONFAILED] Invalid credentials") -> FakeImap:
        self.auth_failure = detail
        return self

    def searched_criteria(self) -> list[str]:
        """Criteria of every ``UID SEARCH``, in order."""
        return [criteria for _, criteria in self.searches]

    # --- ImapConnection -------------------------------------------------
    async def select(self, mailbox: str) -> bool:
        if self.auth_failure is not None:
            raise ConnectorAuthError(
                f"Gmail rejected the app password (IMAP login failed): {self.auth_failure}",
                source="gmail",
            )
        self.selects.append(mailbox)
        if mailbox not in self.mailboxes:
            self.selected = None
            return False
        self.selected = mailbox
        return True

    async def uid_search(self, criteria: str) -> list[str]:
        assert self.selected is not None, "uid_search called without a selected mailbox"
        self.searches.append((self.selected, criteria))
        messages = self.mailboxes[self.selected]
        if criteria == "ALL":
            return [message.uid for message in messages]
        if criteria.startswith("SINCE "):
            since = _parse_since(criteria)
            return [
                message.uid
                for message in messages
                if message.internaldate is not None
                and _utc_day(message.internaldate) >= since
            ]
        if criteria.startswith("X-GM-MSGID "):
            wanted = criteria.split(" ", 1)[1]
            return [message.uid for message in messages if message.gm_msgid == wanted]
        raise AssertionError(f"unexpected search criteria: {criteria!r}")

    async def uid_fetch(self, uids: Sequence[str]) -> list[FetchedMessage]:
        assert self.selected is not None, "uid_fetch called without a selected mailbox"
        self.fetches.append((self.selected, tuple(uids)))
        by_uid = {message.uid: message for message in self.mailboxes[self.selected]}
        return [by_uid[uid].as_fetched() for uid in uids if uid in by_uid]

    async def logout(self) -> None:
        self.logged_out = True


def _parse_since(criteria: str) -> datetime:
    _, literal = criteria.split(" ", 1)
    day, month, year = literal.split("-")
    return datetime(int(year), _MONTHS.index(month) + 1, int(day), tzinfo=UTC)


def _utc_day(value: datetime) -> datetime:
    utc = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime(utc.year, utc.month, utc.day, tzinfo=UTC)


def default_inbox() -> list[FakeMessage]:
    """Five messages covering the shapes the connector has to normalize."""
    return [
        build_message(
            uid="101",
            gm_msgid="9001",
            subject="Kickoff notes",
            message_id="<kickoff@example.com>",
            body=(
                "Kickoff is Monday at ten.\n\n"
                "On Fri, 14 Nov 2026 at 10:00, Bob <bob@example.com> wrote:\n"
                "> earlier text that must not be indexed\n"
            ),
            internaldate=datetime(2026, 11, 16, 9, 0, tzinfo=UTC),
        ),
        build_message(
            uid="102",
            gm_msgid="9002",
            subject="Re: Kickoff notes",
            message_id="<reply@example.com>",
            in_reply_to="<kickoff@example.com>",
            body="Monday works for me.",
            internaldate=datetime(2026, 11, 16, 10, 0, tzinfo=UTC),
        ),
        build_message(
            uid="103",
            # No X-GM-MSGID: the connector has to fall back to the Message-ID.
            gm_msgid=None,
            subject="HTML only",
            message_id="<html@example.com>",
            content_type="text/html",
            body=(
                "<html><body><p>Shipped <b>v2</b> of the reader.</p>"
                "<p>Second paragraph.</p></body></html>"
            ),
            internaldate=datetime(2026, 11, 17, 8, 0, tzinfo=UTC),
        ),
        build_message(
            uid="104",
            gm_msgid="9004",
            subject=None,
            message_id="<nosubject@example.com>",
            body="No subject on this one.",
            internaldate=datetime(2026, 11, 17, 9, 0, tzinfo=UTC),
        ),
        build_message(
            uid="105",
            gm_msgid="9005",
            subject="Naive timestamp",
            message_id="<naive@example.com>",
            body="Server forgot the zone.",
            # Naive on purpose: some servers omit the INTERNALDATE offset.
            internaldate=datetime(2026, 11, 17, 15, 0),
        ),
    ]


def default_mailboxes() -> FakeImap:
    """Two allowlisted labels, one non-allowlisted label, plus All Mail."""
    inbox = default_inbox()
    phoenix = [
        build_message(
            uid="201",
            gm_msgid="9101",
            subject="Phoenix status",
            message_id="<phoenix@example.com>",
            body="Phoenix is green.",
            internaldate=datetime(2026, 11, 16, 11, 0, tzinfo=UTC),
        )
    ]
    sent = [
        build_message(
            uid="301",
            gm_msgid="9201",
            subject="Sent only",
            message_id="<sent@example.com>",
            body="This mailbox is not allowlisted.",
            internaldate=datetime(2026, 11, 16, 12, 0, tzinfo=UTC),
        )
    ]
    everything = [*inbox, *phoenix, *sent]
    return (
        FakeImap()
        .with_mailbox("INBOX", *inbox)
        .with_mailbox("project-phoenix", *phoenix)
        .with_mailbox("SENT", *sent)
        .with_mailbox("[Gmail]/All Mail", *everything)
    )
