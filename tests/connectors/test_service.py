"""Service dispatch, environment guards, and the Notion stub."""

from __future__ import annotations

import json

import pytest
from gmail_fixtures import APP_PASSWORD, OWNER, make_gmail_settings
from jira_fixtures import FIXED_NOW, JiraStub, load_json, make_settings

from connectors import service
from connectors.base import (
    ConnectorAuthError,
    ConnectorBadRequestError,
    ConnectorConfigError,
    ConnectorError,
    ConnectorServerError,
)
from connectors.contracts import ErrorReport, SyncRequest, SyncResult
from connectors.gmail import (
    DEFAULT_ALLOWED_LABELS,
    GmailConnector,
    gmail_message_is_in_scope,
    is_label_allowed,
)
from connectors.jira import JiraConnector
from connectors.normalizer import CanonicalDoc
from connectors.notion import NotionConnector
from db.tracing import InMemoryEventSink, Tracer


def memory_tracer(trace_id: str | None = None) -> tuple[Tracer, InMemoryEventSink]:
    """Tracer over an in-memory sink — service tests must never touch the DB."""
    sink = InMemoryEventSink()
    return Tracer(sink, trace_id), sink


# --- stubs ----------------------------------------------------------------
def test_gmail_connector_requires_credentials() -> None:
    """Gmail is a real IMAP adapter now, so it fails closed without credentials.

    This replaces the M5 stub test (``NotImplementedError``): the stub is gone,
    and what has to hold instead is that an unconfigured connector reports the
    missing environment variables and opens no connection.
    """
    with pytest.raises(ConnectorConfigError, match="GMAIL_ADDRESS"):
        GmailConnector(None, make_gmail_settings(gmail_address="", gmail_app_password=None))


async def test_notion_sync_is_not_implemented_and_fetches_nothing() -> None:
    connector = NotionConnector()
    with pytest.raises(NotImplementedError, match="M2"):
        await connector.sync(SyncRequest(source="notion"))
    with pytest.raises(NotImplementedError, match="M2"):
        await connector.fetch_targeted(source_id="page-1")
    await connector.aclose()


def test_gmail_label_scoping_is_fail_closed() -> None:
    allowed = DEFAULT_ALLOWED_LABELS
    assert allowed == ("INBOX",)
    assert is_label_allowed(["INBOX"], allowed) is True
    assert is_label_allowed(["inbox"], allowed) is True  # case-insensitive
    assert is_label_allowed(["INBOX", "project-phoenix"], ("INBOX", "project-phoenix")) is True
    assert is_label_allowed(["SENT"], allowed) is False
    assert is_label_allowed([], allowed) is False
    assert is_label_allowed(["INBOX"], ()) is False  # empty allowlist allows nothing


def test_gmail_message_outside_allowlist_is_out_of_scope() -> None:
    inside = {"id": "1", "labelIds": ["INBOX", "UNREAD"]}
    outside = {"id": "2", "labelIds": ["SENT", "IMPORTANT"]}
    no_labels = {"id": "3"}

    allowed = ("INBOX", "project-phoenix")
    assert gmail_message_is_in_scope(inside, allowed) is True
    assert gmail_message_is_in_scope(outside, allowed) is False
    assert gmail_message_is_in_scope(no_labels, allowed) is False


async def test_service_gmail_dispatch_is_a_config_error_when_unconfigured() -> None:
    """``service.sync`` reaches the real adapter, which refuses before any IMAP work.

    Replaces the old stub assertion (``NotImplementedError``): Gmail is wired,
    so the interesting behaviour is that an unconfigured source fails closed with
    the missing variable names instead of attempting a connection.
    """
    tracer, _ = memory_tracer()
    with pytest.raises(ConnectorConfigError, match="Gmail is not configured"):
        await service.sync(
            SyncRequest(source="gmail"),
            settings=make_gmail_settings(gmail_address="", gmail_app_password=None),
            tracer=tracer,
        )


async def test_service_stub_dispatch_raises_for_notion() -> None:
    tracer, _ = memory_tracer()
    with pytest.raises(NotImplementedError, match="M2"):
        await service.sync(SyncRequest(source="notion"), tracer=tracer)


# --- dispatch -------------------------------------------------------------
class FakeConnector:
    source = "jira"

    def __init__(self) -> None:
        self.closed = False
        self.requests: list[SyncRequest] = []

    async def sync(self, req: SyncRequest) -> SyncResult:
        self.requests.append(req)
        return SyncResult(
            source="jira",
            documents_upserted=2,
            watermark=FIXED_NOW,
            errors=[ErrorReport(source="jira", stage="normalize", error_type="X", message="y")],
            full=req.full,
        )

    async def fetch_targeted(
        self, *, source_id: str | None = None, query: str | None = None, limit: int = 10
    ) -> list[CanonicalDoc]:
        return []

    async def aclose(self) -> None:
        self.closed = True


async def test_service_delegates_to_injected_connector() -> None:
    connector = FakeConnector()
    tracer, sink = memory_tracer()
    result = await service.sync(
        SyncRequest(source="jira", full=True), connector=connector, tracer=tracer
    )

    assert connector.requests == [SyncRequest(source="jira", full=True)]
    assert result.documents_upserted == 2
    assert result.errors[0].error_type == "X"
    assert result.ok is False
    assert result.trace_id == tracer.trace_id
    assert [event.event for event in sink.events_for(tracer.trace_id)] == [
        "connector.sync.started",
        "connector.sync.completed",
    ]


async def test_service_unknown_source_is_rejected() -> None:
    with pytest.raises(ConnectorConfigError, match="unknown source"):
        service.build_connector("slack")
    with pytest.raises(ConnectorConfigError, match="unknown source"):
        service.build_connector("")


async def test_service_jira_sync_without_database_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service.db_session, "get_session_factory", lambda: None)
    tracer, sink = memory_tracer()
    with pytest.raises(ConnectorConfigError, match="DATABASE_URL is not set"):
        await service.sync(SyncRequest(source="jira"), settings=make_settings(), tracer=tracer)

    # The failure is still auditable (and no database was needed to record it).
    assert [event.event for event in sink.events_for(tracer.trace_id)] == [
        "connector.sync.started",
        "connector.sync.failed",
    ]


async def test_build_connector_returns_jira_adapter() -> None:
    connector = service.build_connector("jira", settings=make_settings())
    assert isinstance(connector, JiraConnector)
    await connector.aclose()


async def test_build_connector_returns_gmail_adapter() -> None:
    """Fetch-only wiring: a targeted fetch needs no database, so no store is passed."""
    connector = service.build_connector("gmail", settings=make_gmail_settings())
    assert isinstance(connector, GmailConnector)
    await connector.aclose()


async def test_fetch_targeted_service_path_uses_jira_adapter() -> None:
    stub = (
        JiraStub()
        .with_issue(load_json("issue_phx_102.json"))
        .with_fixture_comments("PHX-102")
    )
    tracer, _ = memory_tracer()
    docs = await service.fetch_targeted(
        "jira",
        "PHX-102",
        transport=stub.transport,
        settings=make_settings(),
        tracer=tracer,
    )

    assert [doc.source_id for doc in docs] == ["PHX-102"]
    assert docs[0].doc_type.value == "issue"
    assert all(doc.source.value == "jira" for doc in docs)


async def test_fetch_targeted_gmail_needs_source_id_and_notion_raises() -> None:
    tracer, _ = memory_tracer()
    with pytest.raises(ConnectorBadRequestError, match="query"):
        await service.fetch_targeted(
            "gmail", query="from:ada", settings=make_gmail_settings(), tracer=tracer
        )
    with pytest.raises(NotImplementedError, match="M2"):
        await service.fetch_targeted("notion", "page-1", tracer=tracer)


# --- audit events (docs/08-decisions.md, integration gap #2) ---------------
async def test_sync_emits_started_and_completed_audit_events() -> None:
    tracer, sink = memory_tracer()
    connector = FakeConnector()

    result = await service.sync(SyncRequest(source="jira", full=True), connector=connector,
                                tracer=tracer)

    events = sink.events_for(tracer.trace_id)
    assert [event.event for event in events] == [
        "connector.sync.started",
        "connector.sync.completed",
    ]
    assert all(event.agent == "connector" for event in events)
    assert events[0].payload == {"source": "jira", "full": True}

    completed = events[1]
    assert completed.payload["source"] == "jira"
    assert completed.payload["full"] is True
    assert completed.payload["partial"] is True  # the fake reports one error
    assert completed.payload["errors"] == 1
    assert completed.payload["error_types"] == ["X"]
    assert completed.payload["watermark"] == FIXED_NOW.isoformat()
    assert completed.payload["documents_upserted"] == 2
    assert completed.latency_ms is not None and completed.latency_ms >= 0
    assert result.trace_id == tracer.trace_id


# --- answer-cache invalidation (docs/08-decisions.md, ruling #14) ---------
async def test_sync_invalidates_the_answer_cache_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def invalidate() -> int:
        calls.append("invalidate")
        return 7

    monkeypatch.setattr(service, "invalidate_answer_cache", invalidate)
    tracer, sink = memory_tracer()
    result = await service.sync(
        SyncRequest(source="jira"), connector=FakeConnector(), tracer=tracer
    )

    assert calls == ["invalidate"]
    assert result.documents_upserted == 2
    # The version travels in the completed event, so a trace explains why a
    # cached answer went missing.
    completed = sink.events_for(tracer.trace_id)[-1]
    assert completed.event == "connector.sync.completed"
    assert completed.payload["answer_cache_version"] == 7


async def test_sync_completes_when_invalidation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documents are already committed: a cache failure cannot undo them."""

    async def invalidate() -> int:
        raise ConnectionError("redis is unreachable")

    monkeypatch.setattr(service, "invalidate_answer_cache", invalidate)
    tracer, sink = memory_tracer()
    result = await service.sync(
        SyncRequest(source="jira"), connector=FakeConnector(), tracer=tracer
    )

    assert result.documents_upserted == 2
    assert result.ok is False  # the fake still reports its per-item error
    completed = sink.events_for(tracer.trace_id)[-1]
    assert completed.event == "connector.sync.completed"
    assert completed.payload["answer_cache_version"] is None


async def test_noop_sync_keeps_the_warm_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheduled sync that changed nothing must not bump the version."""

    calls: list[str] = []

    async def invalidate() -> int:
        calls.append("invalidate")
        return 8

    class NoopConnector(FakeConnector):
        async def sync(self, req: SyncRequest) -> SyncResult:
            return SyncResult(source="jira", watermark=FIXED_NOW, full=req.full)

    monkeypatch.setattr(service, "invalidate_answer_cache", invalidate)
    tracer, sink = memory_tracer()
    await service.sync(SyncRequest(source="jira"), connector=NoopConnector(), tracer=tracer)

    assert calls == [], "a no-op sync must not throw away a warm cache"
    completed = sink.events_for(tracer.trace_id)[-1]
    assert completed.event == "connector.sync.completed"
    assert completed.payload["answer_cache_version"] is None


async def test_sync_failure_emits_failed_event_with_retryable_flag() -> None:
    class AuthFailingConnector(FakeConnector):
        async def sync(self, req: SyncRequest) -> SyncResult:
            raise ConnectorAuthError("credentials rejected (HTTP 401)")

    tracer, sink = memory_tracer()
    with pytest.raises(ConnectorAuthError):
        await service.sync(
            SyncRequest(source="jira"), connector=AuthFailingConnector(), tracer=tracer
        )

    events = sink.events_for(tracer.trace_id)
    assert [event.event for event in events] == [
        "connector.sync.started",
        "connector.sync.failed",
    ]
    failed = events[1].payload
    assert failed["error_type"] == "ConnectorAuthError"
    assert "credentials rejected" in failed["error"]
    assert failed["retryable"] is False  # terminal: a human must fix credentials
    assert failed["source"] == "jira"


async def test_sync_retryable_failure_is_flagged_retryable() -> None:
    class RateLimitedConnector(FakeConnector):
        async def sync(self, req: SyncRequest) -> SyncResult:
            raise ConnectorServerError("source error (HTTP 503)")

    tracer, sink = memory_tracer()
    with pytest.raises(ConnectorServerError):
        await service.sync(
            SyncRequest(source="jira"), connector=RateLimitedConnector(), tracer=tracer
        )

    assert sink.records[-1].payload["retryable"] is True


async def test_sync_honours_supplied_trace_id() -> None:
    tracer, sink = memory_tracer("trace-from-orchestrator")
    result = await service.sync(
        SyncRequest(source="jira"), connector=FakeConnector(), tracer=tracer
    )

    assert result.trace_id == "trace-from-orchestrator"
    assert {event.trace_id for event in sink.records} == {"trace-from-orchestrator"}


async def test_fetch_targeted_emits_its_own_audit_events() -> None:
    stub = (
        JiraStub().with_issue(load_json("issue_phx_102.json")).with_fixture_comments("PHX-102")
    )
    tracer, sink = memory_tracer()
    docs = await service.fetch_targeted(
        "jira", "PHX-102", transport=stub.transport, settings=make_settings(), tracer=tracer
    )

    assert len(docs) == 1
    events = sink.events_for(tracer.trace_id)
    assert [event.event for event in events] == [
        "connector.fetch.started",
        "connector.fetch.completed",
    ]
    assert events[0].payload == {
        "source": "jira",
        "source_id": "PHX-102",
        "query": None,
        "limit": 10,
    }
    assert events[1].payload["documents"] == 1
    assert events[1].payload["requests"] >= 1
    assert events[1].latency_ms is not None


async def test_fetch_targeted_failure_is_audited() -> None:
    stub = JiraStub()  # no issue registered -> 404 from the fake source
    tracer, sink = memory_tracer()
    with pytest.raises(ConnectorError):
        await service.fetch_targeted(
            "jira", "PHX-999", transport=stub.transport, settings=make_settings(), tracer=tracer
        )

    events = sink.events_for(tracer.trace_id)
    assert [event.event for event in events] == [
        "connector.fetch.started",
        "connector.fetch.failed",
    ]
    assert events[1].payload["error_type"] == "ConnectorNotFoundError"


# --- config / diagnostics -------------------------------------------------
def test_describe_settings_never_exposes_secrets() -> None:
    settings = make_settings(jira_api_token="super-secret-token-value")
    described = service.describe_settings(settings)

    assert described["jira_configured"] is True
    assert described["jira_projects"] == ["PHX"]
    assert "super-secret-token-value" not in json.dumps(described)
    assert "token" not in json.dumps(described).lower()


def test_describe_settings_reports_gmail_without_the_app_password() -> None:
    described = service.describe_settings(make_gmail_settings())

    assert described["gmail_configured"] is True
    assert described["gmail_address"] == OWNER  # an address, not a secret
    assert described["gmail_labels"] == ["INBOX", "project-phoenix"]
    assert APP_PASSWORD not in json.dumps(described)


def test_gmail_credentials_repr_redacts_the_app_password() -> None:
    credentials = make_gmail_settings().require_gmail()
    assert APP_PASSWORD not in repr(credentials)
    assert "SecretStr('**********')" in repr(credentials)


def test_gmail_is_not_configured_without_a_password() -> None:
    assert make_gmail_settings().gmail_configured is True
    assert make_gmail_settings(gmail_address="").gmail_configured is False
    assert make_gmail_settings(gmail_app_password=None).gmail_configured is False
    assert make_gmail_settings(gmail_app_password="").gmail_configured is False


def test_jira_credentials_repr_redacts_token() -> None:
    credentials = make_settings(jira_api_token="super-secret-token-value").require_jira()
    assert "super-secret-token-value" not in repr(credentials)
    assert "SecretStr('**********')" in repr(credentials)
    assert "super-secret-token-value" not in str(credentials)


def test_missing_jira_config_lists_the_missing_variables() -> None:
    with pytest.raises(ConnectorConfigError) as excinfo:
        make_settings(jira_projects="").require_jira()
    assert "JIRA_PROJECTS" in str(excinfo.value)

    with pytest.raises(ConnectorConfigError) as excinfo:
        make_settings(jira_base_url="", jira_email="").require_jira()
    assert "JIRA_BASE_URL" in str(excinfo.value)
    assert "JIRA_EMAIL" in str(excinfo.value)


def test_jira_auth_header_is_basic_and_not_logged() -> None:
    connector = JiraConnector(settings=make_settings(), store=None, transport=JiraStub().transport)
    headers = connector._headers
    assert headers["Authorization"].startswith("Basic ")
    assert "test-token-not-real" not in repr(connector)
    assert connector.source == "jira"

