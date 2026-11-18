"""Jira sync behaviour: full/incremental, hashes, comments, ACL, retirement, errors.

Everything runs offline: ``JiraStub`` (``httpx.MockTransport``) serves the
hand-built payloads, and ``InMemoryDocumentStore`` implements the same
persistence port the SQLAlchemy store does.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import httpx
import pytest
from jira_fixtures import FIXED_NOW, JiraStub, issue_from, load_json, make_settings

from connectors.base import ConnectorAuthError, ConnectorConfigError
from connectors.contracts import SyncRequest
from connectors.jira import JiraConnector
from connectors.normalizer import (
    CanonicalDoc,
    InMemoryDocumentStore,
    apply_document,
    jira_issue_to_docs,
)
from db.models import DocType, Source


def connector_for(
    stub: JiraStub,
    store: InMemoryDocumentStore,
    **settings_overrides: object,
) -> JiraConnector:
    return JiraConnector(
        store=store,
        settings=make_settings(**settings_overrides),
        transport=stub.transport,
        sleep=_no_sleep,
        now=lambda: FIXED_NOW,
    )


async def _no_sleep(_seconds: float) -> None:
    return None


def doc_by_source_id(store: InMemoryDocumentStore, source_id: str) -> list[CanonicalDoc]:
    """Current rows for a native id."""
    return [doc for doc in store.rows.values() if doc.source_id == source_id and doc.is_current]


def all_rows_for(store: InMemoryDocumentStore, source_id: str) -> list[CanonicalDoc]:
    """Every version of an item: the current row plus all history rows.

    History rows keep the native ``source_id`` — only *current* rows are uniquely
    constrained (partial index, migration 0003).
    """
    return [doc for doc in store.rows.values() if doc.source_id == source_id]


# --- full sync ------------------------------------------------------------
async def test_full_sync_upserts_issues_and_comments(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # 3 issues + 3 comments
    assert result.documents_upserted == 6
    assert result.documents_unchanged == 0
    assert result.documents_retired == 0
    assert result.errors == []
    assert result.full is True
    assert result.watermark == datetime(2026, 11, 16, 14, 22, tzinfo=UTC)
    assert result.requests > 0

    assert len([doc for doc in store.rows.values() if doc.is_current]) == 6
    issue = doc_by_source_id(store, "PHX-101")[0]
    assert issue.doc_type is DocType.issue
    assert "Status: In Progress" in issue.content

    state = store.sync_state["jira"]
    assert state.last_status == "ok"
    assert state.watermark == datetime(2026, 11, 16, 14, 22, tzinfo=UTC)
    assert state.error is None


async def test_search_pages_are_followed(stub: JiraStub, store: InMemoryDocumentStore) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    starts = [request.url.params.get("startAt") for request in stub.calls_to("/rest/api/3/search")]
    assert starts == [None, "2"]  # first page needs no offset
    assert stub.calls_to("/rest/api/3/issue/PHX-101/comment")
    assert stub.calls_to("/rest/api/3/issue/PHX-102/comment")
    assert stub.calls_to("/rest/api/3/issue/PHX-103/comment")


async def test_token_pagination_shape_is_supported(store: InMemoryDocumentStore) -> None:
    first = issue_from("search_page_1.json", 0)
    second = issue_from("search_page_1.json", 1)
    stub = (
        JiraStub()
        .with_search_payloads(
            {"issues": [first], "nextPageToken": "tok-2", "isLast": False},
            {"issuedToken": "tok-2", "issues": [second], "isLast": True},
        )
        .with_fixture_comments("PHX-101")
        .with_fixture_comments("PHX-102")
    )

    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert result.errors == []
    assert {doc.source_id for doc in store.rows.values() if doc.doc_type is DocType.issue} == {
        "PHX-101",
        "PHX-102",
    }
    tokens = [
        request.url.params.get("nextPageToken")
        for request in stub.calls_to("/rest/api/3/search")
    ]
    assert tokens == [None, "tok-2"]


async def test_out_of_allowlist_project_is_skipped(store: InMemoryDocumentStore) -> None:
    """The allowlist is enforced on the payload, not just in the JQL string."""
    issue = issue_from("search_page_1.json", 0)
    issue["fields"]["project"] = {"key": "OPS", "name": "Operations", "id": "10099"}
    stub = (
        JiraStub()
        .with_search_payloads({"startAt": 0, "maxResults": 50, "total": 1, "issues": [issue]})
        .with_fixture_comments("PHX-101")
    )

    connector = connector_for(stub, store)  # allowlist is PHX
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert result.documents_skipped == 1
    assert result.documents_upserted == 0
    assert result.errors == []
    assert store.rows == {}
    assert not stub.calls_to("/rest/api/3/issue/PHX-101/comment")  # never even fetched


# --- incremental ----------------------------------------------------------
async def test_incremental_sync_advances_watermark(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    stub.with_search_payloads(load_json("search_updated_page.json"))
    stub.with_comments("PHX-101", *load_json("comments_phx_101_updated.json")["comments"])

    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert result.full is False
    assert result.watermark == datetime(2026, 11, 17, 8, 0, tzinfo=UTC)
    assert store.sync_state["jira"].watermark == datetime(2026, 11, 17, 8, 0, tzinfo=UTC)

    jqls = stub.search_jqls()
    assert 'updated >= "2026-11-16 14:22"' in jqls[-1]
    assert "project IN (\"PHX\")" in jqls[-1]
    assert jqls[-1].endswith("ORDER BY updated ASC")


async def test_full_sync_is_requested_explicitly_and_has_no_cursor(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira", full=True))
    finally:
        await connector.aclose()

    assert result.full is True
    assert "updated >=" not in stub.search_jqls()[-1]


async def test_unchanged_content_is_skipped(stub: JiraStub, store: InMemoryDocumentStore) -> None:
    connector = connector_for(stub, store)
    try:
        first = await connector.sync(SyncRequest(source="jira"))
        rows_after_first = len(store.rows)
        second = await connector.sync(SyncRequest(source="jira", full=True))
    finally:
        await connector.aclose()

    assert first.documents_upserted == 6
    assert second.documents_upserted == 0
    assert second.documents_unchanged == 6
    assert len(store.rows) == rows_after_first  # no new rows, no rewrites
    assert second.errors == []


# --- supersession ---------------------------------------------------------
async def test_changed_issue_supersedes_previous_row(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    stub.with_search_payloads(load_json("search_updated_page.json"))
    stub.with_fixture_comments("PHX-101")  # comments unchanged in this scenario
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    rows = all_rows_for(store, "PHX-101")
    assert len(rows) == 2
    assert sum(1 for doc in rows if doc.is_current) == 1
    current = next(doc for doc in rows if doc.is_current)
    previous = next(doc for doc in rows if not doc.is_current)
    assert "Status: Done" in current.content
    assert "Status: In Progress" in previous.content
    assert previous.is_current is False  # kept for audit, never deleted
    assert previous.source_id == "PHX-101"  # native id, no suffix
    assert result.documents_upserted == 1
    assert result.documents_retired == 0


async def test_deleted_comment_is_retired_not_deleted(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    stub.with_search_payloads(load_json("search_updated_page.json"))
    stub.with_comments("PHX-101", *load_json("comments_phx_101_updated.json")["comments"])
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # Comment 20102 was removed from the source.
    retired = all_rows_for(store, "PHX-101/comment/20102")
    assert len(retired) == 1  # still present ...
    assert retired[0].is_current is False  # ... but no longer current
    assert result.documents_retired == 1

    # Comment 20101 changed: new current row, old one superseded.
    edited = all_rows_for(store, "PHX-101/comment/20101")
    assert len(edited) == 2
    assert sum(1 for doc in edited if doc.is_current) == 1


async def test_last_comment_deleted_retires_it(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # PHX-103's only comment is gone; the issue itself is unchanged.
    page = load_json("search_page_2.json")
    page["startAt"] = 0
    page["total"] = 1
    stub.with_search_payloads(page)
    stub.with_comments("PHX-103")
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert result.documents_retired == 1
    assert doc_by_source_id(store, "PHX-103/comment/20103") == []
    assert doc_by_source_id(store, "PHX-103") != []


async def test_third_revision_keeps_full_history(store: InMemoryDocumentStore) -> None:
    """Two successive changes: the frozen unique constraint must not be violated."""
    v1 = issue_from("search_page_1.json", 0)
    v2 = issue_from("search_updated_page.json", 0)
    v3 = copy.deepcopy(v2)
    v3["fields"]["summary"] = "Write connector retry/backoff policy (follow-up)"
    v3["fields"]["updated"] = "2026-11-18T10:00:00.000+0000"

    docs = [
        jira_issue_to_docs(v1, base_url="https://acme.atlassian.net", fetched_at=FIXED_NOW)[0],
        jira_issue_to_docs(v2, base_url="https://acme.atlassian.net", fetched_at=FIXED_NOW)[0],
        jira_issue_to_docs(v3, base_url="https://acme.atlassian.net", fetched_at=FIXED_NOW)[0],
    ]
    for doc in docs:
        await apply_document(store, doc)

    rows = [doc for doc in store.rows.values() if doc.doc_type is DocType.issue]
    assert len(rows) == 3
    assert sum(1 for doc in rows if doc.is_current) == 1
    assert {doc.is_current for doc in rows} == {True, False}
    # Every revision shares the native source_id; only current rows are unique.
    assert {doc.source_id for doc in rows} == {"PHX-101"}



async def test_repeated_changes_keep_history_under_native_id(
    store: InMemoryDocumentStore,
) -> None:
    """Regression: 1 current row + 2 history rows, all with the native source_id.

    Three sync runs (v1, v2, v3) and a fourth unchanged run. The old
    ``UNIQUE (source, source_id, is_current)`` constraint forced a ``#history-``
    suffix on revisions; with the partial unique index they all keep ``PHX-101``.
    """
    v1 = issue_from("search_page_1.json", 0)
    v2 = issue_from("search_updated_page.json", 0)
    v3 = copy.deepcopy(v2)
    v3["fields"]["summary"] = "Write connector retry/backoff policy (follow-up)"
    v3["fields"]["updated"] = "2026-11-18T10:00:00.000+0000"

    def page_for(issue: dict[str, object]) -> dict[str, object]:
        return {"startAt": 0, "maxResults": 50, "total": 1, "issues": [issue]}

    stub = JiraStub().with_fixture_comments("PHX-101")
    results = []
    for issue in (v1, v2, v3, v3):
        stub.with_search_payloads(page_for(issue))
        connector = connector_for(stub, store)
        try:
            results.append(await connector.sync(SyncRequest(source="jira")))
        finally:
            await connector.aclose()

    # Run 3 changes the issue *summary*, which comment docs embed as their
    # "Issue: PHX-101 — <summary>" line, so the two comments are rewritten too.
    assert [r.documents_upserted for r in results] == [3, 1, 3, 0]
    assert results[-1].documents_unchanged == 3

    rows = all_rows_for(store, "PHX-101")
    assert len(rows) == 3  # 1 current + 2 history
    assert sum(1 for doc in rows if doc.is_current) == 1
    assert {doc.source_id for doc in rows} == {"PHX-101"}  # native id everywhere

    # Lookups return the current row, and it is the newest content.
    record = await store.find_current(Source.jira, "PHX-101")
    assert record is not None
    current = next(doc for doc in rows if doc.is_current)
    assert record.id in {row_id for row_id, doc in store.rows.items() if doc.is_current}
    assert "follow-up" in current.content
    assert current.content_hash == record.content_hash
    # History is intact and ordered oldest → newest by content.
    history = [doc.content for doc in rows if not doc.is_current]
    assert any("Status: In Progress" in content for content in history)
    assert any("Status: Done" in content for content in history)


async def test_deleted_issue_and_children_retired_on_full_sync(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # Only PHX-101 still exists in the source.
    stub.with_search_payloads(load_json("search_updated_page.json"))
    stub.with_comments("PHX-101", *load_json("comments_phx_101.json")["comments"])
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira", full=True))
    finally:
        await connector.aclose()

    # PHX-102 and PHX-103, plus PHX-103's comment.
    assert result.documents_retired == 3
    assert doc_by_source_id(store, "PHX-102") == []
    assert doc_by_source_id(store, "PHX-103") == []
    assert doc_by_source_id(store, "PHX-103/comment/20103") == []
    assert doc_by_source_id(store, "PHX-101") != []
    # Nothing was hard-deleted.
    assert len(store.rows) == 7


async def test_incremental_sync_never_retires_scope(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # An incremental page that mentions only PHX-101 must not look like a deletion.
    stub.with_search_payloads(load_json("search_updated_page.json"))
    stub.with_comments("PHX-101", *load_json("comments_phx_101.json")["comments"])
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert result.documents_retired == 0
    assert doc_by_source_id(store, "PHX-102") != []


# --- failure handling -----------------------------------------------------
async def test_malformed_item_is_reported_and_does_not_abort(
    store: InMemoryDocumentStore,
) -> None:
    stub = JiraStub().with_search_payloads(load_json("malformed_issue_page.json"))
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.source == "jira"
    assert error.source_id == "PHX-199"
    assert error.stage == "normalize"
    assert error.error_type == "ValueError"
    assert "fields" in error.message

    assert result.documents_upserted == 1
    assert doc_by_source_id(store, "PHX-198") != []
    assert store.sync_state["jira"].last_status == "partial"


async def test_failed_item_is_not_treated_as_deleted(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    connector = connector_for(stub, store)
    try:
        await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    # A comment without an id makes PHX-102 fail to normalize.
    stub.with_comments("PHX-102", {"body": "no id here"})
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira", full=True))
    finally:
        await connector.aclose()

    assert [error.source_id for error in result.errors] == ["PHX-102"]
    assert result.documents_retired == 0
    assert doc_by_source_id(store, "PHX-102") != []  # previous doc untouched


async def test_watermark_clamped_before_failed_item(store: InMemoryDocumentStore) -> None:
    stub = JiraStub().with_search_payloads(
        load_json("search_page_1.json"), load_json("search_page_2.json")
    )
    # PHX-102 (updated 11:40) fails; PHX-101 (09:12) and PHX-103 (14:22) succeed.
    stub.with_comments("PHX-102", {"body": "no id here"})
    connector = connector_for(stub, store)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert [error.source_id for error in result.errors] == ["PHX-102"]
    # Cursor stops just before the failed item so its next change is not skipped.
    assert result.watermark == datetime(2026, 11, 16, 11, 39, 59, tzinfo=UTC)


async def test_auth_error_propagates_and_records_state(store: InMemoryDocumentStore) -> None:
    stub = JiraStub().with_search_payloads(load_json("search_page_1.json"))
    stub.fail("/rest/api/3/search", httpx.Response(401, text="invalid token"))
    connector = connector_for(stub, store)
    try:
        with pytest.raises(ConnectorAuthError):
            await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    state = store.sync_state["jira"]
    assert state.last_status == "error"
    assert "credentials rejected" in (state.error or "")


async def test_budget_exhaustion_returns_partial_result(
    stub: JiraStub, store: InMemoryDocumentStore
) -> None:
    # Search page 1 (1 request) + PHX-101 comments (2nd) exhausts the budget.
    connector = connector_for(stub, store, jira_rate_budget_calls=2)
    try:
        result = await connector.sync(SyncRequest(source="jira"))
    finally:
        await connector.aclose()

    assert [error.error_type for error in result.errors] == ["ConnectorRateBudgetExhausted"]
    assert result.errors[0].stage == "fetch"
    assert result.errors[0].retryable is False
    # Whatever completed before the budget ran out is kept.
    assert result.documents_upserted == 3  # PHX-101 + its 2 comments
    assert result.watermark == datetime(2026, 11, 16, 9, 12, tzinfo=UTC)


# --- targeted fetches -----------------------------------------------------
async def test_fetch_targeted_by_source_id(store: InMemoryDocumentStore) -> None:
    stub = (
        JiraStub()
        .with_issue(issue_from("search_page_1.json", 0))
        .with_fixture_comments("PHX-101")
    )
    connector = connector_for(stub, store)
    try:
        docs = await connector.fetch_targeted(source_id="PHX-101")
    finally:
        await connector.aclose()

    assert [doc.doc_type for doc in docs] == [
        DocType.issue,
        DocType.issue_comment,
        DocType.issue_comment,
    ]
    assert docs[0].source_id == "PHX-101"
    assert docs[1].parent_source_id == "PHX-101"
    assert store.rows == {}  # targeted fetch never writes


async def test_fetch_targeted_by_query_scopes_to_projects(store: InMemoryDocumentStore) -> None:
    stub = (
        JiraStub()
        .with_search_payloads(load_json("search_page_1.json"))
        .with_fixture_comments("PHX-101")
        .with_fixture_comments("PHX-102")
    )
    connector = connector_for(stub, store)
    try:
        docs = await connector.fetch_targeted(query="duedate <= 2026-11-20", limit=2)
    finally:
        await connector.aclose()

    jql = stub.search_jqls()[0]
    assert 'project IN ("PHX")' in jql
    assert "(duedate <= 2026-11-20)" in jql
    assert len(docs) == 2
    assert store.rows == {}


async def test_fetch_targeted_rejects_injection_attempt(store: InMemoryDocumentStore) -> None:
    connector = connector_for(JiraStub(), store)
    try:
        with pytest.raises(ConnectorConfigError, match="invalid Jira issue key"):
            await connector.fetch_targeted(source_id='PHX-1") OR project != "PHX')
    finally:
        await connector.aclose()


async def test_fetch_targeted_requires_a_selector(store: InMemoryDocumentStore) -> None:
    connector = connector_for(JiraStub(), store)
    try:
        with pytest.raises(ConnectorConfigError, match="needs either source_id or query"):
            await connector.fetch_targeted()
    finally:
        await connector.aclose()


# --- configuration --------------------------------------------------------
async def test_project_keys_are_validated(store: InMemoryDocumentStore) -> None:
    with pytest.raises(ConnectorConfigError, match="invalid Jira project key"):
        connector_for(JiraStub(), store, jira_projects='PHX") OR ("1')


async def test_missing_credentials_are_reported_by_name(store: InMemoryDocumentStore) -> None:
    with pytest.raises(ConnectorConfigError, match="JIRA_API_TOKEN"):
        JiraConnector(
            store=store,
            settings=make_settings(jira_api_token=""),
            transport=JiraStub().transport,
        )

