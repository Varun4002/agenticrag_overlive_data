"""Operator dashboard tests.

Everything here runs without a database: the read endpoints must degrade with an
explicit 503 instead of crashing, and the mutating endpoints must refuse before
touching anything. Database-backed behaviour is exercised by hand against a
scratch database (see docs/09-operator-dashboard.md).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import indexing.indexer as indexing_indexer
from api import admin
from api.main import UI_FILE, create_app
from config import Settings, get_settings
from connectors.settings import get_connector_settings
from db import session as db_session
from indexing.indexer import IndexRequest, IndexResult
from memory.answer_cache import reset_answer_cache


@pytest.fixture
def no_jira(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the connector looks unconfigured for this test."""
    for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECTS"):
        monkeypatch.delenv(name, raising=False)
    get_connector_settings.cache_clear()


def _fresh_settings() -> None:
    get_settings.cache_clear()
    get_connector_settings.cache_clear()
    db_session.reset_engine_state()


def test_dashboard_asset_exists() -> None:
    """The single-file UI ships with the package data."""
    assert UI_FILE.is_file(), f"missing dashboard asset: {UI_FILE}"
    html = UI_FILE.read_text(encoding="utf-8")
    assert "operator dashboard" in html
    # The Actions tab covers every admin mutation, including the answer cache.
    assert "/admin/cache/invalidate" in html, "the clear-cache button must call the cache route"
    assert 'id="cache-run"' in html, "no clear-cache button in the dashboard"
    assert "Jira sync" not in html, "the sync card is source-agnostic, not Jira-only"
    assert 'id="sync-source"' in html, "the source selector must stay on the sync card"


def test_query_box_is_the_landing_page() -> None:
    """Regression guard: the question box must be the first thing on screen."""
    html = UI_FILE.read_text(encoding="utf-8")
    assert "id=\"search-q\"" in html, "no question input in the dashboard"
    assert '<section class="tab active" id="tab-query">' in html, (
        "the Ask tab must be the active section on load"
    )
    # Ask is the first tab in the nav, ahead of overview.
    assert html.index('data-tab="query"') < html.index('data-tab="overview"')


def test_dashboard_served(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "operator dashboard" in response.text
    # Self-contained: no CDN/build step, so the page works offline.
    assert 'src="http' not in response.text
    assert 'href="http' not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_root_redirects_to_dashboard(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/ui"


def test_admin_disabled_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADMIN_ENABLED=false closes both the page and the JSON API."""
    monkeypatch.setenv("ADMIN_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "")
    _fresh_settings()
    app = create_app()
    try:
        with TestClient(app) as client:
            assert client.get("/ui").status_code == 403
            assert client.get("/admin/overview").status_code == 403
            assert client.get("/health").status_code == 200  # frozen path unaffected
    finally:
        _fresh_settings()


def test_admin_enabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit registration wins over the environment default."""
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ADMIN_ENABLED", "true")
    _fresh_settings()
    try:
        assert get_settings().admin_available is True
    finally:
        _fresh_settings()


def test_admin_default_is_off_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    _fresh_settings()
    try:
        assert get_settings().admin_available is False
    finally:
        _fresh_settings()


def test_admin_default_is_on_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ENABLED", raising=False)
    settings = Settings(app_env="dev", database_url=None)
    assert settings.admin_available is True


def test_overview_without_database(client: TestClient) -> None:
    """Counts are null, configuration is still reported, no request fails."""
    body = client.get("/admin/overview").json()
    assert body["database"] == {"configured": False, "reachable": False}
    assert body["counts"] is None
    assert body["by_source"] == []
    assert body["answer_mode"] == "pipeline"
    assert body["config"]["embedding_dimensions"] == 768
    assert "jira_configured" in body["connectors"]
    # The default dev salt must be visible rather than silently trusted.
    assert body["config"]["pii_hash_salt_is_default"] is True


def test_documents_requires_database(client: TestClient) -> None:
    response = client.get("/admin/documents")
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_document_detail_rejects_bad_id(client: TestClient) -> None:
    # A malformed id is a client error even without a database.
    assert client.get("/admin/documents/not-a-uuid").status_code == 400


def test_document_detail_requires_database(client: TestClient) -> None:
    response = client.get("/admin/documents/2b7e4d19-8c63-4f20-a5b8-6d1e9c3a7f04")
    assert response.status_code == 503


def test_traces_requires_database(client: TestClient) -> None:
    assert client.get("/admin/traces").status_code == 503
    assert client.get("/admin/traces/some-trace").status_code == 503


def test_search_validates_input(client: TestClient) -> None:
    assert client.post("/admin/search", json={"query": "", "user_id": "u"}).status_code == 422
    assert (
        client.post("/admin/search", json={"query": "q", "user_id": "u", "top_k": 0}).status_code
        == 422
    )
    assert (
        client.post("/admin/search", json={"query": "q", "user_id": "u", "top_k": 51}).status_code
        == 422
    )


def test_search_requires_database(client: TestClient) -> None:
    response = client.post("/admin/search", json={"query": "cutover", "user_id": "u"})
    assert response.status_code == 503


def test_sync_reports_missing_jira_config(client: TestClient, no_jira: None) -> None:
    """Configuration is reported before the database check: the fix is credentials."""
    response = client.post("/admin/sync", json={"source": "jira"})
    assert response.status_code == 400
    assert "JIRA_BASE_URL" in response.json()["detail"]


def test_sync_refuses_unimplemented_sources(client: TestClient) -> None:
    response = client.post("/admin/sync", json={"source": "notion"})
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


def test_sync_requires_database(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://acme.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "dev@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "not-a-real-token")
    monkeypatch.setenv("JIRA_PROJECTS", "PHX")
    get_connector_settings.cache_clear()
    response = client.post("/admin/sync", json={"source": "jira"})
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_backfill_requires_database(client: TestClient) -> None:
    response = client.post("/admin/backfill", json={"limit": 10})
    assert response.status_code == 503


# --- answer-cache invalidation (docs/08-decisions.md, ruling #14) ----------
@pytest.fixture
def no_redis(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Guarantee "no answer cache" for this test: the singleton is process-wide."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()
    reset_answer_cache()
    yield
    reset_answer_cache()


async def _one_document(*_: object, **__: object) -> list[uuid.UUID]:
    return [uuid.uuid4()]


def test_cache_invalidate_without_redis_reports_no_cache(
    client: TestClient, no_redis: None
) -> None:
    """No database, no Redis: the endpoint explains the missing dependency."""
    response = client.post("/admin/cache/invalidate")

    assert response.status_code == 200
    body = response.json()
    assert body["invalidated"] is False
    assert "REDIS_URL" in body["reason"]


async def test_backfill_invalidation_records_no_version_without_a_cache(
    monkeypatch: pytest.MonkeyPatch, no_redis: None
) -> None:
    """The ``answer_cache_version`` write happens only after a page was indexed.

    No database is faked: the document lookup and the indexer are stubbed (which
    is also why the "no matching documents" early return is asserted separately
    — it leaves the cache untouched).
    """

    class FakeIndexer:
        async def run(self, req: IndexRequest) -> IndexResult:
            return IndexResult(chunks_written=1, chunks_skipped=0, embed_tokens=0)

    monkeypatch.setattr(admin, "_factory", lambda: object())
    monkeypatch.setattr(admin, "_select_document_ids", _one_document)
    monkeypatch.setattr(indexing_indexer, "get_indexer", lambda: FakeIndexer())

    body = await admin.backfill(admin.BackfillIn(limit=10), None)

    assert body["documents"] == 1
    assert body["chunks_written"] == 1
    assert body["answer_cache_version"] is None


async def test_backfill_without_documents_never_touches_the_cache(
    monkeypatch: pytest.MonkeyPatch, no_redis: None
) -> None:
    async def no_documents(*_: object, **__: object) -> list[uuid.UUID]:
        return []

    monkeypatch.setattr(admin, "_factory", lambda: object())
    monkeypatch.setattr(admin, "_select_document_ids", no_documents)

    body = await admin.backfill(admin.BackfillIn(limit=10), None)

    assert body["note"] == "no matching documents"
    # Nothing was written, so no version is bumped — and the response shape is
    # stable: the key is always present, None when nothing was invalidated.
    assert body["answer_cache_version"] is None


def test_seed_requires_confirmation(client: TestClient) -> None:
    response = client.post("/admin/seed", json={"confirm": False})
    assert response.status_code == 400
    assert "confirm" in response.json()["detail"]


def test_seed_refused_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ADMIN_ENABLED", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    _fresh_settings()
    app = create_app()
    try:
        with TestClient(app) as client:
            response = client.post("/admin/seed", json={"confirm": True})
            assert response.status_code == 403
            assert "prod" in response.json()["detail"]
    finally:
        monkeypatch.delenv("APP_ENV")
        _fresh_settings()


def test_ask_without_database_reports_the_missing_dependency(client: TestClient) -> None:
    """The dashboard does not change the frozen request path."""
    response = client.post(
        "/ask",
        json={
            "question": "What is due this week?",
            "session_id": "dashboard-test",
            "user_id": "dashboard-test",
        },
    )
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_no_secret_values_in_overview(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard reports configuration, never credentials."""
    secret = "sk-live-should-never-appear"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("JIRA_API_TOKEN", secret)
    _fresh_settings()
    try:
        body = client.get("/admin/overview").text
        assert secret not in body
    finally:
        monkeypatch.delenv("LLM_API_KEY")
        monkeypatch.delenv("JIRA_API_TOKEN")
        _fresh_settings()


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Keep cached settings from leaking between dashboard tests."""
    yield
    get_settings.cache_clear()
    get_connector_settings.cache_clear()
    db_session.reset_engine_state()
