"""Config tests: env-var mapping, defaults, and secret handling."""

from __future__ import annotations

import pytest

from config import Settings


def test_defaults_are_usable_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DATABASE_URL",
        "LLM_MODEL",
        "LLM_API_KEY",
        "EMBEDDING_MODEL",
        "JIRA_SYNC_INTERVAL_S",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.database_url is None
    assert settings.database_configured is False
    assert settings.llm_model == "qwen3.5:4b"
    # Local-first defaults: Ollama on this machine, nomic-embed-text at 768 dims.
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.embedding_dimensions == 768
    assert settings.jira_sync_interval_s == 900


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/rag_test")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    monkeypatch.setenv("NOTION_SYNC_INTERVAL_S", "60")
    settings = Settings(_env_file=None)

    assert settings.database_configured is True
    assert settings.llm_model == "some-model"
    assert settings.notion_sync_interval_s == 60
    # Secrets never appear in repr/log output.
    assert "secret-value" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "secret-value"


def test_invalid_interval_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_SYNC_INTERVAL_S", "0")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
