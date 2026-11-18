"""Application configuration.

All runtime configuration comes from environment variables (or a git-ignored
``.env`` file) via ``pydantic-settings``. No credentials are stored in code:
secret fields are ``SecretStr`` and default to ``None``.

Field names map to upper-case env vars automatically (``llm_model`` ->
``LLM_MODEL``), which is why no prefix is configured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app -----------------------------------------------------------
    app_name: str = "agentic-rag-over-live-data"
    app_env: Literal["dev", "test", "staging", "prod"] = "dev"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- database ------------------------------------------------------
    # Unset means "no database configured": the API still serves requests and
    # /health reports database.reachable=false. Password, if any, comes from
    # the environment (PGPASSWORD or a full URL in DATABASE_URL) — never code.
    database_url: str | None = None
    db_echo: bool = False
    db_connect_timeout_s: int = Field(default=5, ge=1)

    # --- audit ---------------------------------------------------------
    # When true, a request whose audit events cannot be written is refused with
    # 503 instead of served unaudited (docs/06-guardrails-pii.md, "Audit
    # availability policy"). Dev may opt out; prod must not.
    audit_strict: bool = True

    # --- LLM -----------------------------------------------------------
    llm_provider: Literal["openai_compatible"] = "openai_compatible"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "qwen3.5:4b"
    # Structured-output steps (claim extraction, planning) cannot use a
    # reasoning model: qwen3.5 spends its whole budget thinking and returns
    # empty content. Empty means "same as llm_model".
    llm_json_model: str = ""
    # Reasoning models (qwen3.5) spend tokens thinking before answering; a
    # 30s ceiling was too tight for them on this machine.
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_backoff_base_s: float = Field(default=0.5, gt=0)

    # --- embeddings ----------------------------------------------------
    # Local-first: Ollama's nomic-embed-text (768 dims). Must match
    # core.chunks.embedding (migration 0002); a different model means a new
    # migration, since the pgvector column dimension is fixed at DDL time.
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = Field(default=768, gt=0)

    # --- evidence ------------------------------------------------------
    # Fail-closed thresholds for the Evidence Agent (docs/agents/05).
    evidence_sufficiency_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_min_claim_overlap: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- orchestrator --------------------------------------------------
    # Evidence candidates per request; the retriever's own default is 12.
    orchestrator_top_k: int = Field(default=12, ge=1, le=50)
    # Wall-clock ceiling for one question (retrieval + 2 LLM calls). Exceeding
    # it refuses with `degraded=true` rather than answering on unverified text.
    orchestrator_budget_s: float = Field(default=120.0, gt=0)

    # --- answer cache (Redis, M4) --------------------------------------
    # Unset means "no cache": repeat questions run the full pipeline, exactly
    # as they would without Redis. Keys are per user and per data version
    # (src/memory/answer_cache.py) — never keyed on the question alone.
    redis_url: str | None = None
    answer_cache_ttl_s: int = Field(default=900, ge=1)
    # Bypass the cache even when REDIS_URL is set (e.g. while debugging a
    # pipeline that must not be short-circuited).
    answer_cache_enabled: bool = True

    # --- sync intervals (seconds) -------------------------------------
    jira_sync_interval_s: int = Field(default=900, gt=0)
    notion_sync_interval_s: int = Field(default=900, gt=0)
    gmail_sync_interval_s: int = Field(default=1800, gt=0)
    sync_stale_after_s: int = Field(default=900, gt=0)

    # --- operator dashboard ---------------------------------------------
    # Enables ``/ui`` and ``/admin/*``. Unset means "on in dev/test, off
    # everywhere else"; set ADMIN_ENABLED explicitly to override. The dashboard
    # can trigger syncs and embedding runs, so it must stay closed in prod.
    admin_enabled: bool | None = None

    # --- PII scrubbing -------------------------------------------------
    # Salt for audit.redactions.original_hash (see src/pii/audit.py). The
    # default is a documented dev-only value: with a shared salt, the hashes of
    # common names/emails are dictionary-attackable, so a real deployment must
    # set PII_HASH_SALT. The original value itself is never stored.
    pii_hash_salt: str = "dev-only-pii-salt-change-me"
    # Presidio entity classes to redact. Removing one is a policy change
    # (docs/06-guardrails-pii.md); tests assert the default set covers it.
    pii_entities_enabled: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "LOCATION",
            "CREDIT_CARD",
            "ADDRESS",
            "API_KEY",
            "PRIVATE_KEY",
            "DATE_OF_BIRTH",
        ]
    )
    # JSON file of non-person terms that must survive scrubbing (project names,
    # codenames). Unset means the dev fixture corpus manifest, when present.
    pii_allowlist_path: str | None = None

    @property
    def database_configured(self) -> bool:
        """True when a database URL is present (it may still be unreachable)."""
        return bool(self.database_url)

    @property
    def admin_available(self) -> bool:
        """Whether the operator dashboard is exposed for this environment."""
        if self.admin_enabled is not None:
            return self.admin_enabled
        return self.app_env in ("dev", "test")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so every module sees the same configuration; tests clear the cache
    (``get_settings.cache_clear()``) after changing the environment.
    """
    return Settings()
