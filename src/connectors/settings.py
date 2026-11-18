"""Connector configuration, read from the environment only.

Deliberately a *separate* ``BaseSettings`` class instead of adding fields to
``src/config.py``: this component does not own that file while other agents are
building against it. Field names map to upper-case env vars automatically
(``jira_api_token`` -> ``JIRA_API_TOKEN``), matching the existing convention.

Secrets are ``SecretStr``: they never appear in ``repr``/logs. Nothing here has a
default credential — a missing token is a :class:`ConnectorConfigError` at use
time, not a silent anonymous request.

Recommended follow-up for the Lead: fold these fields into ``src/config.py``
(that file is outside this component's scope). ``.env.example`` documents the
variable names.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from connectors.base import ConnectorConfigError


@dataclass(frozen=True, slots=True)
class JiraCredentials:
    """Validated Jira connection material.

    ``api_token`` is a ``SecretStr``; ``repr`` renders ``SecretStr('**********')``
    so an accidental log line cannot leak it.
    """

    base_url: str
    email: str
    api_token: SecretStr
    projects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GmailCredentials:
    """Validated Gmail IMAP connection material.

    Gmail is reached over IMAP with an *app password* (no OAuth client exists for
    this deployment). Same rule as Jira: the secret is a ``SecretStr`` and only
    ``imaplib`` ever sees its value.
    """

    address: str
    app_password: SecretStr
    imap_host: str
    imap_port: int


class ConnectorSettings(BaseSettings):
    """Typed view of the connector-relevant process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Jira Cloud ----------------------------------------------------
    jira_base_url: str = ""
    jira_email: str | None = None
    # API token from https://id.atlassian.com/manage/api-tokens — env only.
    jira_api_token: SecretStr | None = None
    # Allowlisted projects: comma-separated keys, e.g. "PHX,OPS".
    jira_projects: str = ""
    # Optional extra JQL constraint ANDed with the project scope (e.g.
    # "issuetype != Epic"). Cannot widen the project allowlist.
    jira_jql_extra: str | None = None
    jira_page_size: int = Field(default=50, ge=1, le=100)
    # Jira is migrating search endpoints (offset ``/rest/api/3/search`` ->
    # token-based ``/rest/api/3/search/jql``); overridable per tenant.
    jira_search_path: str = "/rest/api/3/search/jql"
    jira_max_retries: int = Field(default=3, ge=0)
    jira_backoff_base_s: float = Field(default=0.5, gt=0)
    jira_request_timeout_s: float = Field(default=30.0, gt=0)
    jira_rate_budget_calls: int = Field(default=300, gt=0)
    jira_rate_budget_window_s: float = Field(default=60.0, gt=0)

    # --- Gmail (IMAP, app password) ------------------------------------
    # Account address: also the ACL identity of every mail document.
    gmail_address: str = ""
    # App password (Google account -> Security -> 2-Step Verification -> App
    # passwords). Environment only — never code, tests, or a committed file.
    gmail_app_password: SecretStr | None = None
    gmail_imap_host: str = "imap.gmail.com"
    gmail_imap_port: int = 993
    # Messages per ``UID FETCH`` round.
    gmail_page_size: int = Field(default=50, ge=1, le=200)
    # Hard cap per label per sync run: an inbox is a firehose otherwise.
    gmail_max_messages: int = Field(default=200, ge=1)
    # Allowlisted labels (comma-separated). Each one is read as an IMAP mailbox.
    gmail_labels: str = "INBOX"

    # --- Notion (stub until M2) ---------------------------------------
    notion_database_ids: str = ""

    @field_validator("jira_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def allowed_projects(self) -> tuple[str, ...]:
        """The allowlisted Jira project keys, in configured order."""
        return split_csv(self.jira_projects)

    @property
    def allowed_gmail_labels(self) -> tuple[str, ...]:
        return split_csv(self.gmail_labels) or ("INBOX",)

    @property
    def gmail_configured(self) -> bool:
        """True when both the address and a non-empty app password are present."""
        # An empty ``SecretStr`` is still truthy, so the *value* decides.
        return bool(self.gmail_address and self._gmail_password())

    def _gmail_password(self) -> str:
        password = self.gmail_app_password
        return password.get_secret_value() if password is not None else ""

    def require_gmail(self) -> GmailCredentials:
        """Validate and return the Gmail connection material, or raise."""
        missing = [
            name
            for name, value in (
                ("GMAIL_ADDRESS", self.gmail_address),
                ("GMAIL_APP_PASSWORD", self._gmail_password()),
            )
            if not value
        ]
        if missing:
            raise ConnectorConfigError(
                f"Gmail is not configured: missing {', '.join(missing)} "
                "(environment only — never code or a committed file)"
            )
        password = self.gmail_app_password
        assert self.gmail_address and password is not None  # guaranteed by `missing`
        return GmailCredentials(
            address=self.gmail_address,
            app_password=password,
            imap_host=self.gmail_imap_host,
            imap_port=self.gmail_imap_port,
        )

    def require_jira(self) -> JiraCredentials:
        """Validate and return the Jira connection material, or raise."""
        base_url = self.jira_base_url
        email = self.jira_email
        token = self.jira_api_token
        token_value = token.get_secret_value() if token is not None else ""
        missing = [
            name
            for name, value in (
                ("JIRA_BASE_URL", base_url),
                ("JIRA_EMAIL", email),
                ("JIRA_API_TOKEN", token_value),
            )
            if not value
        ]
        if missing:
            raise ConnectorConfigError(
                f"Jira is not configured: missing {', '.join(missing)} "
                "(environment only — never code or a committed file)"
            )
        if not self.allowed_projects:
            raise ConnectorConfigError(
                "JIRA_PROJECTS is empty: the connector only syncs an explicit allowlist "
                "(comma-separated project keys)"
            )
        assert email is not None and token is not None  # guaranteed by `missing`
        return JiraCredentials(
            base_url=base_url, email=email, api_token=token, projects=self.allowed_projects
        )


def split_csv(raw: str) -> tuple[str, ...]:
    """Comma-separated env value -> tuple, ignoring blanks."""
    items = (part.strip() for part in raw.split(","))
    return tuple(item for item in items if item)


@lru_cache(maxsize=1)
def get_connector_settings() -> ConnectorSettings:
    """Process-wide connector settings (tests call ``cache_clear()``)."""
    return ConnectorSettings()


__all__ = [
    "ConnectorSettings",
    "GmailCredentials",
    "JiraCredentials",
    "get_connector_settings",
    "split_csv",
]
