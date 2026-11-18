"""Alembic environment.

The database URL comes from ``settings.database_url`` (env ``DATABASE_URL``)
and never from ``alembic.ini``, so no credentials live in the repository.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# `alembic` is normally run from the project root; make `src` importable so the
# models and settings resolve without an editable install.
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import get_settings  # noqa: E402
from db import models  # noqa: E402,F401  (imported for autogenerate metadata)
from db.base import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the migration URL: env var first, then settings."""
    url = os.environ.get("DATABASE_URL") or get_settings().database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Export it (see .env.example) before running migrations."
        )
    # Alembic's sync engine cannot use the async-only drivers' async entries in
    # some setups, but psycopg3 works synchronously through the same URL.
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
