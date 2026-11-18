"""Engine and session management.

The engine is created lazily from ``DATABASE_URL`` so importing the app never
requires a database. ``database_available()`` is the single place that decides
whether DB-backed features may run; callers must handle the false case instead
of assuming connectivity.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(settings: Settings) -> AsyncEngine | None:
    if not settings.database_configured:
        logger.warning("DATABASE_URL is not set: database features are disabled")
        return None
    url = settings.database_url or ""
    return create_async_engine(
        url,
        echo=settings.db_echo,
        poolclass=NullPool,  # short-lived API processes; avoids stale pooled conns
        connect_args={"connect_timeout": settings.db_connect_timeout_s},
    )


def get_engine() -> AsyncEngine | None:
    """Return the process-wide async engine, or ``None`` when unconfigured."""
    global _engine
    if _engine is None:
        _engine = _build_engine(get_settings())
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the process-wide session factory, or ``None`` when unconfigured."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        if engine is None:
            return None
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session; raises if no DB is configured."""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("DATABASE_URL is not set")
    async with factory() as session:
        yield session


async def database_available() -> bool:
    """Cheap connectivity probe used by ``/health`` and test skip logic."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("database probe failed: %s", exc)
        return False
    return True


async def dispose_engine() -> None:
    """Release engine resources (called on app shutdown and in tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def reset_engine_state() -> None:
    """Drop cached engine references without awaiting disposal (tests only)."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
