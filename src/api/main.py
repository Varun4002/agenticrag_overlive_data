"""FastAPI application factory.

Run locally:

    DATABASE_URL=postgresql+psycopg://rag@localhost:5432/agentic_rag \
        uvicorn api.main:app --host 127.0.0.1 --port 8000

(``uvicorn`` must see ``src`` on the path — installing the project with
``pip install -e .`` does that, see HANDOFF.md.)

Routes: the frozen request path (``/ask``, ``/health``), plus the operator
dashboard at ``/ui`` (JSON under ``/admin/*``, enabled by ``ADMIN_ENABLED``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from api.admin import router as admin_router
from api.routes import router
from config import get_settings
from db.session import dispose_engine
from indexing.indexer import aclose_indexer
from memory.answer_cache import aclose_answer_cache

logger = logging.getLogger(__name__)

UI_FILE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log startup configuration and release resources on shutdown."""
    settings = get_settings()
    logger.info(
        "starting %s env=%s database=%s dashboard=%s",
        settings.app_name,
        settings.app_env,
        "configured" if settings.database_configured else "not configured",
        "enabled" if settings.admin_available else "disabled",
    )
    try:
        yield
    finally:
        await aclose_answer_cache()
        await aclose_indexer()
        await dispose_engine()


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app = FastAPI(
        title="Agentic RAG over Live Data",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(admin_router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Send a browser hitting the root to the dashboard."""
        return RedirectResponse(url="/ui")

    @app.get("/ui", include_in_schema=False)
    async def dashboard() -> FileResponse:
        """Serve the single-file operator dashboard (no build step)."""
        if not get_settings().admin_available:
            raise HTTPException(
                status_code=403,
                detail="operator dashboard is disabled; set ADMIN_ENABLED=true",
            )
        if not UI_FILE.is_file():
            raise HTTPException(
                status_code=500, detail=f"dashboard asset missing: {UI_FILE.name}"
            )
        return FileResponse(
            UI_FILE,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
