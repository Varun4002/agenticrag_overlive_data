"""Local embeddings: nomic-embed-text returns 768 dimensions.

M0 pinned ``vector(1536)`` for API-hosted embedders. The project now runs
local-first against Ollama, and ``nomic-embed-text`` — the embedding model
already present on this machine — returns 768 dimensions. The column and its
HNSW index therefore move to 768.

Safe only while ``core.chunks`` is empty (verified 0 rows before writing this
migration). If chunks exist, downgrading/upgrading requires re-embedding, which
is why the migration refuses to run against a non-empty table.

Revision ID: 0002_local_embedding
Revises: 0001_initial_schema
Create Date: 2026-11-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_local_embedding"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

DIMENSIONS = 768
PREVIOUS_DIMENSIONS = 1536
HNWH_INDEX = "ix_chunks_embedding_hnsw"


def _assert_no_embeddings() -> None:
    """Fail loudly instead of silently corrupting/needing a full re-embed."""
    count = op.get_bind().execute(sa.text("SELECT count(*) FROM core.chunks")).scalar_one()
    if count:
        raise RuntimeError(
            f"core.chunks has {count} row(s): changing the embedding dimension "
            "requires re-embedding. Truncate/re-embed first."
        )


def _alter_dimension(dimensions: int) -> None:
    op.execute(f"DROP INDEX IF EXISTS core.{HNWH_INDEX}")
    op.execute(
        f"ALTER TABLE core.chunks ALTER COLUMN embedding TYPE vector({dimensions})"
    )
    op.execute(
        f"CREATE INDEX {HNWH_INDEX} ON core.chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def upgrade() -> None:
    _assert_no_embeddings()
    _alter_dimension(DIMENSIONS)


def downgrade() -> None:
    _assert_no_embeddings()
    _alter_dimension(PREVIOUS_DIMENSIONS)
