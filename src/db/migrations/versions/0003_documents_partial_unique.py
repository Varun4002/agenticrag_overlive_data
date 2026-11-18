"""One current document per item; unlimited history rows (docs/08-decisions.md #2).

M0 modelled document versions with ``UNIQUE (source, source_id, is_current)``,
which permits at most **one** non-current row per item — the connector had to
suffix superseded ``source_id`` values to store a second revision at all. That
constraint is replaced by a *partial* unique index:

- ``uq_documents_current_source_id``: ``UNIQUE (source, source_id) WHERE is_current``
  → exactly one current row per item, any number of historical rows;
- ``ix_documents_source_source_id``: plain index so lookups/history reads by the
  native id stay fast (they no longer all hit a unique prefix).

Data fix: rows retired by the connector under the old rule carry a
``<native>#history-<8 hex>`` suffix. They are renormalised to the native id
*before* the partial index is created and *after* the old constraint is dropped
(the old constraint would reject two non-current rows sharing an id).

``downgrade()`` reverses all of it, including re-suffixing the extra historical
rows so the old constraint can be recreated without deleting audit data.

Safe on an empty and on a populated ``core.documents``: the data fix is a single
idempotent ``UPDATE`` and the index build is a normal (non-concurrent) build.

Revision ID: 0003_documents_partial_unique
Revises: 0002_local_embedding
Create Date: 2026-11-17
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "0003_documents_partial_unique"
down_revision = "0002_local_embedding"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

OLD_CONSTRAINT = "uq_documents_source_source_id_is_current"
CURRENT_INDEX = "uq_documents_current_source_id"
SOURCE_ID_INDEX = "ix_documents_source_source_id"

# Matches what connectors/normalizer.py appended: 8 lowercase hex chars of the row id.
HISTORY_SUFFIX = "#history-[0-9a-f]{8}$"

RENORMALISE = sa.text(
    f"""
    UPDATE core.documents
       SET source_id = regexp_replace(source_id, '{HISTORY_SUFFIX}', '')
     WHERE source_id ~ '{HISTORY_SUFFIX}'
    """
)

# Downgrade only: keep the newest non-current row under the native id and suffix
# the rest, which is exactly the representation the old constraint can hold.
RESUFFIX_EXTRAS = sa.text(
    """
    UPDATE core.documents AS d
       SET source_id = d.source_id || '#history-'
                       || substr(replace(d.id::text, '-', ''), 1, 8)
      FROM (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY source, source_id
                       ORDER BY updated_at DESC, fetched_at DESC, id
                   ) AS rn
              FROM core.documents
             WHERE NOT is_current
           ) AS ranked
     WHERE d.id = ranked.id
       AND ranked.rn > 1
    """
)


def upgrade() -> None:
    bind = op.get_bind()

    op.drop_constraint(OLD_CONSTRAINT, "documents", schema="core", type_="unique")

    renormalised = bind.execute(RENORMALISE).rowcount
    if renormalised:
        logger.info("0003: renormalised %d historical source_id value(s)", renormalised)

    op.create_index(
        CURRENT_INDEX,
        "documents",
        ["source", "source_id"],
        unique=True,
        schema="core",
        postgresql_where=sa.text("is_current"),
    )
    op.create_index(
        SOURCE_ID_INDEX,
        "documents",
        ["source", "source_id"],
        schema="core",
    )


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_index(SOURCE_ID_INDEX, table_name="documents", schema="core")
    op.drop_index(CURRENT_INDEX, table_name="documents", schema="core")

    resuffixed = bind.execute(RESUFFIX_EXTRAS).rowcount
    if resuffixed:
        logger.info("0003 (downgrade): re-suffixed %d extra historical row(s)", resuffixed)

    op.create_unique_constraint(
        OLD_CONSTRAINT,
        "documents",
        ["source", "source_id", "is_current"],
        schema="core",
    )
