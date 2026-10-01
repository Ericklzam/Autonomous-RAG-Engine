"""Initial schema: pgvector extension, document_chunks, HNSW index.

Revision ID: 0001
Revises:
Create Date: 2026-09-09

Sprint 1 created this table via ``Base.metadata.create_all`` with no vector
index. This migration is the authoritative definition going forward.

NOTE: any rows written during Sprint 1 hold all-zero embeddings. Cosine
distance against a zero vector is undefined (division by zero), so those rows
sort unpredictably into every result set and must not be carried over. Start
from an empty table -- see README "Resetting the database".
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    # The extension must exist before a Vector column can be created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("document_name", sa.String(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    )

    op.create_index(
        "ix_document_chunks_document_name",
        "document_chunks",
        ["document_name"],
    )

    # HNSW over IVFFlat: it builds on an empty table and needs no training
    # pass, whereas IVFFlat requires representative data to already exist.
    #
    # vector_cosine_ops MUST match the distance operator used at query time
    # (`<=>`, via .cosine_distance()). A mismatch is silent: Postgres just
    # ignores the index and sequential-scans.
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_name", table_name="document_chunks")
    op.drop_table("document_chunks")
