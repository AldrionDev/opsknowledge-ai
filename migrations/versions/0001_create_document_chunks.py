"""Create document_chunks

Revision ID: 0001
Revises:
Create Date: 2026-10-09

This migration is a historical snapshot. It must stay self-contained: do not
import application code or the embedding configuration here. The embedding
dimension is fixed at 384 (BAAI/bge-small-en-v1.5); changing it requires a new
migration and re-embedding of the stored chunks.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name=op.f("uq_document_chunks_document_id_chunk_index"),
        ),
        sa.CheckConstraint(
            "chunk_index >= 0", name=op.f("ck_document_chunks_chunk_index_non_negative")
        ),
    )


def downgrade() -> None:
    # The vector extension is left in place: it may be shared with other schemas.
    op.drop_table("document_chunks")
