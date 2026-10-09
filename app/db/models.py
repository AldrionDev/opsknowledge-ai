"""Database model of the persisted document chunks.

`DocumentChunkRecord` mirrors `app.ingestion.models.DocumentChunk` plus the
chunk embedding. The schema itself is created by Alembic, never by
`Base.metadata.create_all()`.

`EMBEDDING_DIMENSION` is deliberately a literal and not imported from the
embedding provider: the database column dimension is part of the schema and must
not change just because the embedding model configuration does. Changing it
requires a new Alembic migration and re-embedding of the stored chunks; the
existing migration must never be edited. Tests assert that this value equals the
dimension of the selected embedding model.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Integer, MetaData, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSION = 384

NAMING_CONVENTION = {
    "pk": "pk_%(table_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
    )

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    source_path: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
