"""SQLAlchemy/PostgreSQL implementation of `ChunkRepository`.

Each public operation runs in exactly one transaction (`session_scope`), so a
failure rolls everything back. A batch is written with one `INSERT` execution
instead of one statement per chunk; SQLAlchemy may still split a very large
batch into several statements, which stay inside the same transaction.

`ChunkRepository` is subclassed explicitly so that mypy checks every method
signature against the contract, and `_CONTRACT` below makes it reject a missing
method too. The protocol is not `runtime_checkable` and no isinstance check is
performed anywhere.
"""

from collections.abc import Sequence

from sqlalchemy import delete, insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.db.session import session_scope
from app.ingestion.chunk_repository import (
    ChunkRepository,
    StoredChunk,
    require_chunks_of_document,
    require_embeddings,
)
from app.ingestion.models import DocumentChunk, DocumentType


class SqlAlchemyChunkRepository(ChunkRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add_chunks(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        require_embeddings(chunks, embeddings, EMBEDDING_DIMENSION)
        rows = _build_rows(chunks, embeddings)
        if not rows:
            return
        with session_scope(self._session_factory) as session:
            session.execute(insert(DocumentChunkRecord), rows)

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        with session_scope(self._session_factory) as session:
            record = session.get(DocumentChunkRecord, chunk_id)
            if record is None:
                return None
            return _to_stored_chunk(record)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        require_chunks_of_document(document_id, chunks)
        require_embeddings(chunks, embeddings, EMBEDDING_DIMENSION)
        rows = _build_rows(chunks, embeddings)
        with session_scope(self._session_factory) as session:
            session.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.document_id == document_id
                )
            )
            if rows:
                session.execute(insert(DocumentChunkRecord), rows)

    def delete_document_chunks(self, document_id: str) -> None:
        with session_scope(self._session_factory) as session:
            session.execute(
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.document_id == document_id
                )
            )


# Static assertion only: binding the class to `type[ChunkRepository]` makes mypy
# fail if the adapter stops satisfying the contract, for example by dropping or
# renaming a method, which the signature check alone would not catch.
_CONTRACT: type[ChunkRepository] = SqlAlchemyChunkRepository


def _build_rows(
    chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "source_path": chunk.source_path,
            "document_type": chunk.document_type.value,
            "content": chunk.content,
            "embedding": list(embedding),
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def _to_stored_chunk(record: DocumentChunkRecord) -> StoredChunk:
    return StoredChunk(
        chunk=DocumentChunk(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            chunk_index=record.chunk_index,
            source_path=record.source_path,
            document_type=DocumentType(record.document_type),
            content=record.content,
        ),
        # pgvector returns a numpy array; hand plain floats to the application.
        embedding=[float(value) for value in record.embedding],
    )
