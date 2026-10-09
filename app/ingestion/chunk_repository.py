"""Application-facing persistence boundary for document chunks.

Application code depends on `ChunkRepository`, never on SQLAlchemy. The
SQLAlchemy/PostgreSQL implementation lives in `app.db.chunk_repository`.

Contract of every implementation:

- Chunks and embeddings are passed as parallel sequences of equal length;
  `embeddings[i]` belongs to `chunks[i]`. A length mismatch, an embedding of the
  wrong dimension, or (for `replace_document_chunks`) a chunk of another
  document is rejected with `ValueError` before anything is written.
  `require_embeddings` and `require_chunks_of_document` implement these checks
  so every implementation reports them identically.
- Every operation is atomic: it either fully succeeds or leaves the stored data
  unchanged. A batch is never stored partially.
- `add_chunks` fails if a `chunk_id` or a `(document_id, chunk_index)` position
  already exists; use `replace_document_chunks` to overwrite a document.
- `replace_document_chunks` removes every stored chunk of the document and
  stores the new set in one transaction. Repeating it with the same chunks
  leaves the same state. An empty set removes the document's chunks, exactly
  like `delete_document_chunks`. Chunks of other documents are never touched.
- `get_chunk` returns `None` for an unknown `chunk_id`.
- Database failures are not swallowed; they propagate to the caller.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.models import DocumentChunk


@dataclass(frozen=True)
class StoredChunk:
    """A persisted chunk together with its embedding."""

    chunk: DocumentChunk
    embedding: list[float]


class ChunkRepository(Protocol):
    def add_chunks(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store new chunks with their embeddings."""
        ...

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        """Return the stored chunk, or `None` if `chunk_id` is unknown."""
        ...

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Atomically replace all stored chunks of one document."""
        ...

    def delete_document_chunks(self, document_id: str) -> None:
        """Remove all stored chunks of one document; unknown documents are a no-op."""
        ...


def require_embeddings(
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    dimension: int,
) -> None:
    """Raise `ValueError` unless every chunk has exactly one embedding of
    `dimension` elements.

    `dimension` is supplied by the implementation: the expected vector size is a
    property of the storage, not of this contract.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings; "
            "each chunk needs exactly one embedding"
        )
    for index, embedding in enumerate(embeddings):
        if len(embedding) != dimension:
            raise ValueError(
                f"embeddings[{index}] has {len(embedding)} dimensions, "
                f"expected {dimension}"
            )


def require_chunks_of_document(
    document_id: str, chunks: Sequence[DocumentChunk]
) -> None:
    """Raise `ValueError` if any chunk does not belong to `document_id`.

    A replacement must never write or remove chunks of another document.
    """
    for index, chunk in enumerate(chunks):
        if chunk.document_id != document_id:
            raise ValueError(
                f"chunks[{index}] belongs to document '{chunk.document_id}', "
                f"not to '{document_id}'"
            )
