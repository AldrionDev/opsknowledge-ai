from collections.abc import Sequence

from app.ingestion.chunk_repository import (
    ChunkRepository,
    StoredChunk,
    require_chunks_of_document,
    require_embeddings,
)
from app.ingestion.models import DocumentChunk


class InMemoryChunkRepository(ChunkRepository):
    """In-memory `ChunkRepository` for tests that follows the documented contract.

    Validation uses the contract helpers, and a replacement is all-or-nothing.
    `add_chunks` rejects a `chunk_id` or a `(document_id, chunk_index)` position
    that is already stored or repeated inside the batch, like the unique
    constraints of the database do. `fail_on_document_id` makes the replacement
    of that document raise `failure` before anything is changed, like a
    rolled-back transaction.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self._chunks: dict[str, StoredChunk] = {}
        self.fail_on_document_id: str | None = None
        self.failure: Exception = RuntimeError("database failure")

    def add_chunks(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        require_embeddings(chunks, embeddings, self._dimension)
        self._require_free_positions(chunks)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.chunk_id] = StoredChunk(chunk, list(embedding))

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        return self._chunks.get(chunk_id)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        require_chunks_of_document(document_id, chunks)
        require_embeddings(chunks, embeddings, self._dimension)
        if document_id == self.fail_on_document_id:
            raise self.failure
        self.delete_document_chunks(document_id)
        self.add_chunks(chunks, embeddings)

    def delete_document_chunks(self, document_id: str) -> None:
        self._chunks = {
            chunk_id: stored
            for chunk_id, stored in self._chunks.items()
            if stored.chunk.document_id != document_id
        }

    def _require_free_positions(self, chunks: Sequence[DocumentChunk]) -> None:
        chunk_ids = set(self._chunks)
        positions = {
            (stored.chunk.document_id, stored.chunk.chunk_index)
            for stored in self._chunks.values()
        }
        for index, chunk in enumerate(chunks):
            position = (chunk.document_id, chunk.chunk_index)
            if chunk.chunk_id in chunk_ids:
                raise ValueError(
                    f"chunks[{index}] repeats the chunk_id '{chunk.chunk_id}'"
                )
            if position in positions:
                raise ValueError(
                    f"chunks[{index}] repeats the position "
                    f"(document_id='{chunk.document_id}', "
                    f"chunk_index={chunk.chunk_index})"
                )
            chunk_ids.add(chunk.chunk_id)
            positions.add(position)

    def all_chunks(self) -> list[StoredChunk]:
        """All stored chunks ordered by `(source_path, chunk_index)`."""
        return sorted(
            self._chunks.values(),
            key=lambda stored: (stored.chunk.source_path, stored.chunk.chunk_index),
        )
