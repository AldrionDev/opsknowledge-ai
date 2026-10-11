"""Semantic retrieval: find the persisted chunks most similar to a query.

`SemanticRetrievalService` composes an `EmbeddingService` and a
`ChunkRepository`. It depends only on those abstractions: no SQL, no concrete
embedding library, no web framework.

    query text
        -> EmbeddingService.embed_query
        -> ChunkRepository.search_similar (cosine distance)
        -> ranked `RetrievedChunk` list

Semantics:

- `similarity_score = 1 - cosine_distance`, computed here; the repository only
  reports the raw distance. Higher means more similar.
- Results are ordered by `similarity_score` descending, then by `chunk_id`
  ascending, so repeated queries against unchanged data return the same order.
  The repository orders by distance in the database (required for a correct
  `LIMIT`); the final order is enforced here as well, so it does not depend on
  the repository implementation.
- At most `limit` results are returned, fewer if fewer chunks are stored. An
  empty store yields `[]`.
- The query is passed to the embedding service unchanged.

Errors:

- An empty or whitespace-only query, or a `limit` outside `1..MAX_RESULT_LIMIT`,
  raises `InvalidRetrievalRequestError` before the embedding model is called.
- `EmbeddingError` and every repository exception propagate unchanged.
"""

from app.embeddings.service import EmbeddingService
from app.ingestion.chunk_repository import ChunkMatch, ChunkRepository
from app.retrieval.models import RetrievedChunk

# Application-level bound on the size of a retrieval result; far above typical
# retrieval sizes, and it keeps the payload bounded.
MAX_RESULT_LIMIT = 50


class InvalidRetrievalRequestError(ValueError):
    """Raised when the query text or the result limit is invalid."""


class SemanticRetrievalService:
    def __init__(
        self, embedding_service: EmbeddingService, chunk_repository: ChunkRepository
    ) -> None:
        self._embedding_service = embedding_service
        self._chunk_repository = chunk_repository

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        """Return the `limit` chunks most similar to `query`, most similar first."""
        if not query.strip():
            raise InvalidRetrievalRequestError("query is empty or whitespace-only")
        if not 1 <= limit <= MAX_RESULT_LIMIT:
            raise InvalidRetrievalRequestError(
                f"limit must be between 1 and {MAX_RESULT_LIMIT}, got {limit}"
            )

        query_embedding = self._embedding_service.embed_query(query)
        matches = self._chunk_repository.search_similar(query_embedding, limit)

        results = [_to_retrieved_chunk(match) for match in matches]
        results.sort(key=lambda result: (-result.similarity_score, result.chunk_id))
        return results


def _to_retrieved_chunk(match: ChunkMatch) -> RetrievedChunk:
    chunk = match.chunk
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        source_path=chunk.source_path,
        document_type=chunk.document_type,
        content=chunk.content,
        similarity_score=1.0 - match.cosine_distance,
    )
