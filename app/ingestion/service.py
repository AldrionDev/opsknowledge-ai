"""Ingestion pipeline: load, chunk, embed and persist the knowledge corpus.

`IngestionService` composes the document loader, the chunker, an
`EmbeddingService` and a `ChunkRepository`. It depends only on those
abstractions: no SQL, no concrete embedding library.

For every document, in `source_path` order, the chunk contents are embedded in
one `embed_documents` call and the stored chunks of the document are replaced
with `ChunkRepository.replace_document_chunks`, which is atomic per document.
Ingesting an unchanged corpus again therefore stores the same chunks, and a
changed document never keeps chunks of its previous version.

The whole corpus is loaded before anything is written, so an invalid corpus
changes nothing. Ingestion stops at the first failing document (fail-fast, no
retries). Documents ingested before the failure stay stored; running the
ingestion again is safe. Chunks of documents that were deleted or renamed in the
corpus are not removed.

Error handling:

- `DocumentLoadError` propagates unchanged.
- `EmbeddingError` and an embedding count that differs from the chunk count are
  reported as `IngestionError` that names the document.
- Every exception of the repository, a contract `ValueError` included,
  propagates unchanged with a note naming the document; the service does not
  know the persistence technology.
- Any other exception is a programming error and propagates unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

from app.embeddings.service import EmbeddingError, EmbeddingService
from app.ingestion.chunk_repository import ChunkRepository
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents


class IngestionError(Exception):
    """Raised when a document cannot be embedded consistently."""


@dataclass(frozen=True)
class IngestionSummary:
    """Counts of a completed ingestion run.

    `chunks_persisted` counts the chunks whose document replacement succeeded.
    """

    documents_processed: int
    chunks_generated: int
    chunks_persisted: int


class IngestionService:
    def __init__(
        self, embedding_service: EmbeddingService, chunk_repository: ChunkRepository
    ) -> None:
        self._embedding_service = embedding_service
        self._chunk_repository = chunk_repository

    def ingest(self, corpus_root: Path) -> IngestionSummary:
        """Ingest every document under `corpus_root` and return the summary."""
        documents = load_knowledge_documents(corpus_root)

        documents_processed = 0
        chunks_generated = 0
        chunks_persisted = 0
        for document in documents:
            chunks = chunk_document(document)
            chunks_generated += len(chunks)

            try:
                embeddings = self._embedding_service.embed_documents(
                    [chunk.content for chunk in chunks]
                )
            except EmbeddingError as exc:
                raise IngestionError(
                    f"Embedding failed for document: {document.source_path}"
                ) from exc
            if len(embeddings) != len(chunks):
                raise IngestionError(
                    f"Got {len(embeddings)} embeddings for {len(chunks)} chunks "
                    f"of document: {document.source_path}"
                )

            try:
                self._chunk_repository.replace_document_chunks(
                    document.document_id, chunks, embeddings
                )
            except Exception as exc:
                exc.add_note(
                    f"Persisting chunks failed for document: {document.source_path}"
                )
                raise

            documents_processed += 1
            chunks_persisted += len(chunks)

        return IngestionSummary(
            documents_processed=documents_processed,
            chunks_generated=chunks_generated,
            chunks_persisted=chunks_persisted,
        )
