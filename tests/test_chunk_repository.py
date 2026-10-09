from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION
from app.ingestion.chunk_repository import (
    ChunkRepository,
    require_chunks_of_document,
    require_embeddings,
)
from app.ingestion.models import DocumentChunk, DocumentType


class SessionFactoryMustNotBeUsedError(AssertionError):
    pass


def unused_session_factory() -> Any:
    raise SessionFactoryMustNotBeUsedError("the database must not be accessed")


@pytest.fixture
def repository() -> ChunkRepository:
    # Validation failures must be raised before any session (connection) is opened.
    factory: sessionmaker[Session] = unused_session_factory
    return SqlAlchemyChunkRepository(factory)


def make_chunk(document_id: str = "doc-a", chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{document_id}-{chunk_index}",
        document_id=document_id,
        chunk_index=chunk_index,
        source_path="runbooks/example.md",
        document_type=DocumentType.RUNBOOK,
        content="Check the pod logs.",
    )


def make_embedding(dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    return [0.5] * dimension


def test_add_rejects_more_chunks_than_embeddings(
    repository: ChunkRepository,
) -> None:
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        repository.add_chunks(
            [make_chunk(chunk_index=0), make_chunk(chunk_index=1)], [make_embedding()]
        )


def test_add_rejects_more_embeddings_than_chunks(
    repository: ChunkRepository,
) -> None:
    with pytest.raises(ValueError, match="1 chunks but 2 embeddings"):
        repository.add_chunks([make_chunk()], [make_embedding(), make_embedding()])


def test_add_rejects_embedding_with_wrong_dimension(
    repository: ChunkRepository,
) -> None:
    with pytest.raises(ValueError, match=r"embeddings\[1\] has 3 dimensions"):
        repository.add_chunks(
            [make_chunk(chunk_index=0), make_chunk(chunk_index=1)],
            [make_embedding(), make_embedding(3)],
        )


def test_replace_rejects_count_mismatch(
    repository: ChunkRepository,
) -> None:
    with pytest.raises(ValueError, match="1 chunks but 0 embeddings"):
        repository.replace_document_chunks("doc-a", [make_chunk()], [])


def test_replace_rejects_embedding_with_wrong_dimension(
    repository: ChunkRepository,
) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        repository.replace_document_chunks("doc-a", [make_chunk()], [make_embedding(3)])


def test_replace_rejects_chunk_of_another_document(
    repository: ChunkRepository,
) -> None:
    chunks = [make_chunk("doc-a", 0), make_chunk("doc-b", 1)]

    with pytest.raises(ValueError, match=r"chunks\[1\] belongs to document 'doc-b'"):
        repository.replace_document_chunks(
            "doc-a", chunks, [make_embedding(), make_embedding()]
        )


def test_add_of_nothing_does_not_access_the_database(
    repository: ChunkRepository,
) -> None:
    repository.add_chunks([], [])


def test_require_embeddings_accepts_one_embedding_per_chunk() -> None:
    chunks = [make_chunk(chunk_index=0), make_chunk(chunk_index=1)]

    require_embeddings(
        chunks, [make_embedding(), make_embedding()], EMBEDDING_DIMENSION
    )


def test_require_embeddings_accepts_nothing() -> None:
    require_embeddings([], [], EMBEDDING_DIMENSION)


def test_require_embeddings_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        require_embeddings(
            [make_chunk(chunk_index=0), make_chunk(chunk_index=1)],
            [make_embedding()],
            EMBEDDING_DIMENSION,
        )


def test_require_embeddings_names_the_offending_embedding() -> None:
    with pytest.raises(
        ValueError, match=r"embeddings\[1\] has 3 dimensions, expected 384"
    ):
        require_embeddings(
            [make_chunk(chunk_index=0), make_chunk(chunk_index=1)],
            [make_embedding(), make_embedding(3)],
            EMBEDDING_DIMENSION,
        )


def test_require_embeddings_checks_against_the_given_dimension() -> None:
    # The expected size belongs to the implementation, not to the contract.
    require_embeddings([make_chunk()], [make_embedding(3)], 3)

    with pytest.raises(ValueError, match="expected 3"):
        require_embeddings([make_chunk()], [make_embedding()], 3)


def test_require_chunks_of_document_accepts_chunks_of_that_document() -> None:
    chunks = [make_chunk("doc-a", 0), make_chunk("doc-a", 1)]

    require_chunks_of_document("doc-a", chunks)


def test_require_chunks_of_document_accepts_nothing() -> None:
    require_chunks_of_document("doc-a", [])


def test_require_chunks_of_document_names_the_foreign_chunk() -> None:
    chunks = [make_chunk("doc-a", 0), make_chunk("doc-b", 1)]

    with pytest.raises(
        ValueError, match=r"chunks\[1\] belongs to document 'doc-b', not to 'doc-a'"
    ):
        require_chunks_of_document("doc-a", chunks)
