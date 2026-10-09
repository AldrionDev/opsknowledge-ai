"""Tests of the uniqueness guard of the in-memory test repository.

The fake claims to follow the `ChunkRepository` contract, which requires
`add_chunks` to reject an already used `chunk_id` or `(document_id,
chunk_index)` position. A fake that silently overwrote them would let a later
test pass against behavior the real repository does not have.
"""

import pytest

from app.ingestion.models import DocumentChunk, DocumentType
from tests.fake_chunk_repository import InMemoryChunkRepository

EMBEDDING = [0.5] * 8


def make_chunk(
    chunk_id: str, chunk_index: int = 0, document_id: str = "doc-a"
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path="runbooks/example.md",
        document_type=DocumentType.RUNBOOK,
        content="Check the pod logs.",
    )


@pytest.fixture
def repository() -> InMemoryChunkRepository:
    """Repository holding one chunk: `chunk_id` 'stored' at position 0."""
    repository = InMemoryChunkRepository()
    repository.add_chunks([make_chunk("stored")], [EMBEDDING])
    return repository


def test_rejects_a_chunk_id_that_is_already_stored(
    repository: InMemoryChunkRepository,
) -> None:
    before = repository.all_chunks()

    # Free position, but the chunk_id of the stored chunk.
    with pytest.raises(ValueError, match="repeats the chunk_id"):
        repository.add_chunks([make_chunk("stored", chunk_index=1)], [EMBEDDING])

    assert repository.all_chunks() == before


def test_rejects_a_position_that_is_already_stored(
    repository: InMemoryChunkRepository,
) -> None:
    before = repository.all_chunks()

    # New chunk_id, but the position of the stored chunk.
    with pytest.raises(ValueError, match="repeats the position"):
        repository.add_chunks([make_chunk("other", chunk_index=0)], [EMBEDDING])

    assert repository.all_chunks() == before


def test_rejects_a_chunk_id_repeated_inside_the_batch(
    repository: InMemoryChunkRepository,
) -> None:
    before = repository.all_chunks()
    batch = [make_chunk("new", chunk_index=1), make_chunk("new", chunk_index=2)]

    with pytest.raises(ValueError, match="repeats the chunk_id"):
        repository.add_chunks(batch, [EMBEDDING, EMBEDDING])

    # The first element of the rejected batch was not written either.
    assert repository.all_chunks() == before


def test_rejects_a_position_repeated_inside_the_batch(
    repository: InMemoryChunkRepository,
) -> None:
    before = repository.all_chunks()
    batch = [make_chunk("first", chunk_index=1), make_chunk("second", chunk_index=1)]

    with pytest.raises(ValueError, match="repeats the position"):
        repository.add_chunks(batch, [EMBEDDING, EMBEDDING])

    assert repository.all_chunks() == before
