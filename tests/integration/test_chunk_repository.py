from collections.abc import Sequence

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.db.session import create_session_factory
from app.ingestion.chunk_repository import ChunkRepository
from app.ingestion.models import DocumentChunk, DocumentType

pytestmark = pytest.mark.integration


def make_chunk(
    document_id: str = "doc-a",
    chunk_index: int = 0,
    *,
    chunk_id: str | None = None,
    content: str | None = None,
    document_type: DocumentType = DocumentType.RUNBOOK,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id or f"{document_id}-{chunk_index}",
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=f"{document_type.value}s/{document_id}.md",
        document_type=document_type,
        content=content or f"Content of {document_id} chunk {chunk_index}.",
    )


def make_embedding(seed: int) -> list[float]:
    # Multiples of 1/8 are exactly representable in float32 (pgvector precision),
    # so a database round trip compares equal. Different seeds give different vectors.
    return [((seed + i) % 8) / 8 for i in range(EMBEDDING_DIMENSION)]


def make_document(
    document_id: str, count: int, seed: int = 0
) -> tuple[list[DocumentChunk], list[list[float]]]:
    chunks = [make_chunk(document_id, index) for index in range(count)]
    embeddings = [make_embedding(seed + index) for index in range(count)]
    return chunks, embeddings


@pytest.fixture
def engine(migrated_engine: Engine) -> Engine:
    return migrated_engine


@pytest.fixture
def repository(engine: Engine) -> ChunkRepository:
    # Typed as the contract: every test below drives the repository through it.
    return SqlAlchemyChunkRepository(create_session_factory(engine))


def stored_chunk_ids(engine: Engine, document_id: str | None = None) -> list[str]:
    query = select(DocumentChunkRecord.chunk_id).order_by(
        DocumentChunkRecord.document_id, DocumentChunkRecord.chunk_index
    )
    if document_id is not None:
        query = query.where(DocumentChunkRecord.document_id == document_id)
    with engine.connect() as connection:
        return list(connection.execute(query).scalars())


def row_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(DocumentChunkRecord)
        ).scalar_one()


def snapshot(
    repository: ChunkRepository, chunks: Sequence[DocumentChunk]
) -> list[tuple[DocumentChunk, list[float]]]:
    result = []
    for chunk in chunks:
        stored = repository.get_chunk(chunk.chunk_id)
        assert stored is not None
        result.append((stored.chunk, stored.embedding))
    return result


def test_store_and_get_one_chunk(repository: ChunkRepository) -> None:
    chunk = make_chunk(content="Check the previous container state.")
    embedding = make_embedding(3)

    repository.add_chunks([chunk], [embedding])

    stored = repository.get_chunk(chunk.chunk_id)
    assert stored is not None
    assert stored.chunk == chunk
    assert stored.embedding == embedding


def test_store_multiple_chunks_associates_each_with_its_embedding(
    repository: ChunkRepository,
) -> None:
    chunks, embeddings = make_document("doc-a", 3)

    repository.add_chunks(chunks, embeddings)

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        stored = repository.get_chunk(chunk.chunk_id)
        assert stored is not None
        assert stored.chunk == chunk
        assert stored.embedding == embedding
    assert len({tuple(e) for e in embeddings}) == 3


def test_get_missing_chunk_returns_none(repository: ChunkRepository) -> None:
    assert repository.get_chunk("does-not-exist") is None


def test_content_and_source_metadata_survive_round_trip(
    repository: ChunkRepository,
) -> None:
    chunk = make_chunk(
        "doc-i",
        7,
        chunk_id="custom-chunk-id",
        content="Multi-line\ncontent with ünicode and `markdown`.\n\n  indented",
        document_type=DocumentType.INCIDENT,
    )

    repository.add_chunks([chunk], [make_embedding(0)])

    stored = repository.get_chunk("custom-chunk-id")
    assert stored is not None
    assert stored.chunk == chunk
    assert stored.chunk.document_type is DocumentType.INCIDENT


def test_stored_embedding_has_the_fixed_dimension(
    repository: ChunkRepository,
) -> None:
    chunk = make_chunk()
    repository.add_chunks([chunk], [make_embedding(1)])

    stored = repository.get_chunk(chunk.chunk_id)

    assert stored is not None
    assert len(stored.embedding) == EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in stored.embedding)


def test_mismatched_counts_are_rejected_and_nothing_is_stored(
    repository: ChunkRepository, engine: Engine
) -> None:
    chunks, embeddings = make_document("doc-a", 2)

    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        repository.add_chunks(chunks, embeddings[:1])

    assert row_count(engine) == 0


def test_add_is_all_or_nothing(repository: ChunkRepository, engine: Engine) -> None:
    chunks, embeddings = make_document("doc-a", 2)
    duplicate_id = make_chunk("doc-b", 0, chunk_id=chunks[0].chunk_id)

    with pytest.raises(IntegrityError):
        repository.add_chunks([*chunks, duplicate_id], [*embeddings, make_embedding(9)])

    assert row_count(engine) == 0


def test_add_rejects_existing_chunk_id_and_keeps_stored_data(
    repository: ChunkRepository,
) -> None:
    original = make_chunk(content="original")
    repository.add_chunks([original], [make_embedding(1)])

    with pytest.raises(IntegrityError):
        repository.add_chunks(
            [make_chunk(content="overwrite attempt")], [make_embedding(2)]
        )

    stored = repository.get_chunk(original.chunk_id)
    assert stored is not None
    assert stored.chunk.content == "original"


def test_replace_swaps_the_chunks_of_the_document(
    repository: ChunkRepository,
) -> None:
    old_chunks, old_embeddings = make_document("doc-a", 2, seed=0)
    repository.add_chunks(old_chunks, old_embeddings)
    new_chunks = [
        make_chunk("doc-a", 0, content="updated first chunk"),
        make_chunk("doc-a", 1, content="updated second chunk"),
    ]
    new_embeddings = [make_embedding(5), make_embedding(6)]

    repository.replace_document_chunks("doc-a", new_chunks, new_embeddings)

    assert snapshot(repository, new_chunks) == list(
        zip(new_chunks, new_embeddings, strict=True)
    )


def test_replace_removes_obsolete_chunks(
    repository: ChunkRepository, engine: Engine
) -> None:
    chunks, embeddings = make_document("doc-a", 4)
    repository.add_chunks(chunks, embeddings)

    repository.replace_document_chunks("doc-a", chunks[:2], embeddings[:2])

    assert stored_chunk_ids(engine, "doc-a") == [chunks[0].chunk_id, chunks[1].chunk_id]
    assert repository.get_chunk(chunks[3].chunk_id) is None


def test_replace_stores_chunks_of_a_new_document(
    repository: ChunkRepository, engine: Engine
) -> None:
    chunks, embeddings = make_document("doc-new", 2)

    repository.replace_document_chunks("doc-new", chunks, embeddings)

    assert stored_chunk_ids(engine, "doc-new") == [c.chunk_id for c in chunks]


def test_repeated_replace_does_not_create_duplicates(
    repository: ChunkRepository, engine: Engine
) -> None:
    chunks, embeddings = make_document("doc-a", 3)

    for _ in range(3):
        repository.replace_document_chunks("doc-a", chunks, embeddings)

    assert row_count(engine) == 3
    assert snapshot(repository, chunks) == list(zip(chunks, embeddings, strict=True))


def test_replace_preserves_other_documents(
    repository: ChunkRepository, engine: Engine
) -> None:
    a_chunks, a_embeddings = make_document("doc-a", 2, seed=0)
    b_chunks, b_embeddings = make_document("doc-b", 3, seed=4)
    repository.add_chunks(a_chunks + b_chunks, a_embeddings + b_embeddings)
    before = snapshot(repository, b_chunks)

    repository.replace_document_chunks("doc-a", a_chunks[:1], a_embeddings[:1])

    assert snapshot(repository, b_chunks) == before
    assert stored_chunk_ids(engine, "doc-b") == [c.chunk_id for c in b_chunks]


def test_replace_with_empty_set_removes_the_document(
    repository: ChunkRepository, engine: Engine
) -> None:
    a_chunks, a_embeddings = make_document("doc-a", 2)
    b_chunks, b_embeddings = make_document("doc-b", 1)
    repository.add_chunks(a_chunks + b_chunks, a_embeddings + b_embeddings)

    repository.replace_document_chunks("doc-a", [], [])

    assert stored_chunk_ids(engine, "doc-a") == []
    assert stored_chunk_ids(engine, "doc-b") == [b_chunks[0].chunk_id]


def test_failed_replace_rolls_back_completely(
    repository: ChunkRepository, engine: Engine
) -> None:
    a_chunks, a_embeddings = make_document("doc-a", 3, seed=0)
    b_chunks, b_embeddings = make_document("doc-b", 2, seed=4)
    repository.add_chunks(a_chunks + b_chunks, a_embeddings + b_embeddings)
    before = snapshot(repository, a_chunks + b_chunks)
    # The first new chunk is valid and is written after the old rows are deleted;
    # the second violates the chunk_index CHECK constraint.
    new_chunks = [
        make_chunk("doc-a", 0, content="new content"),
        make_chunk("doc-a", -1, chunk_id="doc-a-invalid"),
    ]

    with pytest.raises(IntegrityError):
        repository.replace_document_chunks(
            "doc-a", new_chunks, [make_embedding(7), make_embedding(7)]
        )

    assert snapshot(repository, a_chunks + b_chunks) == before
    assert row_count(engine) == 5
    assert repository.get_chunk("doc-a-invalid") is None


def test_replace_colliding_with_another_document_rolls_back(
    repository: ChunkRepository,
) -> None:
    a_chunks, a_embeddings = make_document("doc-a", 2, seed=0)
    b_chunks, b_embeddings = make_document("doc-b", 1, seed=4)
    repository.add_chunks(a_chunks + b_chunks, a_embeddings + b_embeddings)
    before = snapshot(repository, a_chunks + b_chunks)
    # Reuses the chunk_id of doc-b, which the replacement must never overwrite.
    colliding = make_chunk("doc-a", 0, chunk_id=b_chunks[0].chunk_id)

    with pytest.raises(IntegrityError):
        repository.replace_document_chunks("doc-a", [colliding], [make_embedding(1)])

    assert snapshot(repository, a_chunks + b_chunks) == before


def test_delete_removes_only_the_document_chunks(
    repository: ChunkRepository, engine: Engine
) -> None:
    a_chunks, a_embeddings = make_document("doc-a", 2)
    b_chunks, b_embeddings = make_document("doc-b", 2)
    repository.add_chunks(a_chunks + b_chunks, a_embeddings + b_embeddings)

    repository.delete_document_chunks("doc-a")

    assert stored_chunk_ids(engine) == [c.chunk_id for c in b_chunks]


def test_delete_of_unknown_document_is_a_no_op(
    repository: ChunkRepository, engine: Engine
) -> None:
    chunks, embeddings = make_document("doc-a", 1)
    repository.add_chunks(chunks, embeddings)

    repository.delete_document_chunks("unknown")

    assert row_count(engine) == 1
