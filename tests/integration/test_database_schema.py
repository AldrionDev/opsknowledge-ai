from typing import Any

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import DataError, IntegrityError

from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.db.session import create_session_factory, session_scope
from app.db.verification import (
    EmbeddingDimensionMismatchError,
    verify_embedding_dimension,
)
from app.embeddings.fastembed_service import MODEL_DIMENSION

pytestmark = pytest.mark.integration

TABLE = DocumentChunkRecord.__table__
# Multiples of 1/8 are exactly representable in float32, so round trips compare equal.
EMBEDDING = [(i % 8) / 8 for i in range(EMBEDDING_DIMENSION)]


def chunk_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chunk_id": "chunk-a",
        "document_id": "doc-a",
        "chunk_index": 0,
        "source_path": "runbooks/kubernetes-crashloopbackoff.md",
        "document_type": "runbook",
        "content": "Check the pod logs and the previous container state.",
        "embedding": EMBEDDING,
    }
    row.update(overrides)
    return row


def insert(engine: Engine, **overrides: Any) -> None:
    with engine.begin() as connection:
        connection.execute(TABLE.insert().values(**chunk_row(**overrides)))


def test_columns_are_present_and_required(migrated_engine: Engine) -> None:
    columns = inspect(migrated_engine).get_columns("document_chunks")

    assert {c["name"] for c in columns} == {
        "chunk_id",
        "document_id",
        "chunk_index",
        "source_path",
        "document_type",
        "content",
        "embedding",
    }
    assert all(not c["nullable"] for c in columns)


def test_constraints_exist(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)

    assert inspector.get_pk_constraint("document_chunks")["constrained_columns"] == [
        "chunk_id"
    ]
    assert [
        u["column_names"] for u in inspector.get_unique_constraints("document_chunks")
    ] == [["document_id", "chunk_index"]]
    assert [c["name"] for c in inspector.get_check_constraints("document_chunks")] == [
        "ck_document_chunks_chunk_index_non_negative"
    ]


def test_duplicate_chunk_id_is_rejected(migrated_engine: Engine) -> None:
    insert(migrated_engine)

    with pytest.raises(IntegrityError):
        insert(migrated_engine, document_id="doc-b")


def test_duplicate_document_position_is_rejected(migrated_engine: Engine) -> None:
    insert(migrated_engine)

    with pytest.raises(IntegrityError):
        insert(migrated_engine, chunk_id="chunk-b")


@pytest.mark.parametrize(
    "column",
    [
        "document_id",
        "chunk_index",
        "source_path",
        "document_type",
        "content",
        "embedding",
    ],
)
def test_required_columns_reject_null(migrated_engine: Engine, column: str) -> None:
    with pytest.raises(IntegrityError):
        insert(migrated_engine, **{column: None})


def test_negative_chunk_index_is_rejected(migrated_engine: Engine) -> None:
    with pytest.raises(IntegrityError):
        insert(migrated_engine, chunk_index=-1)


def test_embedding_with_wrong_dimension_is_rejected(migrated_engine: Engine) -> None:
    wrong = "[" + ",".join(["0.5"] * 8) + "]"

    with pytest.raises(DataError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO document_chunks (chunk_id, document_id, chunk_index, "
                "source_path, document_type, content, embedding) VALUES "
                "('c', 'd', 0, 'p', 'runbook', 'x', CAST(:embedding AS vector))"
            ),
            {"embedding": wrong},
        )


def test_database_dimension_matches_the_embedding_model(
    migrated_engine: Engine,
) -> None:
    verify_embedding_dimension(migrated_engine, MODEL_DIMENSION)


def test_incompatible_dimension_is_detected(migrated_engine: Engine) -> None:
    with pytest.raises(EmbeddingDimensionMismatchError, match="768"):
        verify_embedding_dimension(migrated_engine, 768)


def test_missing_schema_is_detected(temp_database_url: str) -> None:
    from app.db.session import create_db_engine

    engine = create_db_engine(temp_database_url)
    try:
        with pytest.raises(EmbeddingDimensionMismatchError, match="migrations"):
            verify_embedding_dimension(engine, MODEL_DIMENSION)
    finally:
        engine.dispose()


def test_long_content_is_stored_without_truncation(migrated_engine: Engine) -> None:
    content = "runbook line\n" * 20_000

    insert(migrated_engine, content=content)

    with migrated_engine.connect() as connection:
        stored = connection.execute(select(TABLE.c.content)).scalar_one()
    assert stored == content


def test_synthetic_row_round_trips(migrated_engine: Engine) -> None:
    factory = create_session_factory(migrated_engine)
    expected = chunk_row(chunk_index=3)

    with session_scope(factory) as session:
        session.add(DocumentChunkRecord(**expected))

    with session_scope(factory) as session:
        stored = session.scalars(select(DocumentChunkRecord)).one()
        assert {
            key: getattr(stored, key) for key in expected if key != "embedding"
        } == {key: value for key, value in expected.items() if key != "embedding"}
        assert len(stored.embedding) == MODEL_DIMENSION
        assert list(stored.embedding) == EMBEDDING


def test_session_scope_commits_on_success(migrated_engine: Engine) -> None:
    factory = create_session_factory(migrated_engine)

    with session_scope(factory) as session:
        session.add(DocumentChunkRecord(**chunk_row()))

    with migrated_engine.connect() as connection:
        assert connection.execute(select(TABLE.c.chunk_id)).scalars().all() == [
            "chunk-a"
        ]


def test_session_scope_rolls_back_on_error(migrated_engine: Engine) -> None:
    factory = create_session_factory(migrated_engine)

    with pytest.raises(RuntimeError), session_scope(factory) as session:
        session.add(DocumentChunkRecord(**chunk_row()))
        session.flush()
        raise RuntimeError("boom")

    with migrated_engine.connect() as connection:
        assert connection.execute(select(TABLE.c.chunk_id)).all() == []
