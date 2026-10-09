from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select, text

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.db.session import create_session_factory
from app.ingestion import __main__ as cli
from app.ingestion.__main__ import DEFAULT_CORPUS_ROOT
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.ingestion.service import IngestionService
from tests.corpus import multi_chunk_text, write_corpus
from tests.fake_embedding import FakeEmbeddingService

pytestmark = pytest.mark.integration


class FakeModel(FakeEmbeddingService):
    def __init__(self) -> None:
        super().__init__(dimension=EMBEDDING_DIMENSION)


@pytest.fixture
def service(migrated_engine: Engine) -> IngestionService:
    return IngestionService(
        FakeModel(), SqlAlchemyChunkRepository(create_session_factory(migrated_engine))
    )


def stored_chunk_ids(engine: Engine, source_path: str | None = None) -> list[str]:
    query = select(DocumentChunkRecord.chunk_id).order_by(
        DocumentChunkRecord.source_path, DocumentChunkRecord.chunk_index
    )
    if source_path is not None:
        query = query.where(DocumentChunkRecord.source_path == source_path)
    with engine.connect() as connection:
        return list(connection.execute(query).scalars())


def row_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(DocumentChunkRecord)
        ).scalar_one()


def test_repository_corpus_ingestion_is_idempotent(
    service: IngestionService, migrated_engine: Engine
) -> None:
    first = service.ingest(DEFAULT_CORPUS_ROOT)
    ids_after_first = stored_chunk_ids(migrated_engine)

    second = service.ingest(DEFAULT_CORPUS_ROOT)

    assert first.chunks_generated > 0
    assert first.chunks_persisted == first.chunks_generated
    assert second == first
    assert row_count(migrated_engine) == first.chunks_generated
    assert stored_chunk_ids(migrated_engine) == ids_after_first


def test_persisted_rows_match_the_chunking_output(
    service: IngestionService, migrated_engine: Engine
) -> None:
    service.ingest(DEFAULT_CORPUS_ROOT)

    expected = {
        chunk.chunk_id: chunk
        for document in load_knowledge_documents(DEFAULT_CORPUS_ROOT)
        for chunk in chunk_document(document)
    }
    with migrated_engine.connect() as connection:
        rows = connection.execute(select(DocumentChunkRecord)).all()
        dimensions = set(
            connection.execute(
                text("SELECT vector_dims(embedding) FROM document_chunks")
            )
            .scalars()
            .all()
        )

    assert {row.chunk_id for row in rows} == set(expected)
    for row in rows:
        chunk = expected[row.chunk_id]
        assert (row.document_id, row.chunk_index, row.content) == (
            chunk.document_id,
            chunk.chunk_index,
            chunk.content,
        )
        assert row.source_path == chunk.source_path
        assert not Path(row.source_path).is_absolute()
        assert row.document_type == chunk.document_type.value
    assert dimensions == {EMBEDDING_DIMENSION}


def test_changed_document_loses_its_obsolete_chunks(
    service: IngestionService, migrated_engine: Engine, tmp_path: Path
) -> None:
    corpus = write_corpus(
        tmp_path / "corpus",
        {
            "runbooks/a.md": multi_chunk_text("a", 3),
            "incidents/b.md": multi_chunk_text("b", 2),
        },
    )
    service.ingest(corpus)
    old_ids = stored_chunk_ids(migrated_engine, "runbooks/a.md")
    other_ids = stored_chunk_ids(migrated_engine, "incidents/b.md")

    (corpus / "runbooks/a.md").write_text(multi_chunk_text("a2", 1), encoding="utf-8")
    summary = service.ingest(corpus)

    new_ids = stored_chunk_ids(migrated_engine, "runbooks/a.md")
    assert len(old_ids) == 3
    assert new_ids == old_ids[:1]
    assert stored_chunk_ids(migrated_engine, "incidents/b.md") == other_ids
    assert row_count(migrated_engine) == 3
    assert summary.chunks_persisted == 3


def test_command_ingests_the_corpus_into_the_configured_database(
    migrated_engine: Engine,
    temp_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", temp_database_url)
    monkeypatch.setattr(cli, "FastEmbedService", FakeModel)
    corpus = write_corpus(tmp_path, {"runbooks/a.md": multi_chunk_text("a", 2)})

    assert cli.main(["--corpus-root", str(corpus)]) == 0

    assert (
        "Ingestion completed: documents=1 chunks_generated=2 chunks_persisted=2"
        in capsys.readouterr().out
    )
    assert row_count(migrated_engine) == 2
