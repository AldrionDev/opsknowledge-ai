from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from app.embeddings.service import EmbeddingError
from app.ingestion import __main__ as cli
from tests.corpus import multi_chunk_text, write_corpus
from tests.fake_embedding import FakeEmbeddingService


class EngineStub:
    disposed = False

    def dispose(self) -> None:
        self.disposed = True


class ModelMustNotBeLoadedError(AssertionError):
    pass


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> EngineStub:
    stub = EngineStub()
    monkeypatch.setattr(cli, "create_db_engine", lambda: stub)
    monkeypatch.setattr(cli, "create_session_factory", lambda engine: None)
    monkeypatch.setattr(cli, "verify_embedding_dimension", lambda engine, dim: None)
    return stub


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "FastEmbedService", FakeEmbeddingService)


def test_default_corpus_is_the_repository_corpus() -> None:
    assert cli.DEFAULT_CORPUS_ROOT.is_absolute()
    assert (cli.DEFAULT_CORPUS_ROOT / "runbooks").is_dir()
    assert (cli.DEFAULT_CORPUS_ROOT / "incidents").is_dir()


def test_missing_database_configuration_fails_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_to_load() -> None:
        raise ModelMustNotBeLoadedError

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli, "FastEmbedService", fail_to_load)

    assert cli.main([]) == 1

    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_missing_corpus_exits_with_an_error_and_releases_the_engine(
    engine: EngineStub,
    fake_model: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--corpus-root", str(tmp_path / "missing")]) == 1

    assert "error: Corpus root is not a directory" in capsys.readouterr().err
    assert engine.disposed


def test_ingestion_error_is_reported_with_its_cause(
    engine: EngineStub,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingModel(FakeEmbeddingService):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingError("model exploded")

    monkeypatch.setattr(cli, "FastEmbedService", FailingModel)
    corpus = write_corpus(tmp_path, {"runbooks/a.md": multi_chunk_text("a", 1)})

    assert cli.main(["--corpus-root", str(corpus)]) == 1

    error = capsys.readouterr().err
    assert "error: Embedding failed for document: runbooks/a.md" in error
    assert "caused by: EmbeddingError: model exploded" in error


def test_database_error_is_reported_with_the_document_note(
    engine: EngineStub,
    fake_model: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_replace(*args: object) -> None:
        raise OperationalError("INSERT", {}, Exception("connection refused"))

    monkeypatch.setattr(
        cli.SqlAlchemyChunkRepository, "replace_document_chunks", failing_replace
    )
    corpus = write_corpus(tmp_path, {"runbooks/a.md": multi_chunk_text("a", 1)})

    assert cli.main(["--corpus-root", str(corpus)]) == 1

    error = capsys.readouterr().err
    assert "connection refused" in error
    assert "Persisting chunks failed for document: runbooks/a.md" in error


def test_unexpected_errors_are_not_hidden_behind_an_exit_code(
    engine: EngineStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*args: object) -> None:
        raise ValueError("programming error")

    monkeypatch.setattr(cli, "FastEmbedService", FakeEmbeddingService)
    monkeypatch.setattr(cli.IngestionService, "ingest", broken)

    with pytest.raises(ValueError, match="programming error"):
        cli.main(["--corpus-root", str(tmp_path)])
    assert engine.disposed
