from pathlib import Path

import pytest

from app.embeddings.service import EmbeddingError
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import DocumentLoadError, load_knowledge_documents
from app.ingestion.models import DocumentChunk, DocumentType, build_document_id
from app.ingestion.service import IngestionError, IngestionService, IngestionSummary
from tests.corpus import multi_chunk_text, write_corpus
from tests.fake_chunk_repository import InMemoryChunkRepository
from tests.fake_embedding import FakeEmbeddingService

RUNBOOK = "runbooks/disk-space.md"
INCIDENT = "incidents/inc-001.md"
OTHER_RUNBOOK = "runbooks/pod-restart.md"


class RecordingEmbeddingService(FakeEmbeddingService):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed_documents(texts)


class FailingEmbeddingService(FakeEmbeddingService):
    def __init__(self, failure: Exception) -> None:
        super().__init__()
        self._failure = failure

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise self._failure


class ShortEmbeddingService(FakeEmbeddingService):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts)[:-1]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(
        tmp_path / "corpus",
        {
            RUNBOOK: multi_chunk_text("disk", 3),
            INCIDENT: multi_chunk_text("incident", 2),
        },
    )


@pytest.fixture
def embedding_service() -> RecordingEmbeddingService:
    return RecordingEmbeddingService()


@pytest.fixture
def repository() -> InMemoryChunkRepository:
    return InMemoryChunkRepository()


@pytest.fixture
def service(
    embedding_service: RecordingEmbeddingService, repository: InMemoryChunkRepository
) -> IngestionService:
    return IngestionService(embedding_service, repository)


def expected_chunks(corpus: Path) -> list[DocumentChunk]:
    return [
        chunk
        for document in load_knowledge_documents(corpus)
        for chunk in chunk_document(document)
    ]


def test_persists_the_chunker_output_of_every_document(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    service.ingest(corpus)

    persisted = [stored.chunk for stored in repository.all_chunks()]
    assert persisted == sorted(
        expected_chunks(corpus), key=lambda c: (c.source_path, c.chunk_index)
    )
    assert len(persisted) == 5


def test_embeds_the_chunks_of_a_document_in_one_batch(
    service: IngestionService,
    embedding_service: RecordingEmbeddingService,
    corpus: Path,
) -> None:
    service.ingest(corpus)

    # Documents are processed in source_path order: incidents/ before runbooks/.
    assert [len(batch) for batch in embedding_service.calls] == [2, 3]
    assert embedding_service.calls == [
        [c.content for c in chunk_document(document)]
        for document in load_knowledge_documents(corpus)
    ]


def test_every_chunk_keeps_its_own_embedding(
    service: IngestionService,
    repository: InMemoryChunkRepository,
    embedding_service: FakeEmbeddingService,
    corpus: Path,
) -> None:
    service.ingest(corpus)

    for stored in repository.all_chunks():
        assert (
            stored.embedding
            == embedding_service.embed_documents([stored.chunk.content])[0]
        )
    contents = {stored.chunk.content for stored in repository.all_chunks()}
    assert len(contents) == 5


def test_persisted_chunks_carry_machine_independent_source_metadata(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    service.ingest(corpus)

    by_path = {
        stored.chunk.source_path: stored.chunk for stored in repository.all_chunks()
    }
    assert set(by_path) == {RUNBOOK, INCIDENT}
    assert by_path[RUNBOOK].document_type is DocumentType.RUNBOOK
    assert by_path[RUNBOOK].document_id == build_document_id(RUNBOOK)
    assert by_path[INCIDENT].document_type is DocumentType.INCIDENT
    assert all(
        str(corpus) not in stored.chunk.source_path
        for stored in repository.all_chunks()
    )


def test_summary_counts_documents_and_chunks(
    service: IngestionService, corpus: Path
) -> None:
    summary = service.ingest(corpus)

    assert summary == IngestionSummary(
        documents_processed=2, chunks_generated=5, chunks_persisted=5
    )


def test_ingesting_an_unchanged_corpus_again_does_not_duplicate_chunks(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    first_summary = service.ingest(corpus)
    first_state = repository.all_chunks()

    second_summary = service.ingest(corpus)

    assert repository.all_chunks() == first_state
    assert second_summary == first_summary


def test_changed_document_replaces_its_previous_chunks(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    service.ingest(corpus)
    untouched = [
        stored
        for stored in repository.all_chunks()
        if stored.chunk.source_path == INCIDENT
    ]
    old_last_chunk_id = [
        stored.chunk.chunk_id
        for stored in repository.all_chunks()
        if stored.chunk.source_path == RUNBOOK
    ][-1]

    (corpus / RUNBOOK).write_text(multi_chunk_text("rewritten", 1), encoding="utf-8")
    summary = service.ingest(corpus)

    runbook_chunks = [
        stored
        for stored in repository.all_chunks()
        if stored.chunk.source_path == RUNBOOK
    ]
    assert [s.chunk.content for s in runbook_chunks] == [
        multi_chunk_text("rewritten", 1)
    ]
    assert repository.get_chunk(old_last_chunk_id) is None
    assert [
        stored
        for stored in repository.all_chunks()
        if stored.chunk.source_path == INCIDENT
    ] == untouched
    assert summary.chunks_persisted == 3


def test_missing_corpus_is_reported_and_nothing_is_stored(
    service: IngestionService, repository: InMemoryChunkRepository, tmp_path: Path
) -> None:
    with pytest.raises(DocumentLoadError, match="not a directory"):
        service.ingest(tmp_path / "does-not-exist")

    assert repository.all_chunks() == []


def test_invalid_document_prevents_any_write(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    (corpus / "runbooks" / "empty.md").write_text("\n\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match=r"runbooks/empty\.md"):
        service.ingest(corpus)

    assert repository.all_chunks() == []


def test_embedding_error_is_reported_with_the_document_and_keeps_old_chunks(
    repository: InMemoryChunkRepository, corpus: Path
) -> None:
    IngestionService(FakeEmbeddingService(), repository).ingest(corpus)
    before = repository.all_chunks()
    (corpus / INCIDENT).write_text(multi_chunk_text("changed", 1), encoding="utf-8")
    failure = EmbeddingError("model failed")

    with pytest.raises(IngestionError, match=INCIDENT) as error:
        IngestionService(FailingEmbeddingService(failure), repository).ingest(corpus)

    assert error.value.__cause__ is failure
    assert repository.all_chunks() == before


def test_unexpected_embedding_exception_is_not_converted(
    repository: InMemoryChunkRepository, corpus: Path
) -> None:
    service = IngestionService(
        FailingEmbeddingService(RuntimeError("programming error")), repository
    )

    with pytest.raises(RuntimeError, match="programming error"):
        service.ingest(corpus)


def test_embedding_count_mismatch_is_reported_and_nothing_is_stored(
    repository: InMemoryChunkRepository, corpus: Path
) -> None:
    service = IngestionService(ShortEmbeddingService(), repository)

    with pytest.raises(IngestionError, match=r"1 embeddings for 2 chunks.*incidents/"):
        service.ingest(corpus)

    assert repository.all_chunks() == []


def test_repository_failure_propagates_unchanged_with_document_note(
    service: IngestionService, repository: InMemoryChunkRepository, corpus: Path
) -> None:
    failure = RuntimeError("connection lost")
    repository.fail_on_document_id = build_document_id(RUNBOOK)
    repository.failure = failure

    with pytest.raises(RuntimeError) as error:
        service.ingest(corpus)

    assert error.value is failure
    assert any(RUNBOOK in note for note in error.value.__notes__)


def test_ingestion_stops_at_the_first_failing_document(
    service: IngestionService, repository: InMemoryChunkRepository, tmp_path: Path
) -> None:
    corpus = write_corpus(
        tmp_path / "corpus",
        {
            INCIDENT: multi_chunk_text("incident", 1),
            OTHER_RUNBOOK: multi_chunk_text("pod", 1),
            RUNBOOK: multi_chunk_text("disk", 1),
        },
    )
    # Source path order: INCIDENT, RUNBOOK (fails), OTHER_RUNBOOK.
    repository.fail_on_document_id = build_document_id(RUNBOOK)

    with pytest.raises(RuntimeError):
        service.ingest(corpus)

    assert {s.chunk.source_path for s in repository.all_chunks()} == {INCIDENT}


def test_contract_violation_of_the_repository_is_not_converted(
    repository: InMemoryChunkRepository, corpus: Path
) -> None:
    # The repository stores 8-dimensional vectors; a 4-dimensional embedding
    # service violates its contract. That is a wiring error, not an ingestion error.
    service = IngestionService(FakeEmbeddingService(dimension=4), repository)

    with pytest.raises(ValueError, match="dimensions"):
        service.ingest(corpus)
