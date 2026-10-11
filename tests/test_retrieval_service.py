from collections.abc import Sequence
from pathlib import Path

import pytest

from app.embeddings.service import EmbeddingError
from app.ingestion.chunk_repository import ChunkMatch, ChunkRepository, StoredChunk
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.ingestion.models import DocumentChunk, DocumentType
from app.ingestion.service import IngestionService
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import (
    MAX_RESULT_LIMIT,
    InvalidRetrievalRequestError,
    SemanticRetrievalService,
)
from tests.fake_chunk_repository import InMemoryChunkRepository
from tests.fake_embedding import FakeEmbeddingService

QUERY_VECTOR = [0.25, 0.5, 0.75, 1.0]


class RecordingEmbeddingService:
    def __init__(self, failure: Exception | None = None) -> None:
        self.queries: list[str] = []
        self.failure = failure

    @property
    def dimension(self) -> int:
        return len(QUERY_VECTOR)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.failure is not None:
            raise self.failure
        return list(QUERY_VECTOR)


class StubChunkRepository(ChunkRepository):
    """Returns canned matches and records the search arguments.

    Retrieval is read-only, so every other operation of the contract fails
    loudly instead of inheriting the protocol stub, which would silently
    return `None`.
    """

    def __init__(self, matches: list[ChunkMatch] | None = None) -> None:
        self.matches = matches or []
        self.searches: list[tuple[list[float], int]] = []

    def search_similar(
        self, query_embedding: Sequence[float], limit: int
    ) -> list[ChunkMatch]:
        self.searches.append((list(query_embedding), limit))
        return list(self.matches)

    def add_chunks(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        raise AssertionError("retrieval must not store chunks")

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        raise AssertionError("retrieval must not read chunks by id")

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("retrieval must not replace document chunks")

    def delete_document_chunks(self, document_id: str) -> None:
        raise AssertionError("retrieval must not delete document chunks")


def make_chunk(
    chunk_id: str = "chunk-a",
    *,
    document_id: str = "doc-a",
    chunk_index: int = 0,
    source_path: str = "runbooks/example.md",
    document_type: DocumentType = DocumentType.RUNBOOK,
    content: str = "Check the pod logs.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=source_path,
        document_type=document_type,
        content=content,
    )


def match(chunk_id: str, distance: float) -> ChunkMatch:
    return ChunkMatch(make_chunk(chunk_id), distance)


def build_service(
    matches: list[ChunkMatch] | None = None,
    embedding_service: RecordingEmbeddingService | None = None,
) -> tuple[SemanticRetrievalService, RecordingEmbeddingService, StubChunkRepository]:
    embedding = embedding_service or RecordingEmbeddingService()
    repository = StubChunkRepository(matches)
    return SemanticRetrievalService(embedding, repository), embedding, repository


def test_valid_query_is_embedded_exactly_once() -> None:
    service, embedding, _ = build_service([match("a", 0.1)])

    service.retrieve("pod keeps restarting", 5)

    assert embedding.queries == ["pod keeps restarting"]


def test_query_is_embedded_unchanged() -> None:
    service, embedding, _ = build_service()

    service.retrieve("  pod keeps restarting \n", 5)

    assert embedding.queries == ["  pod keeps restarting \n"]


def test_query_vector_and_limit_are_passed_to_the_repository() -> None:
    service, _, repository = build_service()

    service.retrieve("pod keeps restarting", 7)

    assert repository.searches == [(QUERY_VECTOR, 7)]


@pytest.mark.parametrize("query", ["", " ", "   \n\t  "])
def test_empty_or_whitespace_query_is_rejected_before_embedding(query: str) -> None:
    service, embedding, repository = build_service()

    with pytest.raises(InvalidRetrievalRequestError, match="query"):
        service.retrieve(query, 5)

    assert embedding.queries == []
    assert repository.searches == []


@pytest.mark.parametrize("limit", [0, -1, MAX_RESULT_LIMIT + 1])
def test_limit_outside_the_supported_range_is_rejected_before_embedding(
    limit: int,
) -> None:
    service, embedding, repository = build_service()

    with pytest.raises(InvalidRetrievalRequestError, match="limit"):
        service.retrieve("pod keeps restarting", limit)

    assert embedding.queries == []
    assert repository.searches == []


@pytest.mark.parametrize("limit", [1, MAX_RESULT_LIMIT])
def test_limits_at_the_boundaries_are_accepted(limit: int) -> None:
    service, _, repository = build_service()

    service.retrieve("pod keeps restarting", limit)

    assert repository.searches == [(QUERY_VECTOR, limit)]


def test_invalid_request_error_is_a_value_error() -> None:
    assert issubclass(InvalidRetrievalRequestError, ValueError)


def test_empty_repository_result_returns_empty_list() -> None:
    service, _, _ = build_service([])

    assert service.retrieve("pod keeps restarting", 5) == []


def test_fewer_results_than_the_limit_are_returned_as_they_are() -> None:
    service, _, _ = build_service([match("a", 0.1), match("b", 0.2)])

    results = service.retrieve("pod keeps restarting", 10)

    assert [result.chunk_id for result in results] == ["a", "b"]


def test_chunk_content_and_source_metadata_are_preserved() -> None:
    chunk = make_chunk(
        "chunk-x",
        document_id="doc-x",
        chunk_index=3,
        source_path="incidents/inc-1.md",
        document_type=DocumentType.INCIDENT,
        content="Multi-line\ncontent with ünicode.",
    )
    service, _, _ = build_service([ChunkMatch(chunk, 0.5)])

    results = service.retrieve("pod keeps restarting", 5)

    assert results == [
        RetrievedChunk(
            chunk_id="chunk-x",
            document_id="doc-x",
            chunk_index=3,
            source_path="incidents/inc-1.md",
            document_type=DocumentType.INCIDENT,
            content="Multi-line\ncontent with ünicode.",
            similarity_score=0.5,
        )
    ]


@pytest.mark.parametrize(
    ("cosine_distance", "similarity_score"),
    [(0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (2.0, -1.0)],
)
def test_similarity_score_is_one_minus_cosine_distance(
    cosine_distance: float, similarity_score: float
) -> None:
    service, _, _ = build_service([match("a", cosine_distance)])

    [result] = service.retrieve("pod keeps restarting", 5)

    assert result.similarity_score == pytest.approx(similarity_score)


def test_results_are_ordered_by_descending_similarity() -> None:
    # The repository returns the matches in an arbitrary order on purpose.
    service, _, _ = build_service(
        [match("c", 0.9), match("a", 0.1), match("d", 1.5), match("b", 0.4)]
    )

    results = service.retrieve("pod keeps restarting", 10)

    assert [result.chunk_id for result in results] == ["a", "b", "c", "d"]
    scores = [result.similarity_score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_equal_similarity_is_ordered_by_chunk_id() -> None:
    matches = [match("c", 0.3), match("a", 0.3), match("b", 0.3), match("z", 0.1)]
    service, _, repository = build_service(matches)

    first = service.retrieve("pod keeps restarting", 10)
    repository.matches = list(reversed(matches))
    second = service.retrieve("pod keeps restarting", 10)

    assert [result.chunk_id for result in first] == ["z", "a", "b", "c"]
    assert second == first


def test_embedding_failure_propagates_and_the_repository_is_not_searched() -> None:
    embedding = RecordingEmbeddingService(failure=EmbeddingError("model failed"))
    service, _, repository = build_service(embedding_service=embedding)

    with pytest.raises(EmbeddingError, match="model failed"):
        service.retrieve("pod keeps restarting", 5)

    assert repository.searches == []


def test_repository_failure_propagates_unchanged() -> None:
    class FailingRepository(StubChunkRepository):
        def search_similar(
            self, query_embedding: Sequence[float], limit: int
        ) -> list[ChunkMatch]:
            raise RuntimeError("database failure")

    service = SemanticRetrievalService(RecordingEmbeddingService(), FailingRepository())

    with pytest.raises(RuntimeError, match="database failure"):
        service.retrieve("pod keeps restarting", 5)


def test_ingested_chunk_text_retrieves_that_chunk_first() -> None:
    # Offline end to end: the fake embedding is a pure function of the text, so a
    # query equal to a chunk's content is its unambiguous nearest neighbour.
    embedding_service = FakeEmbeddingService(dimension=16)
    repository = InMemoryChunkRepository(dimension=16)
    corpus_root = Path(__file__).resolve().parents[1] / "data" / "knowledge"
    IngestionService(embedding_service, repository).ingest(corpus_root)
    target = chunk_document(load_knowledge_documents(corpus_root)[3])[0]
    service = SemanticRetrievalService(embedding_service, repository)

    results = service.retrieve(target.content, 5)

    assert len(results) == 5
    assert results[0].chunk_id == target.chunk_id
    assert results[0].source_path == target.source_path
    assert results[0].similarity_score == pytest.approx(1.0)
    assert all(
        result.similarity_score < results[0].similarity_score for result in results[1:]
    )
