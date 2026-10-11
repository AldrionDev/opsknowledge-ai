from pathlib import Path

import pytest
from sqlalchemy import Engine

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION
from app.db.session import create_session_factory
from app.embeddings.service import require_text
from app.ingestion.__main__ import DEFAULT_CORPUS_ROOT
from app.ingestion.chunk_repository import ChunkRepository
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.ingestion.models import DocumentChunk, DocumentType
from app.ingestion.service import IngestionService
from app.retrieval.service import SemanticRetrievalService
from tests.fake_embedding import FakeEmbeddingService

pytestmark = pytest.mark.integration

SCORE_TOLERANCE = 1e-6  # pgvector computes in float32


def axis(index: int, sign: float = 1.0) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[index] = sign
    return vector


def blend(first: int, second: int) -> list[float]:
    """Unit vector exactly between two axes (cosine similarity 1/sqrt(2))."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[first] = vector[second] = 2**-0.5
    return vector


def make_chunk(
    chunk_id: str,
    *,
    document_id: str | None = None,
    chunk_index: int = 0,
    document_type: DocumentType = DocumentType.RUNBOOK,
    content: str | None = None,
) -> DocumentChunk:
    document_id = document_id or f"doc-{chunk_id}"
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=f"{document_type.value}s/{document_id}.md",
        document_type=document_type,
        content=content or f"Content of {chunk_id}.",
    )


@pytest.fixture
def repository(migrated_engine: Engine) -> ChunkRepository:
    return SqlAlchemyChunkRepository(create_session_factory(migrated_engine))


class FixedQueryEmbedding(FakeEmbeddingService):
    """Fake embedding whose query vector is chosen by the test.

    The documented text contract still applies, so an invalid query is
    rejected here exactly as the real service would reject it.
    """

    def __init__(self, query_vector: list[float]) -> None:
        super().__init__(dimension=EMBEDDING_DIMENSION)
        self._query_vector = query_vector

    def embed_query(self, text: str) -> list[float]:
        require_text(text, "query text")
        return self._query_vector


def test_nearest_neighbour_is_ranked_first(repository: ChunkRepository) -> None:
    repository.add_chunks(
        [make_chunk("c0"), make_chunk("c1"), make_chunk("c2")],
        [axis(0), axis(1), axis(2)],
    )

    matches = repository.search_similar(axis(1), 3)

    assert matches[0].chunk.chunk_id == "c1"
    assert matches[0].cosine_distance == pytest.approx(0.0, abs=SCORE_TOLERANCE)
    assert [m.cosine_distance for m in matches[1:]] == pytest.approx(
        [1.0, 1.0], abs=SCORE_TOLERANCE
    )


def test_similarity_score_follows_cosine_distance_on_the_real_database(
    repository: ChunkRepository,
) -> None:
    repository.add_chunks(
        [
            make_chunk("same"),
            make_chunk("near"),
            make_chunk("ortho"),
            make_chunk("opp"),
        ],
        [axis(0), blend(0, 1), axis(1), axis(0, -1.0)],
    )
    service = SemanticRetrievalService(FixedQueryEmbedding(axis(0)), repository)

    results = service.retrieve("any query", 10)

    assert [r.chunk_id for r in results] == ["same", "near", "ortho", "opp"]
    assert [r.similarity_score for r in results] == pytest.approx(
        [1.0, 2**-0.5, 0.0, -1.0], abs=SCORE_TOLERANCE
    )


def test_limit_returns_the_nearest_chunks_only(repository: ChunkRepository) -> None:
    repository.add_chunks(
        [make_chunk(f"c{i}") for i in range(5)],
        [axis(0), blend(0, 1), blend(0, 2), axis(1), axis(0, -1.0)],
    )

    top_two = repository.search_similar(axis(0), 2)
    everything = repository.search_similar(axis(0), 50)

    assert [m.chunk.chunk_id for m in top_two] == ["c0", "c1"]
    assert len(everything) == 5


def test_equal_distance_is_ordered_by_chunk_id(repository: ChunkRepository) -> None:
    # Inserted in descending chunk_id order so insertion order cannot explain it.
    chunks = [make_chunk(chunk_id) for chunk_id in ("d", "c", "b", "a")]
    repository.add_chunks(chunks, [axis(3)] * 4)
    repository.add_chunks([make_chunk("far")], [axis(4)])

    results = [repository.search_similar(axis(3), 5) for _ in range(3)]

    expected = ["a", "b", "c", "d", "far"]
    for matches in results:
        assert [m.chunk.chunk_id for m in matches] == expected


def test_tie_at_the_limit_boundary_keeps_the_smaller_chunk_id(
    repository: ChunkRepository,
) -> None:
    repository.add_chunks(
        [make_chunk(chunk_id) for chunk_id in ("c", "b", "a")], [axis(3)] * 3
    )

    matches = repository.search_similar(axis(3), 2)

    assert [m.chunk.chunk_id for m in matches] == ["a", "b"]


def test_empty_database_returns_no_results(repository: ChunkRepository) -> None:
    assert repository.search_similar(axis(0), 5) == []
    service = SemanticRetrievalService(FixedQueryEmbedding(axis(0)), repository)
    assert service.retrieve("any query", 5) == []


def test_search_result_preserves_content_and_source_metadata(
    repository: ChunkRepository,
) -> None:
    chunk = make_chunk(
        "meta",
        document_id="doc-i",
        chunk_index=7,
        document_type=DocumentType.INCIDENT,
        content="Multi-line\ncontent with ünicode and `markdown`.\n\n  indented",
    )
    repository.add_chunks([chunk], [axis(0)])

    [match] = repository.search_similar(axis(0), 1)

    assert match.chunk == chunk
    assert match.chunk.document_type is DocumentType.INCIDENT


def test_search_does_not_modify_the_stored_data(repository: ChunkRepository) -> None:
    chunk = make_chunk("c0")
    repository.add_chunks([chunk], [axis(0)])

    repository.search_similar(axis(1), 5)

    stored = repository.get_chunk("c0")
    assert stored is not None
    assert stored.chunk == chunk
    assert stored.embedding == axis(0)


def test_ingested_corpus_chunk_text_is_its_own_nearest_neighbour(
    migrated_engine: Engine,
) -> None:
    # Fake embeddings are a pure function of the text, so the query equal to a
    # chunk's content must rank that chunk first with a score of ~1.
    embedding_service = FakeEmbeddingService(dimension=EMBEDDING_DIMENSION)
    repository = SqlAlchemyChunkRepository(create_session_factory(migrated_engine))
    corpus: Path = DEFAULT_CORPUS_ROOT
    IngestionService(embedding_service, repository).ingest(corpus)
    documents = load_knowledge_documents(corpus)
    target = chunk_document(documents[2])[-1]
    service = SemanticRetrievalService(embedding_service, repository)

    results = service.retrieve(target.content, 5)

    assert len(results) == 5
    assert results[0].chunk_id == target.chunk_id
    assert results[0].source_path == target.source_path
    assert results[0].document_type == target.document_type
    assert results[0].similarity_score == pytest.approx(1.0, abs=SCORE_TOLERANCE)
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)
