"""Behavioral contract shared by every `EmbeddingService` implementation.

Run against the deterministic fake and against `FastEmbedService` with a stubbed
provider, so no model is downloaded.
"""

import math

import pytest

from app.embeddings.fastembed_service import FastEmbedService
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import chunk_document
from app.ingestion.models import DocumentChunk, DocumentType, KnowledgeDocument
from tests.fake_embedding import FakeEmbeddingService, StubTextEmbedding, use_stub


@pytest.fixture(params=["fake", "fastembed"])
def service(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> EmbeddingService:
    if request.param == "fake":
        return FakeEmbeddingService()
    use_stub(monkeypatch, StubTextEmbedding)
    return FastEmbedService()


TEXTS = ["a", "bb ccc", "restart the pod", "dddd" * 10]


def test_one_vector_per_document_with_service_dimension(
    service: EmbeddingService,
) -> None:
    vectors = service.embed_documents(TEXTS)

    assert len(vectors) == len(TEXTS)
    assert all(len(vector) == service.dimension for vector in vectors)


def test_input_order_is_preserved(service: EmbeddingService) -> None:
    batch = service.embed_documents(TEXTS)
    reversed_batch = service.embed_documents(TEXTS[::-1])

    assert batch == [service.embed_documents([text])[0] for text in TEXTS]
    assert reversed_batch == batch[::-1]


def test_different_texts_get_different_vectors(service: EmbeddingService) -> None:
    vectors = service.embed_documents(TEXTS)

    assert len({tuple(vector) for vector in vectors}) == len(TEXTS)


def test_embedding_is_deterministic(service: EmbeddingService) -> None:
    assert service.embed_documents(TEXTS) == service.embed_documents(TEXTS)
    assert service.embed_query("search") == service.embed_query("search")


def test_query_vector_has_the_document_dimension(service: EmbeddingService) -> None:
    query = service.embed_query("how to restart")

    assert len(query) == service.dimension
    assert len(query) == len(service.embed_documents(["how to restart"])[0])


def test_vectors_are_finite_python_floats(service: EmbeddingService) -> None:
    vectors = [*service.embed_documents(TEXTS), service.embed_query("q")]

    assert all(type(value) is float for vector in vectors for value in vector)
    assert all(math.isfinite(value) for vector in vectors for value in vector)


def test_empty_document_list_returns_empty_list(service: EmbeddingService) -> None:
    assert service.embed_documents([]) == []


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \n"])
def test_blank_document_rejects_the_whole_batch(
    service: EmbeddingService, blank: str
) -> None:
    with pytest.raises(ValueError, match=r"texts\[2\]"):
        service.embed_documents(["valid", "also valid", blank, "valid again"])


def test_first_blank_document_is_reported_in_mixed_batch(
    service: EmbeddingService,
) -> None:
    with pytest.raises(ValueError, match=r"texts\[1\]"):
        service.embed_documents(["valid", " ", "valid", ""])


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \n"])
def test_blank_query_is_rejected(service: EmbeddingService, blank: str) -> None:
    with pytest.raises(ValueError, match="query"):
        service.embed_query(blank)


# --- application code depends on the abstraction only ---


def embed_chunks(
    service: EmbeddingService, chunks: list[DocumentChunk]
) -> dict[str, list[float]]:
    """Example application-level consumer typed against the Protocol."""
    vectors = service.embed_documents([chunk.content for chunk in chunks])
    return {
        chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors, strict=True)
    }


def test_application_code_uses_the_abstraction_with_a_fake() -> None:
    document = KnowledgeDocument(
        document_id="doc",
        source_path="runbooks/a.md",
        document_type=DocumentType.RUNBOOK,
        content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
    )
    chunks = chunk_document(document, max_chunk_size=20)

    embedded = embed_chunks(FakeEmbeddingService(dimension=4), chunks)

    assert list(embedded) == [chunk.chunk_id for chunk in chunks]
    assert len(chunks) == 3
    assert all(len(vector) == 4 for vector in embedded.values())


# --- fake behavior ---


def test_fake_vectors_are_l2_normalized_and_configurable() -> None:
    fake = FakeEmbeddingService(dimension=100)

    vector = fake.embed_query("text")

    assert fake.dimension == 100
    assert len(vector) == 100
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)
