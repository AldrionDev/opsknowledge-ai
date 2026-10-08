from typing import Any

import numpy as np
import pytest
from fastembed import TextEmbedding

from app.embeddings.fastembed_service import (
    DOCUMENT_BATCH_SIZE,
    MODEL_DIMENSION,
    MODEL_NAME,
    FastEmbedService,
)
from app.embeddings.service import EmbeddingError
from tests.fake_embedding import StubTextEmbedding, use_stub


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[StubTextEmbedding]:
    use_stub(monkeypatch, StubTextEmbedding)
    return StubTextEmbedding


# --- model identity ---


def test_model_name_and_dimension_are_recorded() -> None:
    assert MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert MODEL_DIMENSION == 384


def test_recorded_dimension_matches_fastembed_catalog() -> None:
    # Catalog metadata only: nothing is downloaded or loaded.
    assert TextEmbedding.get_embedding_size(MODEL_NAME) == MODEL_DIMENSION


def test_dimension_is_exposed(stub: type[StubTextEmbedding]) -> None:
    assert FastEmbedService().dimension == MODEL_DIMENSION


# --- model initialization and reuse ---


def test_model_is_created_with_name_and_cache_dir(
    stub: type[StubTextEmbedding],
) -> None:
    FastEmbedService(cache_dir="/some/cache")

    assert len(stub.instances) == 1
    assert stub.instances[0].model_name == MODEL_NAME
    assert stub.instances[0].cache_dir == "/some/cache"


def test_model_is_loaded_once_and_reused_across_calls(
    stub: type[StubTextEmbedding],
) -> None:
    service = FastEmbedService()

    service.embed_documents(["a", "b"])
    service.embed_documents(["c"])
    service.embed_query("q")
    service.embed_query("r")

    assert len(stub.instances) == 1


def test_load_failure_raises_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingLoad(StubTextEmbedding):
        def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
            raise OSError("no network")

    use_stub(monkeypatch, FailingLoad)

    with pytest.raises(EmbeddingError, match=MODEL_NAME) as exc_info:
        FastEmbedService()
    assert isinstance(exc_info.value.__cause__, OSError)


# --- document batches ---


def test_documents_are_embedded_in_one_provider_call_in_order(
    stub: type[StubTextEmbedding],
) -> None:
    texts = ["x" * n for n in range(1, 101)]

    vectors = FastEmbedService().embed_documents(texts)

    assert stub.instances[0].calls == [("passage", texts)]
    assert stub.instances[0].batch_sizes == [DOCUMENT_BATCH_SIZE]
    assert [vector[0] for vector in vectors] == [float(n + 1) for n in range(1, 101)]


def test_empty_document_list_does_not_call_the_provider(
    stub: type[StubTextEmbedding],
) -> None:
    assert FastEmbedService().embed_documents([]) == []
    assert stub.instances[0].calls == []


def test_invalid_input_is_rejected_before_the_provider_is_called(
    stub: type[StubTextEmbedding],
) -> None:
    service = FastEmbedService()

    with pytest.raises(ValueError, match=r"texts\[1\]"):
        service.embed_documents(["valid", " \n\t", ""])
    with pytest.raises(ValueError, match="query"):
        service.embed_query("  ")

    assert stub.instances[0].calls == []


def test_valid_text_is_passed_to_the_provider_unchanged(
    stub: type[StubTextEmbedding],
) -> None:
    service = FastEmbedService()

    service.embed_documents(["  padded text \n"])
    service.embed_query("  padded query ")

    assert stub.instances[0].calls == [
        ("passage", ["  padded text \n"]),
        ("query", ["  padded query "]),
    ]


# --- output conversion ---


def test_vectors_are_plain_python_floats(stub: type[StubTextEmbedding]) -> None:
    service = FastEmbedService()

    document_vector = service.embed_documents(["abc"])[0]
    query_vector = service.embed_query("abc")

    for vector in (document_vector, query_vector):
        assert type(vector) is list
        assert len(vector) == MODEL_DIMENSION
        assert all(type(value) is float for value in vector)


def test_query_and_document_vectors_have_the_same_dimension(
    stub: type[StubTextEmbedding],
) -> None:
    service = FastEmbedService()

    assert len(service.embed_query("abc")) == len(service.embed_documents(["abc"])[0])


# --- provider failures and invalid provider output ---


def failing_service(monkeypatch: pytest.MonkeyPatch, vectors: Any) -> FastEmbedService:
    class Misbehaving(StubTextEmbedding):
        def _vectors(self, texts: list[str]) -> list[Any]:
            if isinstance(vectors, Exception):
                raise vectors
            return vectors

    use_stub(monkeypatch, Misbehaving)
    return FastEmbedService()


def vector(length: int = MODEL_DIMENSION, value: float = 1.0) -> Any:
    return np.full(length, value, dtype=np.float32)


def test_inference_failure_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = failing_service(monkeypatch, RuntimeError("onnx failure"))

    with pytest.raises(EmbeddingError) as documents_error:
        service.embed_documents(["a"])
    with pytest.raises(EmbeddingError) as query_error:
        service.embed_query("a")
    assert isinstance(documents_error.value.__cause__, RuntimeError)
    assert isinstance(query_error.value.__cause__, RuntimeError)


def test_wrong_document_vector_count_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = failing_service(monkeypatch, [vector()])

    with pytest.raises(EmbeddingError, match="Expected 2 vectors"):
        service.embed_documents(["a", "b"])


@pytest.mark.parametrize("returned", [[], [vector(), vector()]])
def test_query_must_yield_exactly_one_vector(
    monkeypatch: pytest.MonkeyPatch, returned: list[Any]
) -> None:
    service = failing_service(monkeypatch, returned)

    with pytest.raises(EmbeddingError, match="Expected 1 vectors"):
        service.embed_query("a")


def test_wrong_dimension_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = failing_service(monkeypatch, [vector(MODEL_DIMENSION - 1)])

    with pytest.raises(EmbeddingError, match="dimension"):
        service.embed_documents(["a"])
    with pytest.raises(EmbeddingError, match="dimension"):
        service.embed_query("a")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_raise_embedding_error(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    bad = vector()
    bad[3] = bad_value
    service = failing_service(monkeypatch, [bad])

    with pytest.raises(EmbeddingError, match="non-finite"):
        service.embed_documents(["a"])
    with pytest.raises(EmbeddingError, match="non-finite"):
        service.embed_query("a")
