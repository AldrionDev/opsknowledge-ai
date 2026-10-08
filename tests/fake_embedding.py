import hashlib
import math
from typing import Any

import numpy as np
import pytest

from app.embeddings import fastembed_service
from app.embeddings.fastembed_service import MODEL_DIMENSION
from app.embeddings.service import require_text


class FakeEmbeddingService:
    """Deterministic, offline `EmbeddingService` for tests.

    The vector of a text depends only on the text: it is derived from the
    SHA-256 digest of the text and L2-normalized, like the real model output.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for index, text in enumerate(texts):
            require_text(text, f"texts[{index}]")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        require_text(text, "query text")
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            values.extend(byte / 255 + 0.01 for byte in digest)
            counter += 1
        values = values[: self._dimension]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


class StubTextEmbedding:
    """Offline stand-in for `fastembed.TextEmbedding`.

    Records constructions and calls, and returns deterministic float32 arrays
    (the dtype of the real model), one per text, derived from the text length.
    Subclasses override `_vectors` to simulate misbehaving providers.
    """

    instances: list["StubTextEmbedding"] = []

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.calls: list[tuple[str, list[str]]] = []
        self.batch_sizes: list[int | None] = []
        StubTextEmbedding.instances.append(self)

    def passage_embed(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls.append(("passage", list(texts)))
        self.batch_sizes.append(kwargs.get("batch_size"))
        yield from self._vectors(texts)

    def query_embed(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(("query", [query]))
        yield from self._vectors([query])

    def _vectors(self, texts: list[str]) -> list[Any]:
        return [
            np.full(MODEL_DIMENSION, len(text) + 1, dtype=np.float32) for text in texts
        ]


def use_stub(monkeypatch: pytest.MonkeyPatch, stub: type[StubTextEmbedding]) -> None:
    stub.instances = []
    monkeypatch.setattr(fastembed_service, "TextEmbedding", stub)
