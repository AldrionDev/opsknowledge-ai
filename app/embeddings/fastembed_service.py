"""Local embedding service backed by FastEmbed (ONNX Runtime, CPU only).

Model: `BAAI/bge-small-en-v1.5`, 384-dimensional, English, 512 input tokens,
L2-normalized output. FastEmbed downloads a Qdrant-converted ONNX build of the
model from the Hugging Face Hub on first use and caches it; later runs work
offline. See `docs/embeddings.md` for cache, offline and revision details.

No query instruction prefix is added: for this model version it is optional, and
FastEmbed does not add one either.
"""

import math

from fastembed import TextEmbedding
from fastembed.common.types import NumpyArray

from app.embeddings.service import EmbeddingError, require_text

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIMENSION = 384
DOCUMENT_BATCH_SIZE = 32


class FastEmbedService:
    """`EmbeddingService` implementation using a local FastEmbed model.

    The model is loaded once, in the constructor, and reused by every call.
    Creating another instance loads the model again. `cache_dir` overrides the
    FastEmbed cache location (default: `FASTEMBED_CACHE_PATH` or a directory in
    the system temp dir).
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        try:
            self._model = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)
        except Exception as exc:
            raise EmbeddingError(f"Cannot load embedding model {MODEL_NAME}") from exc

    @property
    def dimension(self) -> int:
        return MODEL_DIMENSION

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for index, text in enumerate(texts):
            require_text(text, f"texts[{index}]")
        if not texts:
            return []

        try:
            raw = list(self._model.passage_embed(texts, batch_size=DOCUMENT_BATCH_SIZE))
        except Exception as exc:
            raise EmbeddingError("Document embedding failed") from exc
        return _to_vectors(raw, expected_count=len(texts))

    def embed_query(self, text: str) -> list[float]:
        require_text(text, "query text")

        try:
            raw = list(self._model.query_embed(text))
        except Exception as exc:
            raise EmbeddingError("Query embedding failed") from exc
        return _to_vectors(raw, expected_count=1)[0]


def _to_vectors(raw: list[NumpyArray], expected_count: int) -> list[list[float]]:
    # `tolist()` yields plain Python floats; `list(array)` would keep numpy scalars.
    vectors = [vector.tolist() for vector in raw]
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"Expected {expected_count} vectors, model returned {len(vectors)}"
        )
    for vector in vectors:
        if len(vector) != MODEL_DIMENSION:
            raise EmbeddingError(
                f"Expected vectors of dimension {MODEL_DIMENSION}, "
                f"model returned {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingError("Model returned non-finite vector values")
    return vectors
