"""Application-facing embedding boundary.

Application code depends on `EmbeddingService`, never on a concrete embedding
library. Vectors are plain Python `list[float]` so they can be handed to any
later persistence layer without coupling to it.

Contract of every implementation:

- `embed_documents([])` returns `[]` without running the model.
- Empty or whitespace-only texts are rejected with `ValueError` before the model
  runs; the whole batch is rejected and the message names the offending index.
  Valid texts are passed to the model unchanged.
- `embed_documents` returns exactly one vector per text, in input order.
- All vectors, including query vectors, have exactly `dimension` elements and
  contain only finite values.
- Implementations are expected to return L2-normalized vectors, so cosine
  similarity and dot product rank alike. This is a property of the model, not
  something the service verifies.
- Failures of the model or violations of the output contract raise
  `EmbeddingError`.
"""

from typing import Protocol


class EmbeddingError(Exception):
    """Raised when the embedding model fails or returns an invalid result."""


class EmbeddingService(Protocol):
    @property
    def dimension(self) -> int:
        """Number of elements of every returned vector."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts; the result matches `texts` in length and order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query into the same vector space as the documents."""
        ...


def require_text(text: str, name: str) -> None:
    """Raise `ValueError` if `text` is empty or whitespace-only.

    `name` identifies the input in the error message, for example `texts[2]`.
    """
    if not text.strip():
        raise ValueError(f"{name} is empty or whitespace-only")
