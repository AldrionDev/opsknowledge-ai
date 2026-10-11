from dataclasses import dataclass

from app.ingestion.models import DocumentType


@dataclass(frozen=True)
class RetrievedChunk:
    """A persisted chunk found by semantic retrieval, with its source metadata.

    `similarity_score` is `1 - cosine_distance` between the query embedding and
    the chunk embedding. Higher means more similar: 1 is the same direction, 0
    orthogonal, a negative value the opposite direction. The range is
    `[-1, 1]`; the value is not clamped and, as pgvector computes in float32,
    an identical vector can score slightly below 1 (for example 0.99999994).
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    source_path: str
    document_type: DocumentType
    content: str
    similarity_score: float
