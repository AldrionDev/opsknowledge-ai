import hashlib
from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    RUNBOOK = "runbook"
    INCIDENT = "incident"


@dataclass(frozen=True)
class KnowledgeDocument:
    """A normalized knowledge document loaded from the corpus.

    `source_path` is relative to the corpus root and uses `/` separators.
    """

    document_id: str
    source_path: str
    document_type: DocumentType
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    """A contiguous piece of a `KnowledgeDocument`, in document order.

    `chunk_index` is zero-based and sequential within the document. The source
    metadata (`document_id`, `source_path`, `document_type`) is copied unchanged
    from the document.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    source_path: str
    document_type: DocumentType
    content: str


def build_document_id(source_path: str) -> str:
    """Return the deterministic identifier of the document at `source_path`.

    The identifier is the SHA-256 hex digest of the corpus-relative POSIX path.
    It identifies the source location, not the content: editing a document keeps
    its ID, renaming or moving it produces a new one.
    """
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()


def build_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return the deterministic identifier of a chunk of a document.

    The identifier is the SHA-256 hex digest of `"<document_id>:<chunk_index>"`.
    Like the document ID it identifies a position, not content: it does not
    depend on the chunk text or on the chunking configuration.
    """
    return hashlib.sha256(f"{document_id}:{chunk_index}".encode()).hexdigest()
