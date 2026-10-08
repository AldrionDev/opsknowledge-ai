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


def build_document_id(source_path: str) -> str:
    """Return the deterministic identifier of the document at `source_path`.

    The identifier is the SHA-256 hex digest of the corpus-relative POSIX path.
    It identifies the source location, not the content: editing a document keeps
    its ID, renaming or moving it produces a new one.
    """
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()
