import dataclasses

import pytest

from app.ingestion.models import (
    DocumentChunk,
    DocumentType,
    KnowledgeDocument,
    build_chunk_id,
    build_document_id,
)


def test_build_document_id_is_stable_sha256_of_source_path() -> None:
    # Fixed vector: guards against algorithm changes and process-dependent hashing.
    assert build_document_id("runbooks/example.md") == (
        "991ae7f3b28781f3f7d5b99d2bdf4737f976b002df98faf84e20e497f8e37c4c"
    )


def test_build_document_id_differs_for_different_paths() -> None:
    assert build_document_id("runbooks/a.md") != build_document_id("incidents/a.md")


def test_build_document_id_is_repeatable() -> None:
    assert build_document_id("runbooks/a.md") == build_document_id("runbooks/a.md")


def test_knowledge_document_is_immutable() -> None:
    document = KnowledgeDocument(
        document_id="id",
        source_path="runbooks/a.md",
        document_type=DocumentType.RUNBOOK,
        content="# A",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        document.content = "changed"  # type: ignore[misc]


def test_build_chunk_id_is_stable_sha256_of_document_id_and_index() -> None:
    # Fixed vector: guards against algorithm changes and process-dependent hashing.
    assert build_chunk_id("abc", 0) == (
        "5f36efce86f68877cee18fda5637abb2a1bb409f4326cc741fdf13200aa26980"
    )


def test_build_chunk_id_differs_by_index_and_by_document() -> None:
    assert build_chunk_id("abc", 0) != build_chunk_id("abc", 1)
    assert build_chunk_id("abc", 0) != build_chunk_id("abd", 0)


def test_build_chunk_id_is_repeatable() -> None:
    assert build_chunk_id("abc", 3) == build_chunk_id("abc", 3)


def test_document_chunk_is_immutable() -> None:
    chunk = DocumentChunk(
        chunk_id="c",
        document_id="d",
        chunk_index=0,
        source_path="runbooks/a.md",
        document_type=DocumentType.RUNBOOK,
        content="# A",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.content = "changed"  # type: ignore[misc]
