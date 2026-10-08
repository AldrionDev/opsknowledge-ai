import dataclasses

import pytest

from app.ingestion.models import DocumentType, KnowledgeDocument, build_document_id


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
