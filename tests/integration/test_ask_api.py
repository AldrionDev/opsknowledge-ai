"""End to end `POST /ask` on PostgreSQL + pgvector with the real application wiring.

    get_answer_service -> get_retrieval_service -> SemanticRetrievalService
        -> SqlAlchemyChunkRepository -> PostgreSQL / pgvector

No dependency is overridden. Fakes replace only the lowest boundaries, through
the composition root's own factory names: the embedding service factory
(`FakeEmbeddingService(dimension=384)`) and the chat model factory, whose
structured-output runnable is scripted. No API key is set and no model request
is made: the real chat model factory would answer HTTP 503 without a key.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event

from app.answering.models import AnswerDraft, Claim
from app.api import dependencies
from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION
from app.db.session import create_session_factory
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.ingestion.models import DocumentChunk
from app.ingestion.service import IngestionService
from app.main import app
from tests.corpus import write_corpus
from tests.fake_embedding import FakeEmbeddingService
from tests.fake_llm import FakeStructuredChatModel, clear_llm_environment

pytestmark = pytest.mark.integration

POD_MEMORY = (
    "# Pod memory\n\n"
    "The payments pod restarts because the container exceeds its memory limit. "
    "Raise the memory limit in the deployment manifest and redeploy the pod."
)
DISK_FULL = (
    "# Disk full\n\n"
    "The node ran out of disk space because old container images were never "
    "removed. Prune unused container images to free disk space on the node."
)
QUOTE = "the container exceeds its memory limit"


@dataclass
class Wiring:
    """What the composition root created while the test ran."""

    model: FakeStructuredChatModel
    embedding_services: list[FakeEmbeddingService] = field(default_factory=list)
    engines: list[Engine] = field(default_factory=list)
    disposals: list[Engine] = field(default_factory=list)
    chat_models: int = 0


@pytest.fixture
def wiring(
    migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Wiring]:
    """Real wiring on the disposable database, fakes at the lowest boundaries.

    `migrated_engine` points `DATABASE_URL` at the disposable database; that
    setup engine is created outside the composition root and is not counted.
    """
    assert app.dependency_overrides == {}
    clear_llm_environment(monkeypatch)

    wiring = Wiring(model=FakeStructuredChatModel())
    real_create_db_engine = dependencies.create_db_engine

    def create_db_engine(*args: Any, **kwargs: Any) -> Engine:
        engine = real_create_db_engine(*args, **kwargs)
        event.listen(engine, "engine_disposed", wiring.disposals.append)
        wiring.engines.append(engine)
        return engine

    def create_embedding_service() -> FakeEmbeddingService:
        service = FakeEmbeddingService(dimension=EMBEDDING_DIMENSION)
        wiring.embedding_services.append(service)
        return service

    def create_chat_model() -> FakeStructuredChatModel:
        wiring.chat_models += 1
        return wiring.model

    monkeypatch.setattr(dependencies, "_engine", None)
    monkeypatch.setattr(dependencies, "_retrieval_service", None)
    monkeypatch.setattr(dependencies, "_answer_chain", None)
    monkeypatch.setattr(dependencies, "create_db_engine", create_db_engine)
    monkeypatch.setattr(dependencies, "FastEmbedService", create_embedding_service)
    monkeypatch.setattr(dependencies, "create_chat_model", create_chat_model)
    yield wiring
    dependencies.clear_answer_chain()
    dependencies.dispose_retrieval_service()


@pytest.fixture
def target(migrated_engine: Engine, tmp_path: Path) -> DocumentChunk:
    """Ingest a small corpus through the real pipeline; return the pod chunk."""
    corpus = write_corpus(
        tmp_path / "corpus",
        {"runbooks/pod-memory.md": POD_MEMORY, "incidents/disk-full.md": DISK_FULL},
    )
    repository = SqlAlchemyChunkRepository(create_session_factory(migrated_engine))
    embedding_service = FakeEmbeddingService(dimension=EMBEDDING_DIMENSION)
    IngestionService(embedding_service, repository).ingest(corpus)
    [document] = [
        document
        for document in load_knowledge_documents(corpus)
        if document.source_path == "runbooks/pod-memory.md"
    ]
    [chunk] = chunk_document(document)
    return chunk


def ask(client: TestClient, question: str) -> httpx.Response:
    return client.post("/ask", json={"question": question, "limit": 5})


def test_real_retrieval_produces_an_answer_with_validated_citations(
    wiring: Wiring, target: DocumentChunk
) -> None:
    wiring.model.result = AnswerDraft(
        insufficient=False,
        claims=[
            Claim(text="The pod exceeds its memory", source_id=1, evidence_quote=QUOTE)
        ],
    )

    with TestClient(app) as client:
        # Fake embeddings are a pure function of the text, so the chunk's own
        # content retrieves that chunk first: it becomes source 1.
        response = ask(client, target.content)

    assert response.status_code == 200
    assert response.json() == {
        "question": target.content,
        "status": "answered",
        "reason": None,
        "answer": "The pod exceeds its memory [1]",
        "citations": [
            {
                "source_id": 1,
                "chunk_id": target.chunk_id,
                "document_id": target.document_id,
                "source_path": "runbooks/pod-memory.md",
                "document_type": "runbook",
                "chunk_index": target.chunk_index,
                "evidence_quote": QUOTE,
            }
        ],
    }
    prompt = wiring.model.prompt_text()
    assert target.content in prompt
    assert "removed. Prune unused container images" in prompt


def test_model_declining_on_real_context_is_http_200(
    wiring: Wiring, target: DocumentChunk
) -> None:
    wiring.model.result = AnswerDraft(insufficient=True, claims=[])

    with TestClient(app) as client:
        response = ask(client, "What is the on-call rotation of the database team?")

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["reason"] == "model_declined"
    assert len(wiring.model.prompts) == 1


def test_empty_database_is_no_retrieval_without_a_model_call(wiring: Wiring) -> None:
    with TestClient(app) as client:
        response = ask(client, "Why is the payments pod restarting?")

    assert response.status_code == 200
    assert response.json() == {
        "question": "Why is the payments pod restarting?",
        "status": "insufficient_context",
        "reason": "no_retrieval",
        "answer": None,
        "citations": [],
    }
    assert wiring.model.prompts == []


def test_invalid_model_output_on_real_context_is_http_502(
    wiring: Wiring, target: DocumentChunk
) -> None:
    wiring.model.result = AnswerDraft(
        insufficient=False,
        claims=[Claim(text="Made up", source_id=99, evidence_quote=QUOTE)],
    )

    with TestClient(app) as client:
        response = ask(client, target.content)

    assert response.status_code == 502
    assert response.json()["code"] == "unknown_citation"
    assert "Made up" not in response.text


def test_search_and_ask_share_one_embedding_service_and_engine_until_shutdown(
    wiring: Wiring, target: DocumentChunk
) -> None:
    wiring.model.result = AnswerDraft(insufficient=True, claims=[])

    with TestClient(app) as client:
        search = client.post("/search", json={"query": target.content})
        first = ask(client, target.content)
        second = ask(client, target.content)

        assert [r["chunk_id"] for r in search.json()["results"]][0] == target.chunk_id
        assert (first.status_code, second.status_code) == (200, 200)
        assert len(wiring.embedding_services) == 1
        assert len(wiring.engines) == 1
        assert wiring.chat_models == 1
        assert wiring.disposals == []

    [engine] = wiring.engines
    assert wiring.disposals == [engine]
    assert dependencies._engine is None
    assert dependencies._retrieval_service is None
    assert dependencies._answer_chain is None
