from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_retrieval_service
from app.api.search import DEFAULT_RESULT_LIMIT
from app.embeddings.service import EmbeddingError
from app.ingestion.models import DocumentType
from app.main import app
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import MAX_RESULT_LIMIT

QUERY = "Why is the payments pod repeatedly restarting?"


class RecordingRetrievalService:
    """Fake retrieval service: records its calls, returns or raises what it was given.

    It keeps the API tests free of an embedding model, PostgreSQL and pgvector.
    """

    def __init__(
        self,
        results: list[RetrievedChunk] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.failure = failure
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.calls.append((query, limit))
        if self.failure is not None:
            raise self.failure
        return list(self.results)


InstallService = Callable[..., RecordingRetrievalService]


@pytest.fixture
def install_service() -> Iterator[InstallService]:
    """Install a fake retrieval service for the duration of one test."""

    def install(
        results: list[RetrievedChunk] | None = None,
        failure: Exception | None = None,
    ) -> RecordingRetrievalService:
        service = RecordingRetrievalService(results, failure)
        app.dependency_overrides[get_retrieval_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def make_chunk(
    chunk_id: str = "chunk-a",
    *,
    document_id: str = "doc-a",
    chunk_index: int = 0,
    source_path: str = "runbooks/kubernetes-crashloopbackoff.md",
    document_type: DocumentType = DocumentType.RUNBOOK,
    content: str = "Check the pod logs.",
    similarity_score: float = 0.84,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=source_path,
        document_type=document_type,
        content=content,
        similarity_score=similarity_score,
    )


def test_valid_request_returns_http_200(
    client: TestClient, install_service: InstallService
) -> None:
    install_service([make_chunk()])

    response = client.post("/search", json={"query": QUERY, "limit": 5})

    assert response.status_code == 200


def test_query_is_passed_to_the_retrieval_service_unchanged(
    client: TestClient, install_service: InstallService
) -> None:
    service = install_service()

    client.post("/search", json={"query": "  pod keeps restarting \n"})

    assert [query for query, _ in service.calls] == ["  pod keeps restarting \n"]


def test_requested_limit_is_passed_to_the_retrieval_service(
    client: TestClient, install_service: InstallService
) -> None:
    service = install_service()

    client.post("/search", json={"query": QUERY, "limit": 7})

    assert service.calls == [(QUERY, 7)]


def test_default_limit_is_applied_when_the_limit_is_omitted(
    client: TestClient, install_service: InstallService
) -> None:
    service = install_service()

    client.post("/search", json={"query": QUERY})

    assert service.calls == [(QUERY, DEFAULT_RESULT_LIMIT)]


@pytest.mark.parametrize("limit", [1, MAX_RESULT_LIMIT])
def test_limits_at_the_supported_boundaries_are_accepted(
    client: TestClient, install_service: InstallService, limit: int
) -> None:
    service = install_service()

    response = client.post("/search", json={"query": QUERY, "limit": limit})

    assert response.status_code == 200
    assert service.calls == [(QUERY, limit)]


def test_retrieved_chunks_are_serialized_with_their_source_metadata(
    client: TestClient, install_service: InstallService
) -> None:
    install_service(
        [
            make_chunk(
                "chunk-x",
                document_id="doc-x",
                chunk_index=3,
                source_path="incidents/inc-2025-004.md",
                document_type=DocumentType.INCIDENT,
                content="Multi-line\ncontent with ünicode.",
                similarity_score=0.84,
            )
        ]
    )

    response = client.post("/search", json={"query": QUERY})

    assert response.json() == {
        "query": QUERY,
        "results": [
            {
                "chunk_id": "chunk-x",
                "document_id": "doc-x",
                "chunk_index": 3,
                "source_path": "incidents/inc-2025-004.md",
                "document_type": "incident",
                "content": "Multi-line\ncontent with ünicode.",
                "similarity_score": 0.84,
            }
        ],
    }


def test_retrieval_service_result_order_is_preserved(
    client: TestClient, install_service: InstallService
) -> None:
    # Deliberately not ordered by score: the API must not reorder the results.
    install_service(
        [
            make_chunk("chunk-a", similarity_score=0.42),
            make_chunk("chunk-b", similarity_score=0.91),
            make_chunk("chunk-c", similarity_score=0.13),
        ]
    )

    response = client.post("/search", json={"query": QUERY})

    results = response.json()["results"]
    assert [result["chunk_id"] for result in results] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


@pytest.mark.parametrize("similarity_score", [1.0, 0.75, 0.0, -0.5])
def test_similarity_scores_are_returned_unchanged(
    client: TestClient, install_service: InstallService, similarity_score: float
) -> None:
    install_service([make_chunk(similarity_score=similarity_score)])

    response = client.post("/search", json={"query": QUERY})

    [result] = response.json()["results"]
    assert result["similarity_score"] == pytest.approx(similarity_score)


def test_empty_retrieval_result_returns_http_200_and_an_empty_result_list(
    client: TestClient, install_service: InstallService
) -> None:
    install_service([])

    response = client.post("/search", json={"query": QUERY})

    assert response.status_code == 200
    assert response.json() == {"query": QUERY, "results": []}


def test_missing_query_is_rejected(
    client: TestClient, install_service: InstallService
) -> None:
    service = install_service()

    response = client.post("/search", json={"limit": 5})

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("query", ["", " ", "   \n\t  "])
def test_empty_or_whitespace_only_query_is_rejected(
    client: TestClient, install_service: InstallService, query: str
) -> None:
    service = install_service()

    response = client.post("/search", json={"query": query})

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("limit", [0, -1, MAX_RESULT_LIMIT + 1])
def test_limit_outside_the_supported_range_is_rejected(
    client: TestClient, install_service: InstallService, limit: int
) -> None:
    service = install_service()

    response = client.post("/search", json={"query": QUERY, "limit": limit})

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    ("failure", "secrets"),
    [
        (
            # Fabricated message with the kind of detail a database failure
            # carries: credentials, host, schema and SQL.
            RuntimeError(
                "connection to postgresql+psycopg://api_user:fake-test-password"
                "@db:5432/opsknowledge failed: SELECT document_chunks.chunk_id"
            ),
            [
                "fake-test-password",
                "postgresql+psycopg",
                "document_chunks",
                "SELECT",
            ],
        ),
        (
            EmbeddingError("Cannot load embedding model /home/user/.cache/fastembed"),
            ["/home/user/.cache/fastembed", "embedding model"],  # model path
        ),
    ],
)
def test_internal_retrieval_failure_does_not_leak_internal_details(
    install_service: InstallService, failure: Exception, secrets: list[str]
) -> None:
    install_service(failure=failure)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/search", json={"query": QUERY})

    assert response.status_code == 500
    body = response.text
    for secret in secrets:
        assert secret not in body
    assert str(failure) not in body
    assert type(failure).__name__ not in body
    assert "Traceback" not in body


def test_internal_retrieval_failure_returns_a_structured_json_error(
    install_service: InstallService,
) -> None:
    install_service(failure=RuntimeError("database failure"))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/search", json={"query": QUERY})

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Internal server error"}


def test_validation_errors_keep_their_own_response_shape(
    client: TestClient, install_service: InstallService
) -> None:
    install_service()

    response = client.post("/search", json={"query": QUERY, "limit": 0})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_interactive_documentation_is_served(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/openapi.json" in response.text


def test_search_and_its_schemas_appear_in_the_openapi_document(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()

    assert "post" in schema["paths"]["/search"]
    assert "SearchRequest" in schema["components"]["schemas"]
    assert "SearchResponse" in schema["components"]["schemas"]
    assert "SearchResult" in schema["components"]["schemas"]


def test_health_is_unaffected_by_the_search_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
