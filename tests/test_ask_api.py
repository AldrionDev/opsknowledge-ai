from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)

from app.answering.models import AnswerDraft, Claim, InvalidOutputCode
from app.answering.prompt import build_answer_chain
from app.answering.service import AnswerService
from app.api import dependencies
from app.api.ask import (
    LLM_NOT_CONFIGURED_MESSAGE,
    LLM_UNAVAILABLE_MESSAGE,
    MAX_QUESTION_LENGTH,
)
from app.api.dependencies import get_answer_service, get_retrieval_service
from app.api.search import DEFAULT_RESULT_LIMIT
from app.ingestion.models import DocumentType
from app.llm import config
from app.main import app
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import MAX_RESULT_LIMIT
from tests.fake_llm import (
    FakeStructuredChatModel,
    block_network,
    clear_llm_environment,
)

QUESTION = "SENTINEL-QUESTION-0b7e why does the pod restart?"
PODS = "The pod restarts because the container exceeds its memory limit."
DISK = (
    "SENTINEL-SOURCE-CONTENT-c41d Free disk space by removing unused container images."
)
SENTINEL_OUTPUT = "SENTINEL-MODEL-OUTPUT-9a3f"
SENTINEL_PROVIDER = "SENTINEL-PROVIDER-TEXT-e52c"
SENTINEL_KEY = "sk-ant-SENTINEL-KEY-77d0"
SENTINELS = (
    QUESTION,
    "SENTINEL-QUESTION",
    "SENTINEL-SOURCE-CONTENT",
    SENTINEL_OUTPUT,
    SENTINEL_PROVIDER,
    SENTINEL_KEY,
)

INVALID_OUTPUT_CODES = {code.value for code in InvalidOutputCode}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No LLM configuration, no cached chain and no network for every test."""
    clear_llm_environment(monkeypatch)
    block_network(monkeypatch)
    monkeypatch.setattr(dependencies, "_answer_chain", None)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class FakeRetrieval:
    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.failure = failure
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.calls.append((query, limit))
        if self.failure is not None:
            raise self.failure
        return list(self.chunks)


Install = Callable[..., tuple[FakeRetrieval, FakeStructuredChatModel]]


@pytest.fixture
def install() -> Install:
    """Serve `/ask` from a real `AnswerService` over a fake retrieval and model."""

    def install(
        chunks: list[RetrievedChunk] | None = None,
        result: object = None,
        failure: Exception | None = None,
        retrieval_failure: Exception | None = None,
    ) -> tuple[FakeRetrieval, FakeStructuredChatModel]:
        retrieval = FakeRetrieval(chunks, retrieval_failure)
        model = FakeStructuredChatModel(result, failure)
        service = AnswerService(retrieval, build_answer_chain(model))
        app.dependency_overrides[get_answer_service] = lambda: service
        return retrieval, model

    return install


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    document_type: DocumentType = DocumentType.RUNBOOK,
    chunk_index: int = 0,
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=chunk_index,
        source_path=f"{document_type.value}s/{chunk_id}.md",
        document_type=document_type,
        content=content,
        similarity_score=score,
    )


def two_chunks() -> list[RetrievedChunk]:
    return [
        make_chunk("a", PODS, score=0.9),
        make_chunk(
            "b", DISK, document_type=DocumentType.INCIDENT, chunk_index=3, score=0.8
        ),
    ]


def claim(
    source_id: int = 1,
    quote: str = "container exceeds its memory limit",
    text: str = "The pod runs out of memory",
) -> Claim:
    return Claim(text=text, source_id=source_id, evidence_quote=quote)


def answered_draft() -> AnswerDraft:
    return AnswerDraft(
        insufficient=False,
        claims=[
            claim(1),
            claim(
                2,
                "removing   unused container\nimages",
                "Unused images fill the disk",
            ),
        ],
    )


def assert_no_leak(body: str, failure: Exception | None = None) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in body
    assert "Traceback" not in body
    if failure is not None:
        assert type(failure).__name__ not in body


# --- success ------------------------------------------------------------------


def test_answered_question_returns_the_answer_and_full_citations(
    client: TestClient, install: Install
) -> None:
    install(two_chunks(), answered_draft())

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json() == {
        "question": QUESTION,
        "status": "answered",
        "reason": None,
        "answer": "The pod runs out of memory [1]\nUnused images fill the disk [2]",
        "citations": [
            {
                "source_id": 1,
                "chunk_id": "a",
                "document_id": "doc-a",
                "source_path": "runbooks/a.md",
                "document_type": "runbook",
                "chunk_index": 0,
                "evidence_quote": "container exceeds its memory limit",
            },
            {
                "source_id": 2,
                "chunk_id": "b",
                "document_id": "doc-b",
                "source_path": "incidents/b.md",
                "document_type": "incident",
                "chunk_index": 3,
                "evidence_quote": "removing unused container images",
            },
        ],
    }


def test_the_model_reads_the_question_and_the_retrieved_sources(
    client: TestClient, install: Install
) -> None:
    _, model = install(two_chunks(), answered_draft())

    client.post("/ask", json={"question": QUESTION})

    assert model.structured_output_calls == [(AnswerDraft, "json_schema")]
    prompt = model.prompt_text()
    assert QUESTION in prompt
    assert PODS in prompt
    assert DISK in prompt


def test_question_and_limit_are_passed_to_the_service_unchanged(
    client: TestClient, install: Install
) -> None:
    retrieval, _ = install()

    client.post("/ask", json={"question": "  pod keeps restarting \n", "limit": 7})

    assert retrieval.calls == [("  pod keeps restarting \n", 7)]


def test_default_limit_is_the_search_default(
    client: TestClient, install: Install
) -> None:
    retrieval, _ = install()

    client.post("/ask", json={"question": QUESTION})

    assert retrieval.calls == [(QUESTION, DEFAULT_RESULT_LIMIT)]
    assert DEFAULT_RESULT_LIMIT == 5


@pytest.mark.parametrize(
    ("chunks", "result", "reason", "model_calls"),
    [
        ([], None, "no_retrieval", 0),
        ([make_chunk("blank", "  \n\t ")], None, "empty_context", 0),
        (two_chunks(), AnswerDraft(insufficient=True, claims=[]), "model_declined", 1),
    ],
    ids=["no_retrieval", "empty_context", "model_declined"],
)
def test_genuine_insufficient_context_is_http_200(
    client: TestClient,
    install: Install,
    chunks: list[RetrievedChunk],
    result: object,
    reason: str,
    model_calls: int,
) -> None:
    _, model = install(chunks, result)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json() == {
        "question": QUESTION,
        "status": "insufficient_context",
        "reason": reason,
        "answer": None,
        "citations": [],
    }
    assert len(model.prompts) == model_calls


# --- request validation ---------------------------------------------------------


def test_missing_question_is_rejected(client: TestClient, install: Install) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"limit": 5})

    assert response.status_code == 422
    assert retrieval.calls == []


@pytest.mark.parametrize("question", ["", " ", "   \n\t  "])
def test_empty_or_whitespace_only_question_is_rejected(
    client: TestClient, install: Install, question: str
) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert retrieval.calls == []


@pytest.mark.parametrize(
    "question",
    ["x" * MAX_QUESTION_LENGTH, " " + "x" * (MAX_QUESTION_LENGTH - 2) + " "],
    ids=["max-length", "max-length-with-padding"],
)
def test_question_at_the_maximum_length_is_accepted(
    client: TestClient, install: Install, question: str
) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert retrieval.calls == [(question, DEFAULT_RESULT_LIMIT)]


@pytest.mark.parametrize(
    "question",
    ["x" * (MAX_QUESTION_LENGTH + 1), "x" * MAX_QUESTION_LENGTH + " "],
    ids=["too-long", "too-long-by-trailing-space"],
)
def test_question_over_the_maximum_length_is_rejected(
    client: TestClient, install: Install, question: str
) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert retrieval.calls == []


@pytest.mark.parametrize("limit", [1, MAX_RESULT_LIMIT])
def test_limits_at_the_supported_boundaries_are_accepted(
    client: TestClient, install: Install, limit: int
) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"question": QUESTION, "limit": limit})

    assert response.status_code == 200
    assert retrieval.calls == [(QUESTION, limit)]


@pytest.mark.parametrize("limit", [0, -1, MAX_RESULT_LIMIT + 1])
def test_limit_outside_the_supported_range_is_rejected(
    client: TestClient, install: Install, limit: int
) -> None:
    retrieval, _ = install()

    response = client.post("/ask", json={"question": QUESTION, "limit": limit})

    assert response.status_code == 422
    assert retrieval.calls == []


# --- answering errors -----------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "failure", "code"),
    [
        (
            None,
            OutputParserException(
                f"Invalid json output: {SENTINEL_OUTPUT}", llm_output=SENTINEL_OUTPUT
            ),
            InvalidOutputCode.MALFORMED_OUTPUT,
        ),
        ({"text": SENTINEL_OUTPUT}, None, InvalidOutputCode.MALFORMED_OUTPUT),
        (
            AnswerDraft(insufficient=False, claims=[claim(99, text=SENTINEL_OUTPUT)]),
            None,
            InvalidOutputCode.UNKNOWN_CITATION,
        ),
        (
            AnswerDraft(
                insufficient=False,
                claims=[claim(1, f"{SENTINEL_OUTPUT} is quoted from nowhere")],
            ),
            None,
            InvalidOutputCode.MISSING_EVIDENCE,
        ),
        (
            AnswerDraft(insufficient=True, claims=[claim(1, text=SENTINEL_OUTPUT)]),
            None,
            InvalidOutputCode.INCONSISTENT_RESPONSE,
        ),
    ],
    ids=[
        "malformed-parser-failure",
        "malformed-not-a-draft",
        "unknown-citation",
        "missing-evidence",
        "inconsistent-response",
    ],
)
def test_invalid_model_output_is_http_502_with_its_code(
    client: TestClient,
    install: Install,
    result: object,
    failure: Exception | None,
    code: InvalidOutputCode,
) -> None:
    install(two_chunks(), result, failure)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"detail", "code"}
    assert body["code"] == code.value
    assert body["detail"]
    assert_no_leak(response.text, failure)


@pytest.mark.parametrize(
    "failure",
    [
        ModelRateLimitError(SENTINEL_PROVIDER),
        ModelTimeoutError(SENTINEL_PROVIDER),
        ModelConnectionError(SENTINEL_PROVIDER),
        ModelAuthenticationError(f"{SENTINEL_PROVIDER} key {SENTINEL_KEY}"),
    ],
    ids=["rate-limit", "timeout", "connection", "authentication"],
)
def test_provider_failure_is_http_502_llm_unavailable(
    client: TestClient, install: Install, failure: Exception
) -> None:
    install(two_chunks(), failure=failure)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 502
    assert response.json() == {
        "detail": LLM_UNAVAILABLE_MESSAGE,
        "code": "llm_unavailable",
    }
    assert_no_leak(response.text, failure)


@pytest.mark.parametrize(
    "environment",
    [{}, {config.API_KEY_ENV: SENTINEL_KEY, config.TIMEOUT_ENV: "not-a-number"}],
    ids=["missing-api-key", "invalid-setting"],
)
def test_missing_llm_configuration_is_http_503_llm_not_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    # The real `get_answer_service` runs; only retrieval is replaced.
    retrieval = FakeRetrieval(two_chunks())
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert response.json() == {
        "detail": LLM_NOT_CONFIGURED_MESSAGE,
        "code": "llm_not_configured",
    }
    assert config.API_KEY_ENV not in response.text
    assert config.TIMEOUT_ENV not in response.text
    assert_no_leak(response.text)
    assert retrieval.calls == []
    assert dependencies._answer_chain is None


@pytest.mark.parametrize(
    ("chain_failure", "retrieval_failure"),
    [
        (None, RuntimeError(f"database failure {SENTINEL_PROVIDER}")),
        (RuntimeError(f"bug {SENTINEL_OUTPUT}"), None),
    ],
    ids=["retrieval-failure", "unmapped-chain-failure"],
)
def test_unexpected_failure_keeps_the_generic_500_response(
    install: Install,
    chain_failure: Exception | None,
    retrieval_failure: Exception | None,
) -> None:
    install(two_chunks(), failure=chain_failure, retrieval_failure=retrieval_failure)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Internal server error"}
    assert_no_leak(response.text, chain_failure or retrieval_failure)


# --- wiring without LLM configuration -------------------------------------------


def test_ask_uses_the_retrieval_service_of_the_retrieval_dependency(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = FakeRetrieval(two_chunks())
    model = FakeStructuredChatModel(answered_draft())
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    monkeypatch.setattr(dependencies, "create_chat_model", lambda: model)

    response = client.post("/ask", json={"question": QUESTION, "limit": 2})

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert retrieval.calls == [(QUESTION, 2)]


def test_application_and_search_work_without_llm_configuration() -> None:
    retrieval = FakeRetrieval([make_chunk("a", PODS)])
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval

    with TestClient(app) as client:
        health = client.get("/health")
        schema = client.get("/openapi.json")
        search = client.post("/search", json={"query": "pod restarts"})

    assert health.json() == {"status": "ok"}
    assert schema.status_code == 200
    assert search.status_code == 200
    assert [result["chunk_id"] for result in search.json()["results"]] == ["a"]
    assert dependencies._answer_chain is None


# --- OpenAPI ----------------------------------------------------------------------


def resolve(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref is None:
        return schema
    return components[ref.rsplit("/", 1)[-1]]


def allowed_values(schema: dict[str, Any], components: dict[str, Any]) -> set[str]:
    """String values a schema admits through `$ref`, `anyOf`, `enum` and `const`."""
    schema = resolve(schema, components)
    values: set[str] = set(schema.get("enum", []))
    if "const" in schema:
        values.add(schema["const"])
    for option in schema.get("anyOf", []):
        values |= allowed_values(option, components)
    return values


@pytest.fixture
def openapi(client: TestClient) -> dict[str, Any]:
    return client.get("/openapi.json").json()


def response_schema(openapi: dict[str, Any], status: str) -> tuple[str, dict[str, Any]]:
    response = openapi["paths"]["/ask"]["post"]["responses"][status]
    ref = response["content"]["application/json"]["schema"]["$ref"]
    name = ref.rsplit("/", 1)[-1]
    return name, openapi["components"]["schemas"][name]


def test_ask_and_its_schemas_appear_in_the_openapi_document(
    openapi: dict[str, Any],
) -> None:
    operation = openapi["paths"]["/ask"]["post"]
    schemas = openapi["components"]["schemas"]

    assert set(operation["responses"]) == {"200", "422", "500", "502", "503"}
    assert response_schema(openapi, "200")[0] == "AskResponse"
    assert response_schema(openapi, "422")[0] == "HTTPValidationError"
    assert response_schema(openapi, "502")[0] == "UpstreamErrorResponse"
    assert response_schema(openapi, "503")[0] == "ServiceUnavailableResponse"
    assert response_schema(openapi, "500")[0] == "InternalErrorResponse"
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_ref["$ref"].endswith("/AskRequest")
    assert "AskCitation" in schemas


def test_openapi_request_schema_documents_the_bounds(
    openapi: dict[str, Any],
) -> None:
    request = openapi["components"]["schemas"]["AskRequest"]

    assert request["required"] == ["question"]
    assert request["properties"]["question"]["minLength"] == 1
    assert request["properties"]["question"]["maxLength"] == MAX_QUESTION_LENGTH
    limit = request["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (
        1,
        MAX_RESULT_LIMIT,
        DEFAULT_RESULT_LIMIT,
    )


def test_openapi_success_schema_documents_the_answer_contract(
    openapi: dict[str, Any],
) -> None:
    components = openapi["components"]["schemas"]
    response = components["AskResponse"]
    properties = response["properties"]
    citation = components["AskCitation"]

    assert set(response["required"]) == {
        "question",
        "status",
        "reason",
        "answer",
        "citations",
    }
    assert allowed_values(properties["status"], components) == {
        "answered",
        "insufficient_context",
    }
    assert allowed_values(properties["reason"], components) == {
        "no_retrieval",
        "empty_context",
        "model_declined",
    }
    assert {"type": "null"} in properties["reason"]["anyOf"]
    assert {"type": "null"} in properties["answer"]["anyOf"]
    assert properties["citations"]["items"]["$ref"].endswith("/AskCitation")
    assert set(citation["required"]) == {
        "source_id",
        "chunk_id",
        "document_id",
        "source_path",
        "document_type",
        "chunk_index",
        "evidence_quote",
    }


def test_openapi_error_schemas_document_the_codes(openapi: dict[str, Any]) -> None:
    components = openapi["components"]["schemas"]
    _, upstream = response_schema(openapi, "502")
    _, unavailable = response_schema(openapi, "503")
    _, internal = response_schema(openapi, "500")

    assert set(upstream["required"]) == {"detail", "code"}
    assert allowed_values(upstream["properties"]["code"], components) == (
        INVALID_OUTPUT_CODES | {"llm_unavailable"}
    )
    assert set(unavailable["required"]) == {"detail", "code"}
    assert allowed_values(unavailable["properties"]["code"], components) == {
        "llm_not_configured"
    }
    assert set(internal["properties"]) == {"detail"}
