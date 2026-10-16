import ast
import socket
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from langchain_anthropic import chat_models as anthropic_chat_models
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.answering.errors import InvalidModelOutputError
from app.answering.models import (
    AnswerDraft,
    AnswerStatus,
    Claim,
    InsufficientReason,
    InvalidOutputCode,
)
from app.answering.prompt import ANSWER_PROMPT
from app.answering.service import AnswerChain, AnswerService
from app.embeddings.service import EmbeddingError
from app.ingestion.models import DocumentType
from app.llm.errors import LLMError
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import InvalidRetrievalRequestError

ANSWERING_DIR = Path(__file__).resolve().parent.parent / "app" / "answering"

QUESTION = "SENTINEL-QUESTION-5d1e why does the pod restart?"
PODS = "The pod restarts because the container exceeds its memory limit."
DISK = "Free disk space by removing unused container images from the node."
SENTINEL_OUTPUT = "SENTINEL-MODEL-OUTPUT-8c2a"
SENTINEL_PROVIDER = "SENTINEL-PROVIDER-TEXT-3f9b"
SENTINEL_SOURCE = "SENTINEL-SOURCE-CONTENT-71aa"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key and no network: a test that tries either fails."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def make_chunk(chunk_id: str, content: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        source_path=f"runbooks/{chunk_id}.md",
        document_type=DocumentType.RUNBOOK,
        content=content,
        similarity_score=score,
    )


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


class ScriptedChain:
    """An injectable runnable that records its inputs and returns or raises."""

    def __init__(self, result: object = None, failure: Exception | None = None) -> None:
        self.result = result
        self.failure = failure
        self.inputs: list[dict[str, Any]] = []
        self.runnable: AnswerChain = RunnableLambda(self._invoke)

    def _invoke(self, inputs: dict[str, Any]) -> Any:
        self.inputs.append(inputs)
        if self.failure is not None:
            raise self.failure
        return self.result


def service_with(
    chunks: list[RetrievedChunk] | None = None,
    result: object = None,
    failure: Exception | None = None,
    retrieval_failure: Exception | None = None,
) -> tuple[AnswerService, FakeRetrieval, ScriptedChain]:
    retrieval = FakeRetrieval(chunks, retrieval_failure)
    chain = ScriptedChain(result, failure)
    return AnswerService(retrieval, chain.runnable), retrieval, chain


def two_chunks() -> list[RetrievedChunk]:
    return [make_chunk("a", PODS, 0.9), make_chunk("b", DISK, 0.8)]


def claim(
    source_id: int = 1, quote: str = "container exceeds its memory limit"
) -> Claim:
    return Claim(
        text="The pod runs out of memory", source_id=source_id, evidence_quote=quote
    )


def formatted(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


# --- orchestration ---------------------------------------------------------


def test_answered_with_validated_citations() -> None:
    draft = AnswerDraft(
        insufficient=False,
        claims=[claim(1), claim(2, "removing unused container images")],
    )
    service, retrieval, chain = service_with(two_chunks(), draft)

    answer = service.answer(QUESTION, 5)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.reason is None
    assert answer.answer is not None
    assert [c.chunk_id for c in answer.citations] == ["a", "b"]
    assert len(chain.inputs) == 1
    assert retrieval.calls == [(QUESTION, 5)]


def test_chain_receives_question_numbered_sources_and_count() -> None:
    draft = AnswerDraft(insufficient=False, claims=[claim()])
    service, _, chain = service_with(two_chunks(), draft)

    service.answer(QUESTION, 5)

    (inputs,) = chain.inputs
    assert inputs["question"] == QUESTION
    assert inputs["source_count"] == 2
    assert PODS in inputs["sources"] and DISK in inputs["sources"]


def test_no_retrieval_never_invokes_the_model() -> None:
    service, retrieval, chain = service_with(
        [], AnswerDraft(insufficient=True, claims=[])
    )

    answer = service.answer(QUESTION, 5)

    assert answer.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert answer.reason is InsufficientReason.NO_RETRIEVAL
    assert answer.answer is None and answer.citations == ()
    assert chain.inputs == []
    assert len(retrieval.calls) == 1


@pytest.mark.parametrize(
    "chunks",
    [
        [make_chunk("blank", "   \n\t ")],
        [make_chunk("huge", "x" * 20_000)],
        [make_chunk("huge", "x" * 20_000, 0.9), make_chunk("small", PODS, 0.5)],
    ],
    ids=[
        "blank-only",
        "first-chunk-exceeds-budget",
        "ranked-prefix-stops-at-oversized",
    ],
)
def test_empty_context_never_invokes_the_model(chunks: list[RetrievedChunk]) -> None:
    service, _, chain = service_with(chunks, AnswerDraft(insufficient=True, claims=[]))

    answer = service.answer(QUESTION, 5)

    assert answer.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert answer.reason is InsufficientReason.EMPTY_CONTEXT
    assert chain.inputs == []


def test_model_declined_is_a_genuine_insufficient_context() -> None:
    service, _, chain = service_with(
        two_chunks(), AnswerDraft(insufficient=True, claims=[])
    )

    answer = service.answer(QUESTION, 5)

    assert answer.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert answer.reason is InsufficientReason.MODEL_DECLINED
    assert len(chain.inputs) == 1


# --- invalid output --------------------------------------------------------


@pytest.mark.parametrize(
    ("draft", "code"),
    [
        (
            AnswerDraft(insufficient=False, claims=[claim(9)]),
            InvalidOutputCode.UNKNOWN_CITATION,
        ),
        (
            AnswerDraft(
                insufficient=False,
                claims=[claim(1, "not in the source at all, really")],
            ),
            InvalidOutputCode.MISSING_EVIDENCE,
        ),
        (
            AnswerDraft(insufficient=False, claims=[claim(1, "too short")]),
            InvalidOutputCode.MISSING_EVIDENCE,
        ),
        (
            AnswerDraft(insufficient=True, claims=[claim()]),
            InvalidOutputCode.INCONSISTENT_RESPONSE,
        ),
        (
            AnswerDraft(insufficient=False, claims=[]),
            InvalidOutputCode.INCONSISTENT_RESPONSE,
        ),
        (
            AnswerDraft(
                insufficient=False,
                claims=[
                    Claim(
                        text="two\nlines",
                        source_id=1,
                        evidence_quote="container exceeds its memory limit",
                    )
                ],
            ),
            InvalidOutputCode.INCONSISTENT_RESPONSE,
        ),
        (
            AnswerDraft(
                insufficient=False,
                claims=[
                    Claim(
                        text="marker [1]",
                        source_id=1,
                        evidence_quote="container exceeds its memory limit",
                    )
                ],
            ),
            InvalidOutputCode.INCONSISTENT_RESPONSE,
        ),
        ({"insufficient": True, "claims": []}, InvalidOutputCode.MALFORMED_OUTPUT),
        (None, InvalidOutputCode.MALFORMED_OUTPUT),
    ],
    ids=[
        "unknown-citation",
        "missing-evidence",
        "short-evidence",
        "declined-with-claims",
        "no-claims-not-declined",
        "multiline-claim",
        "marker-in-claim",
        "not-an-answer-draft-dict",
        "not-an-answer-draft-none",
    ],
)
def test_invalid_output_raises_its_code_and_is_never_insufficient_context(
    draft: object, code: InvalidOutputCode
) -> None:
    service, _, chain = service_with(two_chunks(), draft)

    with pytest.raises(InvalidModelOutputError) as info:
        service.answer(QUESTION, 5)

    assert info.value.code is code
    assert len(chain.inputs) == 1


def _validation_error() -> ValidationError:
    try:
        AnswerDraft.model_validate({"insufficient": "yes", "claims": []})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


@pytest.mark.parametrize(
    "failure",
    [
        OutputParserException(
            f"Invalid json output: {SENTINEL_OUTPUT}", llm_output=SENTINEL_OUTPUT
        ),
        _validation_error(),
    ],
    ids=["output-parser-exception", "validation-error"],
)
def test_parsing_failures_become_malformed_output_without_leaking(
    failure: Exception,
) -> None:
    service, _, _ = service_with(two_chunks(), failure=failure)

    with pytest.raises(InvalidModelOutputError) as info:
        service.answer(QUESTION, 5)

    error = info.value
    assert error.code is InvalidOutputCode.MALFORMED_OUTPUT
    assert not isinstance(error, LLMError)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert SENTINEL_OUTPUT not in str(error) + formatted(error)


def test_real_parser_failure_on_unparseable_text_is_malformed_output() -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content=f"not json {SENTINEL_OUTPUT}")])
    )
    chain = ANSWER_PROMPT | model | PydanticOutputParser(pydantic_object=AnswerDraft)
    service = AnswerService(FakeRetrieval(two_chunks()), chain)

    with pytest.raises(InvalidModelOutputError) as info:
        service.answer(QUESTION, 5)

    assert info.value.code is InvalidOutputCode.MALFORMED_OUTPUT
    assert SENTINEL_OUTPUT not in formatted(info.value)
    assert info.value.__context__ is None


@pytest.mark.parametrize(
    "content",
    ['{"insufficient": "yes", "claims": []}', '{"insufficient": true}', "[]", ""],
)
def test_real_parser_schema_mismatch_is_malformed_output(content: str) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content=content)]))
    chain = ANSWER_PROMPT | model | PydanticOutputParser(pydantic_object=AnswerDraft)
    service = AnswerService(FakeRetrieval(two_chunks()), chain)

    with pytest.raises(InvalidModelOutputError) as info:
        service.answer(QUESTION, 5)

    assert info.value.code is InvalidOutputCode.MALFORMED_OUTPUT


def test_real_parser_with_a_valid_draft_is_answered() -> None:
    content = '{"insufficient": false, "claims": [{"text": "OOM", "source_id": 1, "evidence_quote": "container exceeds its memory limit"}]}'
    model = GenericFakeChatModel(messages=iter([AIMessage(content=content)]))
    chain = ANSWER_PROMPT | model | PydanticOutputParser(pydantic_object=AnswerDraft)
    service = AnswerService(FakeRetrieval(two_chunks()), chain)

    assert service.answer(QUESTION, 5).status is AnswerStatus.ANSWERED


# --- provider failures -----------------------------------------------------


def _response(status_code: int) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status_code, request=request)


def _status_error(error_type: type[Exception], status_code: int) -> Exception:
    return error_type(
        message=SENTINEL_PROVIDER, response=_response(status_code), body=None
    )


PROVIDER_FAILURES: list[Callable[[], Exception]] = [
    lambda: ModelRateLimitError(SENTINEL_PROVIDER),
    lambda: ModelTimeoutError(SENTINEL_PROVIDER),
    lambda: ModelConnectionError(SENTINEL_PROVIDER),
    lambda: ModelAuthenticationError(SENTINEL_PROVIDER),
    lambda: _status_error(anthropic_chat_models.AnthropicRateLimitError, 429),
    lambda: _status_error(anthropic_chat_models.AnthropicAPIError, 500),
    lambda: _status_error(anthropic_chat_models.AnthropicInvalidRequestError, 400),
    lambda: _status_error(anthropic.ConflictError, 409),
    lambda: _status_error(anthropic.UnprocessableEntityError, 422),
]


@pytest.mark.parametrize("make_failure", PROVIDER_FAILURES)
def test_provider_failures_map_to_llm_error_without_leaking(
    make_failure: Callable[[], Exception],
) -> None:
    service, _, chain = service_with(two_chunks(), failure=make_failure())

    with pytest.raises(LLMError) as info:
        service.answer(QUESTION, 5)

    error = info.value
    assert not isinstance(error, InvalidModelOutputError)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert SENTINEL_PROVIDER not in str(error) + formatted(error)
    assert len(chain.inputs) == 1  # no application-level retry


def test_an_llm_error_from_the_runnable_is_kept() -> None:
    failure = LLMError("LLM request timed out")
    service, _, _ = service_with(two_chunks(), failure=failure)

    with pytest.raises(LLMError) as info:
        service.answer(QUESTION, 5)

    assert info.value is failure


# --- errors that must propagate unchanged ----------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        EmbeddingError("embedding model failed"),
        RuntimeError("database unavailable"),
        InvalidRetrievalRequestError("query is empty or whitespace-only"),
    ],
    ids=["embedding-error", "repository-error", "invalid-retrieval-request"],
)
def test_retrieval_failures_propagate_unchanged_without_a_model_call(
    failure: Exception,
) -> None:
    service, _, chain = service_with(retrieval_failure=failure)

    with pytest.raises(type(failure)) as info:
        service.answer(QUESTION, 5)

    assert info.value is failure
    assert chain.inputs == []


@pytest.mark.parametrize(
    "failure",
    [
        KeyError("missing prompt variable"),
        TypeError("programming error"),
        AssertionError("invariant"),
        InvalidModelOutputError(InvalidOutputCode.UNKNOWN_CITATION),
    ],
    ids=["key-error", "type-error", "assertion-error", "coded-error"],
)
def test_other_chain_failures_are_not_hidden_as_llm_errors(failure: Exception) -> None:
    service, _, _ = service_with(two_chunks(), failure=failure)

    with pytest.raises(type(failure)) as info:
        service.answer(QUESTION, 5)

    assert info.value is failure


# --- safety ----------------------------------------------------------------


def test_nothing_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    draft = AnswerDraft(insufficient=False, claims=[claim(9)])
    service, _, _ = service_with([make_chunk("a", PODS + SENTINEL_SOURCE)], draft)

    with pytest.raises(InvalidModelOutputError):
        service.answer(QUESTION, 5)

    assert caplog.records == []


def test_exceptions_carry_no_question_or_source_text() -> None:
    draft = AnswerDraft(insufficient=False, claims=[claim(9, SENTINEL_OUTPUT * 2)])
    service, _, _ = service_with([make_chunk("a", PODS + SENTINEL_SOURCE)], draft)

    with pytest.raises(InvalidModelOutputError) as info:
        service.answer(QUESTION, 5)

    text = str(info.value) + formatted(info.value)
    for sentinel in (SENTINEL_OUTPUT, SENTINEL_SOURCE, "SENTINEL-QUESTION-5d1e"):
        assert sentinel not in text


def test_service_is_reusable_and_stateless() -> None:
    draft = AnswerDraft(insufficient=False, claims=[claim()])
    service, _, chain = service_with(two_chunks(), draft)

    first = service.answer(QUESTION, 5)
    second = service.answer(QUESTION, 5)

    assert first == second
    assert len(chain.inputs) == 2


# --- boundaries ------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


FORBIDDEN_ROOTS = {
    "fastapi",
    "starlette",
    "sqlalchemy",
    "pgvector",
    "psycopg",
    "alembic",
    "anthropic",
    "langchain",
    "langchain_anthropic",
    "langchain_community",
    "langgraph",
}
FORBIDDEN_APP_PREFIXES = (
    "app.db",
    "app.embeddings",
    "app.api",
    "app.ingestion.chunk_repository",
)


@pytest.mark.parametrize("module", ["prompt.py", "service.py"])
def test_modules_keep_to_their_dependency_boundary(module: str) -> None:
    imported = _imported_modules(ANSWERING_DIR / module)

    roots = {name.split(".")[0] for name in imported}
    assert not roots & FORBIDDEN_ROOTS
    assert not any(name.startswith(FORBIDDEN_APP_PREFIXES) for name in imported)


def test_retrieval_is_called_exactly_once() -> None:
    # A second retrieve call would hit the exhausted iterator.
    results: Iterator[list[RetrievedChunk]] = iter([two_chunks()])

    class OneShot:
        def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
            return next(results)

    chain = ScriptedChain(AnswerDraft(insufficient=True, claims=[]))
    answer = AnswerService(OneShot(), chain.runnable).answer(QUESTION, 5)

    assert answer.reason is InsufficientReason.MODEL_DECLINED
