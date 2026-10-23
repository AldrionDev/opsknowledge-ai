"""Cost bounds, the request limiter, provider failure kinds and the recorders."""

from typing import Any

import anthropic
import httpx2
import pytest
from langchain_anthropic import chat_models as anthropic_chat_models
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.runnables import RunnableLambda

from app.answering.models import AnswerDraft
from app.answering.prompt import ANSWER_PROMPT, build_answer_chain
from app.answering.service import AnswerService
from app.evaluation.guards import (
    MAX_OUTPUT_TOKENS_CEILING,
    MAX_REQUESTS_CEILING,
    MAX_RETRIES_CEILING,
    BoundsError,
    CostBounds,
    ProviderFailureKind,
    RecordingRetrievalService,
    RequestCapReachedError,
    RequestLimiter,
    TokenUsage,
    UsageRecorder,
    classify_provider_failure,
)
from app.llm.errors import LLMError
from tests.evaluation.support import MappedRetrieval, chunk
from tests.fake_llm import FakeStructuredChatModel

DECLINED = AnswerDraft(insufficient=True, claims=[])
USAGE = {"input_tokens": 1200, "output_tokens": 85, "total_tokens": 1285}


def status_error(error_type: type[Exception], status_code: int) -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request)
    return error_type(message="provider text", response=response, body=None)


# --- cost bounds -------------------------------------------------------------


def test_the_absolute_ceilings_are_fixed() -> None:
    assert (MAX_REQUESTS_CEILING, MAX_RETRIES_CEILING, MAX_OUTPUT_TOKENS_CEILING) == (
        30,
        2,
        2048,
    )


def test_defaults_and_the_worst_case_output_token_bound() -> None:
    bounds = CostBounds(max_requests=24)

    assert (bounds.max_retries, bounds.max_output_tokens) == (1, 1024)
    assert bounds.worst_case_output_tokens == 24 * 2 * 1024 == 49152
    assert CostBounds(30, 2, 2048).worst_case_output_tokens == 184320


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": 1, "max_retries": 0, "max_output_tokens": 1},
        {"max_requests": 30, "max_retries": 2, "max_output_tokens": 2048},
    ],
)
def test_bounds_at_the_limits_are_accepted(kwargs: dict[str, int]) -> None:
    CostBounds(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": 31},
        {"max_requests": 0},
        {"max_requests": 24, "max_retries": 3},
        {"max_requests": 24, "max_retries": -1},
        {"max_requests": 24, "max_output_tokens": 2049},
        {"max_requests": 24, "max_output_tokens": 0},
        {"max_requests": True},
        {"max_requests": 24, "max_retries": 1.0},
    ],
)
def test_bounds_outside_the_ceilings_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(BoundsError):
        CostBounds(**kwargs)


# --- request limiter ---------------------------------------------------------


class CountingChain:
    def __init__(
        self, result: Any = DECLINED, failure: Exception | None = None
    ) -> None:
        self.calls = 0
        self.result = result
        self.failure = failure

    def __call__(self, inputs: dict[str, Any]) -> Any:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.result


def test_the_limiter_refuses_before_the_chain_is_called() -> None:
    chain = CountingChain()
    limiter = RequestLimiter(RunnableLambda(chain), max_requests=2)

    limiter({})
    limiter({})
    with pytest.raises(RequestCapReachedError):
        limiter({})

    assert chain.calls == 2
    assert limiter.invocations == 2


def test_the_cap_error_passes_through_the_answer_service_untranslated() -> None:
    limiter = RequestLimiter(RunnableLambda(CountingChain()), max_requests=1)
    service = AnswerService(
        MappedRetrieval({"q1": [chunk("c1")], "q2": [chunk("c1")]}),
        limiter.as_runnable(),
    )

    service.answer("q1", 5)
    with pytest.raises(RequestCapReachedError):
        service.answer("q2", 5)


def test_the_limiter_reraises_the_same_exception_and_keeps_only_the_kind() -> None:
    failure = ModelAuthenticationError("secret provider text")
    limiter = RequestLimiter(RunnableLambda(CountingChain(failure=failure)), 5)

    with pytest.raises(ModelAuthenticationError) as caught:
        limiter({})

    assert caught.value is failure
    assert limiter.last_failure_kind is ProviderFailureKind.NONRECOVERABLE
    assert all(value is not failure for value in vars(limiter).values())


def test_a_successful_call_clears_the_previous_failure_kind() -> None:
    chain = CountingChain(failure=ModelRateLimitError("x"))
    limiter = RequestLimiter(RunnableLambda(chain), 5)
    with pytest.raises(ModelRateLimitError):
        limiter({})

    chain.failure = None
    limiter({})

    assert limiter.last_failure_kind is None


# --- provider failure kinds ----------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        ModelAuthenticationError("x"),
        ModelPermissionDeniedError("x"),
        ModelNotFoundError("x"),
        ModelInvalidRequestError("x"),
        ContextOverflowError("x"),
        status_error(anthropic_chat_models.AnthropicAuthenticationError, 401),
        status_error(anthropic_chat_models.AnthropicPermissionDeniedError, 403),
        status_error(anthropic_chat_models.AnthropicInvalidRequestError, 400),
        status_error(anthropic_chat_models.AnthropicModelNotFoundError, 404),
        status_error(anthropic_chat_models.AnthropicContextOverflowError, 400),
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.BadRequestError, 400),
        status_error(anthropic.RequestTooLargeError, 413),
        status_error(anthropic.UnprocessableEntityError, 422),
    ],
)
def test_nonrecoverable_provider_failures(failure: Exception) -> None:
    assert classify_provider_failure(failure) is ProviderFailureKind.NONRECOVERABLE


@pytest.mark.parametrize(
    "failure",
    [
        ModelRateLimitError("x"),
        ModelTimeoutError("x"),
        ModelConnectionError("x"),
        ModelAPIError("x"),
        status_error(anthropic_chat_models.AnthropicRateLimitError, 429),
        status_error(anthropic_chat_models.AnthropicAPIError, 500),
        status_error(anthropic_chat_models.AnthropicOverloadedError, 529),
        status_error(anthropic.ConflictError, 409),
        status_error(anthropic.InternalServerError, 500),
    ],
)
def test_transient_provider_failures(failure: Exception) -> None:
    assert classify_provider_failure(failure) is ProviderFailureKind.TRANSIENT


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("x"),
        OutputParserException("x"),
        RuntimeError("x"),
        LLMError("an application error is not a raw provider failure"),
    ],
)
def test_other_exceptions_are_not_provider_failures(failure: Exception) -> None:
    assert classify_provider_failure(failure) is None


# --- retrieval recording ---------------------------------------------------------


def test_the_recorder_delegates_unchanged_and_captures_a_copy() -> None:
    inner = MappedRetrieval({"q": [chunk("c1"), chunk("c2")]})
    recorder = RecordingRetrievalService(inner)

    returned = recorder.retrieve("q", 5)
    returned.append(chunk("c3"))

    assert inner.calls == ["q"]
    assert [c.chunk_id for c in recorder.take() or []] == ["c1", "c2"]
    assert recorder.take() is None


def test_a_failed_retrieval_captures_nothing() -> None:
    recorder = RecordingRetrievalService(MappedRetrieval({}))

    with pytest.raises(KeyError):
        recorder.retrieve("unknown", 5)

    assert recorder.take() is None


def test_more_than_one_retrieval_per_question_is_a_defect() -> None:
    recorder = RecordingRetrievalService(MappedRetrieval({"q": [chunk("c1")]}))
    recorder.retrieve("q", 5)
    recorder.retrieve("q", 5)

    with pytest.raises(RuntimeError, match="expected one retrieval"):
        recorder.take()


# --- usage recording -------------------------------------------------------------


def test_usage_is_read_from_the_message_without_a_model_name() -> None:
    recorder = UsageRecorder()
    with_usage = ChatGeneration(message=AIMessage(content="x", usage_metadata=USAGE))
    without_usage = ChatGeneration(message=AIMessage(content="y"))

    recorder.on_llm_end(LLMResult(generations=[[with_usage], [without_usage]]))
    recorder.on_llm_end(LLMResult(generations=[[Generation(text="plain")]]))

    assert recorder.responses == [TokenUsage(1200, 85)]


def test_usage_is_collected_through_with_config_on_the_chain() -> None:
    recorder = UsageRecorder()
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="{}", usage_metadata=USAGE)])
    )
    chain = ANSWER_PROMPT | model | RunnableLambda(lambda _: DECLINED)
    limiter = RequestLimiter(chain.with_config(callbacks=[recorder]), 5)
    service = AnswerService(
        MappedRetrieval({"q": [chunk("c1")]}), limiter.as_runnable()
    )

    service.answer("q", 5)

    assert recorder.responses == [TokenUsage(1200, 85)]


def test_a_chain_without_a_chat_model_reports_no_usage() -> None:
    recorder = UsageRecorder()
    chain = build_answer_chain(FakeStructuredChatModel(result=DECLINED))  # type: ignore[arg-type]
    service = AnswerService(
        MappedRetrieval({"q": [chunk("c1")]}),
        chain.with_config(callbacks=[recorder]),
    )

    service.answer("q", 5)

    assert recorder.responses == []
