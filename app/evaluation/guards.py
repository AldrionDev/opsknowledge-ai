"""Cost bounds, request limiting and the recorders that observe an evaluation run.

Nothing here changes `AnswerService` or `SemanticRetrievalService`: every piece
wraps an injected dependency behind the interface the answer service already
uses.

Cost bounds
-----------

`CostBounds` holds the three caps of a run and enforces the absolute ceilings in
code, independent of the command line:

- `max_requests`: application-level LLM invocations, `1..MAX_REQUESTS_CEILING`;
- `max_retries`: transport retries of the provider client, `0..MAX_RETRIES_CEILING`;
- `max_output_tokens`: per provider attempt, `1..MAX_OUTPUT_TOKENS_CEILING`.

`worst_case_output_tokens` is `max_requests x (1 + max_retries) x
max_output_tokens`. It bounds **output tokens only**. It is not a cost bound:
input tokens (instructions, schema, the assembled context and the question,
re-sent on every retry) and provider pricing are not part of it.

Request limiting
----------------

`RequestLimiter` wraps the answer chain and refuses invocation `max_requests + 1`
with `RequestCapReachedError` *before* the chain is called. `AnswerService`
translates only output-parsing and provider failures, so that error reaches the
runner unchanged.

The limiter also classifies a provider failure while it is still the raw
exception: `AnswerService` replaces it by an `LLMError` that carries neither cause
nor context. Only the `ProviderFailureKind` is kept, never the exception or its
message, and the exception is re-raised unchanged so the answer service maps it
exactly as in production.

- `NONRECOVERABLE`: the failure depends on credentials, permissions, the model,
  the account or the request shape, so it would repeat for every question:
  authentication, permission, not found, invalid request (HTTP 400, including
  context overflow), HTTP 413 and HTTP 422.
- `TRANSIENT`: every other provider failure: rate limit, timeout, connection,
  server or overload errors, and any other Anthropic API error.

Recorders
---------

`RecordingRetrievalService` delegates `retrieve` unchanged and keeps a copy of
each ranked result for the metrics; it makes no call of its own.

`UsageRecorder` is a LangChain callback handler, attached to the chain with
`with_config(callbacks=[...])`. It records `AIMessage.usage_metadata` of every
completed model response. It does not depend on `response_metadata["model_name"]`,
which LangChain's own `UsageMetadataCallbackHandler` requires and which is not
guaranteed for this provider integration. Failed attempts, transport retries and
timed-out requests produce no response, so their usage is unknown; nothing is
estimated.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import anthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda

from app.answering.service import RetrievalService
from app.retrieval.models import RetrievedChunk

MAX_REQUESTS_CEILING = 30
MAX_RETRIES_CEILING = 2
MAX_OUTPUT_TOKENS_CEILING = 2048

DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_OUTPUT_TOKENS = 1024

MAX_CONSECUTIVE_PROVIDER_ERRORS = 3

WORST_CASE_NOTE = (
    "Bounds output tokens only. It is not a total API cost bound: input tokens "
    "(instructions, schema, the assembled context of up to the fixed character "
    "budget and the question, re-sent on every retry) and provider pricing are "
    "not included."
)


class BoundsError(ValueError):
    """Raised when a cost bound is outside its allowed range."""


@dataclass(frozen=True)
class CostBounds:
    max_requests: int
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        _require_range("max_requests", self.max_requests, 1, MAX_REQUESTS_CEILING)
        _require_range("max_retries", self.max_retries, 0, MAX_RETRIES_CEILING)
        _require_range(
            "max_output_tokens", self.max_output_tokens, 1, MAX_OUTPUT_TOKENS_CEILING
        )

    @property
    def worst_case_output_tokens(self) -> int:
        return self.max_requests * (1 + self.max_retries) * self.max_output_tokens


def _require_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BoundsError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise BoundsError(f"{name} must be between {minimum} and {maximum}")


class RequestCapReachedError(Exception):
    """Raised instead of an LLM invocation once `max_requests` is used up."""


class ProviderFailureKind(StrEnum):
    NONRECOVERABLE = "nonrecoverable"
    TRANSIENT = "transient"


NONRECOVERABLE_PROVIDER_ERRORS: tuple[type[Exception], ...] = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelNotFoundError,
    ModelInvalidRequestError,
    ContextOverflowError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
    anthropic.RequestTooLargeError,
    anthropic.UnprocessableEntityError,
)


def classify_provider_failure(exc: Exception) -> ProviderFailureKind | None:
    """The kind of a provider failure, or `None` if `exc` is not one."""
    if not isinstance(exc, ModelError | anthropic.APIError):
        return None
    if isinstance(exc, NONRECOVERABLE_PROVIDER_ERRORS):
        return ProviderFailureKind.NONRECOVERABLE
    return ProviderFailureKind.TRANSIENT


class RequestLimiter:
    """Counts and caps invocations of `chain`; see the module docstring."""

    def __init__(self, chain: Runnable[dict[str, Any], Any], max_requests: int) -> None:
        self._chain = chain
        self._max_requests = max_requests
        self.invocations = 0
        self.last_failure_kind: ProviderFailureKind | None = None

    def __call__(self, inputs: dict[str, Any]) -> Any:
        if self.invocations >= self._max_requests:
            raise RequestCapReachedError(
                f"refused: the cap of {self._max_requests} LLM requests is reached"
            )
        self.invocations += 1
        self.last_failure_kind = None
        try:
            return self._chain.invoke(inputs)
        except Exception as exc:
            self.last_failure_kind = classify_provider_failure(exc)
            raise

    def as_runnable(self) -> Runnable[dict[str, Any], Any]:
        call: Callable[[dict[str, Any]], Any] = self
        return RunnableLambda(call)


class RecordingRetrievalService:
    """Delegates `retrieve` and captures each ranked result; see the module docstring."""

    def __init__(self, inner: RetrievalService) -> None:
        self._inner = inner
        self._captured: list[list[RetrievedChunk]] = []

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        result = self._inner.retrieve(query, limit)
        self._captured.append(list(result))
        return result

    def take(self) -> list[RetrievedChunk] | None:
        """Return and clear the capture of one question, `None` if retrieval failed.

        Raises `RuntimeError` if more than one retrieval happened since the last
        call: the answer service retrieves exactly once per question.
        """
        captured, self._captured = self._captured, []
        if len(captured) > 1:
            raise RuntimeError(f"expected one retrieval, got {len(captured)}")
        return captured[0] if captured else None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int


class UsageRecorder(BaseCallbackHandler):
    """Records the token usage reported with each completed model response."""

    def __init__(self) -> None:
        super().__init__()
        self.responses: list[TokenUsage] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration) and isinstance(
                    generation.message, AIMessage
                ):
                    usage = generation.message.usage_metadata
                    if usage is not None:
                        self.responses.append(
                            TokenUsage(usage["input_tokens"], usage["output_tokens"])
                        )
