"""Grounded answer service: retrieval, context, structured generation, validation.

    question
        -> RetrievalService.retrieve(question, limit)
        -> no chunks                 -> insufficient_context / no_retrieval
        -> assemble_context
        -> empty context             -> insufficient_context / empty_context
        -> chain.invoke (exactly once)
        -> validate_answer_draft     -> Answer, or InvalidModelOutputError

The model is invoked only when a non-empty context exists, at most once per
call, with no application-level retry: transport retries are the chat model's
own, configured in `app.llm`.

The service is framework-independent and stateless, so one instance can be shared
between threads. It owns no resource: the retrieval service and the runnable are
injected and are not disposed here. It performs no request validation of its own;
`SemanticRetrievalService` rejects a blank question or an invalid limit, and that
error propagates unchanged.

Errors
------

Only the chain invocation is translated; every other step raises as is.

1. `OutputParserException` or a pydantic `ValidationError` (the structured output
   could not be parsed or does not fit the schema) becomes
   `InvalidModelOutputError(MALFORMED_OUTPUT)`. Never an `LLMError`.
2. A provider, transport, rate-limit or timeout failure becomes `LLMError`
   through `map_llm_error`.
3. Any other exception, among them `EmbeddingError`, repository errors,
   an `InvalidModelOutputError` and programming defects, propagates unchanged.

The translated error is raised after the `except` block, so it carries neither a
cause nor a context: the parser exception quotes the raw model output and a
provider exception can echo request details, and neither must reach a log or a
client. Nothing is logged here.
"""

from typing import Any, Protocol

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from app.answering.context import AssembledContext, assemble_context
from app.answering.errors import InvalidModelOutputError
from app.answering.models import (
    Answer,
    AnswerStatus,
    InsufficientReason,
    InvalidOutputCode,
)
from app.answering.prompt import build_prompt_inputs
from app.answering.validation import validate_answer_draft
from app.llm.errors import map_llm_error
from app.retrieval.models import RetrievedChunk

# The runnable maps the prompt variables to an `AnswerDraft`. It is typed with an
# open output because the validator rejects anything that is not a draft.
AnswerChain = Runnable[dict[str, Any], Any]


class RetrievalService(Protocol):
    """What the answer service needs from retrieval; `SemanticRetrievalService` fits."""

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]: ...


class AnswerService:
    def __init__(self, retrieval_service: RetrievalService, chain: AnswerChain) -> None:
        self._retrieval_service = retrieval_service
        self._chain = chain

    def answer(self, question: str, limit: int) -> Answer:
        """Answer `question` from the `limit` most similar chunks.

        Returns an `Answer`: `ANSWERED` with validated citations, or
        `INSUFFICIENT_CONTEXT` with a reason. Raises `InvalidModelOutputError`
        for an output that fails validation and `LLMError` for a provider
        failure; see the module docstring for the full error contract.
        """
        chunks = self._retrieval_service.retrieve(question, limit)
        if not chunks:
            return _insufficient(InsufficientReason.NO_RETRIEVAL)

        context = assemble_context(chunks)
        if context.is_empty:
            return _insufficient(InsufficientReason.EMPTY_CONTEXT)

        draft = self._invoke(question, context)
        return validate_answer_draft(draft, context)

    def _invoke(self, question: str, context: AssembledContext) -> Any:
        inputs = build_prompt_inputs(question, context)
        try:
            return self._chain.invoke(inputs)
        except Exception as exc:
            translated = _translate(exc)
            if translated is None:
                raise
        raise translated


def _translate(exc: Exception) -> Exception | None:
    """The application error for a chain failure, or `None` to re-raise it as is."""
    if isinstance(exc, OutputParserException | ValidationError):
        return InvalidModelOutputError(InvalidOutputCode.MALFORMED_OUTPUT)
    return map_llm_error(exc)


def _insufficient(reason: InsufficientReason) -> Answer:
    return Answer(
        status=AnswerStatus.INSUFFICIENT_CONTEXT,
        answer=None,
        citations=(),
        reason=reason,
    )
