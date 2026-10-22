"""HTTP boundary of grounded answering.

`POST /ask` validates the request, delegates to `AnswerService`, and serializes the
validated `Answer`. It contains no retrieval, prompt, model or validation logic.

Request validation mirrors `POST /search`: the question must not be empty or
whitespace-only, and `limit` uses the same default and bounds, imported so they
cannot drift apart. The question is passed to the service unchanged. Its length
is bounded by `MAX_QUESTION_LENGTH`; invalid input is rejected by Pydantic with
HTTP 422 before the service is called.

Outcomes:

- `answered` and `insufficient_context` are both HTTP 200: missing knowledge is a
  normal result, never an error.
- An invalid model output is HTTP 502 with its `InvalidOutputCode`, a provider
  failure HTTP 502 `llm_unavailable`, a missing or invalid LLM configuration
  HTTP 503 `llm_not_configured`. Their bodies are `{"detail", "code"}` with fixed
  messages; `app.main` registers the handlers.
- Anything else falls through to the generic HTTP 500 handler of `app.main`.

Nothing is logged here: neither the question nor the answer.
"""

from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.answering.models import (
    Answer,
    AnswerStatus,
    Citation,
    InsufficientReason,
    InvalidOutputCode,
)
from app.answering.service import AnswerService
from app.api.dependencies import get_answer_service
from app.api.search import DEFAULT_RESULT_LIMIT
from app.ingestion.models import DocumentType
from app.retrieval.service import MAX_RESULT_LIMIT

# The question is embedded by a model that reads at most 512 tokens; about 2000
# English characters already exceed that, so a longer question could not
# influence retrieval. The bound also keeps the prompt small next to the 8000
# character context budget. Measured in characters (code points), like the
# chunker and the context assembly.
MAX_QUESTION_LENGTH: Final = 2000

LLM_UNAVAILABLE: Final = "llm_unavailable"
LLM_NOT_CONFIGURED: Final = "llm_not_configured"
LLM_UNAVAILABLE_MESSAGE: Final = "Language model provider is unavailable"
LLM_NOT_CONFIGURED_MESSAGE: Final = "Language model is not configured"


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description=(
            f"Natural-language question, at most {MAX_QUESTION_LENGTH} characters."
        ),
        examples=["Why is the payments pod repeatedly restarting?"],
    )
    limit: int = Field(
        default=DEFAULT_RESULT_LIMIT,
        ge=1,
        le=MAX_RESULT_LIMIT,
        description=(
            f"Maximum number of retrieved chunks, between 1 and {MAX_RESULT_LIMIT}."
        ),
    )

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        # The question itself is passed on unchanged: trimming it here would
        # change what gets embedded and what the model reads.
        if not value.strip():
            raise ValueError("question is empty or whitespace-only")
        return value


class AskCitation(BaseModel):
    """The source backing one line of the answer, in answer-line order."""

    source_id: int
    chunk_id: str
    document_id: str
    source_path: str
    document_type: DocumentType
    chunk_index: int
    evidence_quote: str


class AskResponse(BaseModel):
    question: str
    status: AnswerStatus
    reason: InsufficientReason | None = Field(
        description="Why there is no answer; null when the question was answered."
    )
    answer: str | None = Field(
        description="One line per validated claim, ending in its [source_id] marker."
    )
    citations: list[AskCitation]


class UpstreamErrorResponse(BaseModel):
    """HTTP 502: the model output was rejected or the provider call failed."""

    detail: str
    code: InvalidOutputCode | Literal["llm_unavailable"]


class ServiceUnavailableResponse(BaseModel):
    """HTTP 503: the language model is not configured."""

    detail: str
    code: Literal["llm_not_configured"]


class InternalErrorResponse(BaseModel):
    """HTTP 500: the generic error body of the application."""

    detail: str


router = APIRouter(tags=["answering"])


@router.post(
    "/ask",
    summary="Grounded answer from the knowledge corpus",
    responses={
        502: {
            "model": UpstreamErrorResponse,
            "description": "Invalid model output or language model provider failure",
        },
        503: {
            "model": ServiceUnavailableResponse,
            "description": "Language model is not configured",
        },
        500: {
            "model": InternalErrorResponse,
            "description": "Unexpected internal error",
        },
    },
)
def ask(
    request: AskRequest,
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
) -> AskResponse:
    """Answer the question from the retrieved sources, or say why it cannot."""
    answer = answer_service.answer(request.question, request.limit)
    return _to_ask_response(request.question, answer)


def _to_ask_response(question: str, answer: Answer) -> AskResponse:
    return AskResponse(
        question=question,
        status=answer.status,
        reason=answer.reason,
        answer=answer.answer,
        citations=[_to_ask_citation(citation) for citation in answer.citations],
    )


def _to_ask_citation(citation: Citation) -> AskCitation:
    return AskCitation(
        source_id=citation.source_id,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        source_path=citation.source_path,
        document_type=citation.document_type,
        chunk_index=citation.chunk_index,
        evidence_quote=citation.evidence_quote,
    )
