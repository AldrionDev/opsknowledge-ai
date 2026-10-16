"""Data contracts of grounded answering: the model's draft and the final answer.

`AnswerDraft` is what the language model is asked to produce. It is deliberately
flat and free of value constraints, so it maps onto a provider-native JSON
schema; every rule about its content is enforced by `validate_answer_draft`, not
by the schema. It has no free-form answer field: the answer text is built from
validated claims only.

`strict=True` rejects silent coercion (`"1"` or `True` as a `source_id`), and
`extra="forbid"` rejects unknown fields. A draft that does not parse is the
caller's concern (issue #37): a Pydantic `ValidationError` may echo the input
values, so it is never safe to expose to a client as is.

`Answer` and `Citation` are the validated outcome. Everything in them comes
from validated claims or from the trusted `AssembledContext`, never from
model-provided metadata.
"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.ingestion.models import DocumentType


class Claim(BaseModel):
    """One statement of the answer with the evidence the model cites for it.

    `text` must be a single line without citation markers: the answer shows one
    line per claim, ending in the claim's own `[source_id]` marker. Validation
    rejects text with a line break or a `[digits]` marker.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    source_id: int
    evidence_quote: str


class AnswerDraft(BaseModel):
    """The structured output requested from the model, before any validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    insufficient: bool
    claims: list[Claim]


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class InsufficientReason(StrEnum):
    """Why there is no answer although nothing went wrong.

    Genuine lack of knowledge, never an error. The validator only produces
    `MODEL_DECLINED`; the other two are produced by the answer service (#37).
    """

    NO_RETRIEVAL = "no_retrieval"
    EMPTY_CONTEXT = "empty_context"
    MODEL_DECLINED = "model_declined"


class InvalidOutputCode(StrEnum):
    """Machine-readable reason a model output was rejected. Always an error."""

    MALFORMED_OUTPUT = "malformed_output"
    UNKNOWN_CITATION = "unknown_citation"
    MISSING_EVIDENCE = "missing_evidence"
    INCONSISTENT_RESPONSE = "inconsistent_response"


@dataclass(frozen=True)
class Citation:
    """The source backing one claim.

    `source_id`, `chunk_id`, `document_id`, `source_path`, `document_type` and
    `chunk_index` come exclusively from the trusted `AssembledContext`.

    `evidence_quote` is model-supplied data that passed normalization (Unicode
    NFC, consecutive whitespace collapsed to one space, ends trimmed) and the
    substring check. It is the *normalized* quote, so it is not necessarily an
    exact raw substring of the source content: whitespace may differ.
    """

    source_id: int
    chunk_id: str
    document_id: str
    source_path: str
    document_type: DocumentType
    chunk_index: int
    evidence_quote: str


@dataclass(frozen=True)
class Answer:
    """A validated outcome.

    `ANSWERED`: `answer` is the validated claims joined with their `[source_id]`
    markers, one line per claim, `citations` holds one entry per claim in claim
    order (so line n of the answer belongs to citation n), `reason` is
    `None`. `INSUFFICIENT_CONTEXT`: `answer` is `None`, `citations` is empty and
    `reason` says why.
    """

    status: AnswerStatus
    answer: str | None
    citations: tuple[Citation, ...]
    reason: InsufficientReason | None
