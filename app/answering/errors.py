"""Errors of answer validation.

`InvalidModelOutputError` means the model produced output that must not be shown
to a client. It is an error, never a statement that the knowledge base lacks an
answer: that is the `insufficient_context` outcome.

Messages are fixed per code. They contain no model-generated text, no source
content and no validation details, so the exception is safe to log and to map to
an HTTP error body.
"""

from app.answering.models import InvalidOutputCode

_MESSAGES: dict[InvalidOutputCode, str] = {
    InvalidOutputCode.MALFORMED_OUTPUT: "Model output is not a valid answer draft",
    InvalidOutputCode.UNKNOWN_CITATION: "Model output cites a source that does not exist",
    InvalidOutputCode.MISSING_EVIDENCE: "Model output contains evidence not found in the cited source",
    InvalidOutputCode.INCONSISTENT_RESPONSE: "Model output is internally inconsistent",
}


class InvalidModelOutputError(Exception):
    """Raised when a model output fails validation; `code` says why."""

    def __init__(self, code: InvalidOutputCode) -> None:
        super().__init__(_MESSAGES[code])
        self.code = code
