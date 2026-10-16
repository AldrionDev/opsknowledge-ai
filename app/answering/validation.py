"""Deterministic grounding validation of a model's `AnswerDraft`.

`validate_answer_draft` is a pure function: a draft and the `AssembledContext` it
was produced from in, an `Answer` out, or an `InvalidModelOutputError`. It calls
no model and performs no I/O.

Validation levels
-----------------

1. **Structural validity**: the cited `source_id` exists in the context.
2. **Evidence presence**: the `evidence_quote` is a contiguous substring of the
   cited source's content after conservative normalization.
3. **Semantic support: NOT enforced.** Whether a claim actually follows from its
   quote is not checked at runtime. A quote that is present in the source proves
   only that the model did not invent the quote; it does not prove entailment.
   The claim may overstate, negate or misread the quoted passage. Semantic
   grounding is measured by the evaluation harness (#40), not by this module.

Normalization, applied identically to the quote and to the source content:
Unicode NFC, then every run of whitespace (including the ends) collapsed to a
single space and trimmed. Matching is case-sensitive and exact, within the
content of one single cited source. There is no case folding, stemming,
punctuation or markdown stripping, and no fuzzy matching.

`MIN_EVIDENCE_QUOTE_LENGTH` excludes trivial matches: single words, identifiers
and error codes occur in almost any source and prove nothing. 20 sits above such
tokens (`OOMKilled`, `502`, `kubectl get pods`) and below a full clause, which
is what the model is asked to quote. The length is measured with `len()` on the
normalized, trimmed quote, in Unicode code points, the same unit as
`app.answering.context`; padding a quote with whitespace cannot reach it. It is
a heuristic that the evaluation (#40) may tune.

Claim text is shown to the client as one line of the answer, followed by the
claim's own `[source_id]` marker, so each claim yields exactly one answer line
and one citation. Claim text that contains a line break or a `[digits]` marker
would add lines or markers without a matching citation; it is rejected, never
rewritten.

Fail closed: the first invalid element rejects the whole draft; there is no
partial acceptance. Checks run in a fixed order, so the code is deterministic:

1. draft consistency (`insufficient` against the number of claims);
2. claims in order, each checked for blank text, line breaks and citation
   markers in the text (all `INCONSISTENT_RESPONSE`), then a known `source_id`,
   then the evidence quote.

The final answer and citations are built from validated claims and from the
trusted context only.
"""

import re
import unicodedata

from app.answering.context import AssembledContext, ContextSource
from app.answering.errors import InvalidModelOutputError
from app.answering.models import (
    Answer,
    AnswerDraft,
    AnswerStatus,
    Citation,
    Claim,
    InsufficientReason,
    InvalidOutputCode,
)

MIN_EVIDENCE_QUOTE_LENGTH = 20

# Characters `str.splitlines()` treats as line boundaries: CR, LF, VT, FF, the
# C0 separators, NEL and the Unicode line and paragraph separators.
_LINE_BREAK = re.compile("[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
# The canonical citation marker, as the answer text renders it.
_CITATION_MARKER = re.compile(r"\[[0-9]+\]")


def validate_answer_draft(draft: AnswerDraft, context: AssembledContext) -> Answer:
    """Validate `draft` against `context` and build the final `Answer`.

    Returns an `INSUFFICIENT_CONTEXT` answer with reason `MODEL_DECLINED` when
    the model declined without claims. Raises `InvalidModelOutputError` for any
    invalid output; see the module docstring for the levels and the order of
    checks. Evidence presence does not establish semantic support.

    A `draft` that is not an `AnswerDraft` is rejected as `MALFORMED_OUTPUT`;
    turning provider parsing failures into that error is the caller's concern.
    """
    if not isinstance(draft, AnswerDraft):
        raise InvalidModelOutputError(InvalidOutputCode.MALFORMED_OUTPUT)

    if draft.insufficient:
        if draft.claims:
            raise InvalidModelOutputError(InvalidOutputCode.INCONSISTENT_RESPONSE)
        return Answer(
            status=AnswerStatus.INSUFFICIENT_CONTEXT,
            answer=None,
            citations=(),
            reason=InsufficientReason.MODEL_DECLINED,
        )
    if not draft.claims:
        raise InvalidModelOutputError(InvalidOutputCode.INCONSISTENT_RESPONSE)

    lines: list[str] = []
    citations: list[Citation] = []
    for claim in draft.claims:
        source, quote = _validate_claim(claim, context)
        lines.append(f"{claim.text.strip()} [{source.source_id}]")
        citations.append(_to_citation(source, quote))

    return Answer(
        status=AnswerStatus.ANSWERED,
        answer="\n".join(lines),
        citations=tuple(citations),
        reason=None,
    )


def _validate_claim(
    claim: Claim, context: AssembledContext
) -> tuple[ContextSource, str]:
    """Return the cited source and the normalized quote of a valid claim."""
    if (
        not claim.text.strip()
        or _LINE_BREAK.search(claim.text)
        or _CITATION_MARKER.search(claim.text)
    ):
        raise InvalidModelOutputError(InvalidOutputCode.INCONSISTENT_RESPONSE)

    source = context.source_by_id(claim.source_id)
    if source is None:
        raise InvalidModelOutputError(InvalidOutputCode.UNKNOWN_CITATION)

    quote = _normalize(claim.evidence_quote)
    if len(quote) < MIN_EVIDENCE_QUOTE_LENGTH or quote not in _normalize(
        source.content
    ):
        raise InvalidModelOutputError(InvalidOutputCode.MISSING_EVIDENCE)
    return source, quote


def _normalize(text: str) -> str:
    # `str.split()` without arguments splits on any Unicode whitespace and drops
    # empty parts, which also trims the ends.
    return " ".join(unicodedata.normalize("NFC", text).split())


def _to_citation(source: ContextSource, quote: str) -> Citation:
    return Citation(
        source_id=source.source_id,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        source_path=source.source_path,
        document_type=source.document_type,
        chunk_index=source.chunk_index,
        evidence_quote=quote,
    )
