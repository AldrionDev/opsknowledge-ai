import ast
import unicodedata
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.answering.context import AssembledContext, assemble_context
from app.answering.errors import InvalidModelOutputError
from app.answering.models import (
    Answer,
    AnswerDraft,
    AnswerStatus,
    Claim,
    InsufficientReason,
    InvalidOutputCode,
)
from app.answering.validation import MIN_EVIDENCE_QUOTE_LENGTH, validate_answer_draft
from app.ingestion.models import DocumentType
from app.retrieval.models import RetrievedChunk

ANSWERING_DIR = Path(__file__).resolve().parent.parent / "app" / "answering"
NEW_MODULES = ("models.py", "errors.py", "validation.py")

FORBIDDEN_IMPORT_ROOTS = {
    "fastapi",
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "anthropic",
    "sqlalchemy",
    "pgvector",
    "psycopg",
    "alembic",
}
FORBIDDEN_IMPORT_MODULES = {"app.db", "app.llm", "app.api"}

PODS = "The pod restarts because the container exceeds its memory limit."
DISK = "Free disk space by removing unused container images from the node."

SENTINEL_CLAIM = "SENTINEL-CLAIM-TEXT-91f3"
SENTINEL_QUOTE = "SENTINEL-QUOTE-TEXT-47ab-with-enough-length"
SENTINEL_SOURCE = "SENTINEL-SOURCE-CONTENT-c2d8"


def make_chunk(
    chunk_id: str, content: str, *, score: float = 0.9, chunk_index: int = 0
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=chunk_index,
        source_path=f"runbooks/{chunk_id}.md",
        document_type=DocumentType.RUNBOOK,
        content=content,
        similarity_score=score,
    )


def make_context(*contents: str) -> AssembledContext:
    # Descending scores keep the given order: source n holds contents[n - 1].
    chunks = [
        make_chunk(f"chunk-{i}", content, score=1.0 - i / 10, chunk_index=i)
        for i, content in enumerate(contents, start=1)
    ]
    return assemble_context(chunks, 100_000)


def make_claim(
    source_id: int = 1, quote: str = PODS, text: str = "Memory limit restarts pods."
) -> Claim:
    return Claim(text=text, source_id=source_id, evidence_quote=quote)


def make_draft(*claims: Claim, insufficient: bool = False) -> AnswerDraft:
    return AnswerDraft(insufficient=insufficient, claims=list(claims))


def rejected_code(draft: AnswerDraft, context: AssembledContext) -> InvalidOutputCode:
    with pytest.raises(InvalidModelOutputError) as excinfo:
        validate_answer_draft(draft, context)
    return excinfo.value.code


# --- schema ---------------------------------------------------------------


def test_a_well_formed_json_payload_parses_into_a_draft() -> None:
    draft = AnswerDraft.model_validate_json(
        '{"insufficient": false, "claims": '
        '[{"text": "t", "source_id": 2, "evidence_quote": "q"}]}'
    )

    assert draft.claims == [Claim(text="t", source_id=2, evidence_quote="q")]


@pytest.mark.parametrize(
    "payload",
    [
        {"insufficient": False},
        {"insufficient": False, "claims": None},
        {"insufficient": False, "claims": [], "answer": "free text"},
        {"insufficient": "false", "claims": []},
        {
            "insufficient": False,
            "claims": [{"text": "t", "source_id": "1", "evidence_quote": "q"}],
        },
        {
            "insufficient": False,
            "claims": [{"text": "t", "source_id": True, "evidence_quote": "q"}],
        },
        {
            "insufficient": False,
            "claims": [{"text": "t", "source_id": 1.5, "evidence_quote": "q"}],
        },
        {"insufficient": False, "claims": [{"text": "t", "source_id": 1}]},
        {
            "insufficient": False,
            "claims": [
                {"text": "t", "source_id": 1, "evidence_quote": "q", "chunk_id": "x"}
            ],
        },
    ],
)
def test_a_payload_that_breaks_the_schema_is_not_coerced(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate(payload)


def test_the_draft_schema_has_no_free_form_answer_field() -> None:
    assert set(AnswerDraft.model_fields) == {"insufficient", "claims"}
    assert set(Claim.model_fields) == {"text", "source_id", "evidence_quote"}


def test_the_json_schema_forbids_additional_properties_on_both_models() -> None:
    schema = AnswerDraft.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Claim"]["additionalProperties"] is False


# --- valid drafts ----------------------------------------------------------


def test_a_valid_draft_builds_the_answer_text_with_source_markers() -> None:
    context = make_context(PODS, DISK)
    draft = make_draft(
        make_claim(1, PODS, "Pods restart on memory pressure."),
        make_claim(2, DISK, "  Remove unused images.  "),
    )

    answer = validate_answer_draft(draft, context)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.reason is None
    assert (
        answer.answer
        == "Pods restart on memory pressure. [1]\nRemove unused images. [2]"
    )


def test_citation_metadata_comes_from_the_assembled_context() -> None:
    context = make_context(PODS, DISK)

    answer = validate_answer_draft(make_draft(make_claim(2, DISK)), context)

    source = context.sources[1]
    (citation,) = answer.citations
    assert citation.source_id == 2
    assert citation.chunk_id == source.chunk_id
    assert citation.document_id == source.document_id
    assert citation.source_path == source.source_path
    assert citation.document_type is source.document_type
    assert citation.chunk_index == source.chunk_index
    assert citation.evidence_quote == DISK


def test_citations_follow_claim_order_one_per_claim() -> None:
    context = make_context(PODS, DISK)
    draft = make_draft(make_claim(2, DISK), make_claim(1, PODS), make_claim(2, DISK))

    answer = validate_answer_draft(draft, context)

    assert [c.source_id for c in answer.citations] == [2, 1, 2]


def test_whitespace_differences_are_normalized_in_the_returned_citation_quote() -> None:
    context = make_context(
        "Restart   the\n\tservice after the  config change is applied."
    )
    quote = "  the service   after the\nconfig change  "

    answer = validate_answer_draft(make_draft(make_claim(1, quote)), context)

    # The citation carries the normalized quote, not the raw source slice (which
    # has runs of whitespace) and not the raw model quote (padding, newline).
    assert answer.citations[0].evidence_quote == "the service after the config change"


def test_answer_and_citations_are_immutable() -> None:
    answer = validate_answer_draft(make_draft(make_claim()), make_context(PODS))

    with pytest.raises(FrozenInstanceError):
        answer.status = AnswerStatus.INSUFFICIENT_CONTEXT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        answer.citations[0].chunk_id = "other"  # type: ignore[misc]
    assert isinstance(answer.citations, tuple)


# --- genuine insufficient context -----------------------------------------


def test_a_declined_draft_is_insufficient_context_not_an_error() -> None:
    answer = validate_answer_draft(make_draft(insufficient=True), make_context(PODS))

    assert answer == Answer(
        status=AnswerStatus.INSUFFICIENT_CONTEXT,
        answer=None,
        citations=(),
        reason=InsufficientReason.MODEL_DECLINED,
    )


# --- normalization --------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "The pod restarts because the container exceeds its memory limit.",
        "The  pod restarts\tbecause the\ncontainer exceeds   its memory limit.",
        "The pod restarts because the container exceeds its memory limit.",
        "   The pod restarts because the container exceeds its memory limit.   ",
    ],
)
def test_quotes_differing_only_by_whitespace_are_accepted(quote: str) -> None:
    answer = validate_answer_draft(make_draft(make_claim(1, quote)), make_context(PODS))

    assert answer.status is AnswerStatus.ANSWERED


def test_unicode_normalization_form_does_not_matter() -> None:
    composed = "Restart the café service after the configuration change."
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed

    for source, quote in ((composed, decomposed), (decomposed, composed)):
        answer = validate_answer_draft(
            make_draft(make_claim(1, quote)), make_context(source)
        )
        assert answer.status is AnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "quote",
    [
        PODS.lower(),
        PODS.upper(),
        "the pod restarts because the container exceeds its memory limit.",
    ],
)
def test_a_quote_differing_by_case_is_rejected(quote: str) -> None:
    code = rejected_code(make_draft(make_claim(1, quote)), make_context(PODS))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_a_quote_differing_by_markdown_characters_is_rejected() -> None:
    context = make_context(
        "Run `kubectl describe pod` and read the **Events** section."
    )

    for quote in (
        "Run kubectl describe pod and read the Events section.",
        "Run `kubectl describe pod` and read the Events section.",
    ):
        assert (
            rejected_code(make_draft(make_claim(1, quote)), context)
            is InvalidOutputCode.MISSING_EVIDENCE
        )


def test_punctuation_is_not_stripped() -> None:
    quote = PODS.rstrip(".") + "!"

    code = rejected_code(make_draft(make_claim(1, quote)), make_context(PODS))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


# --- minimum quote length --------------------------------------------------


def test_the_minimum_quote_length_is_twenty_characters() -> None:
    assert MIN_EVIDENCE_QUOTE_LENGTH == 20


def test_a_quote_of_exactly_the_minimum_length_is_accepted() -> None:
    source = "x" * 5 + "abcdefghij klmnopqrs" + "y" * 5  # quote below has 20 chars
    quote = "abcdefghij klmnopqrs"
    assert len(quote) == MIN_EVIDENCE_QUOTE_LENGTH

    answer = validate_answer_draft(
        make_draft(make_claim(1, quote)), make_context(source)
    )

    assert answer.status is AnswerStatus.ANSWERED


def test_a_quote_one_character_below_the_minimum_is_rejected() -> None:
    quote = "abcdefghij klmnopqr"  # 19 characters, present in the source
    source = f"intro {quote} outro"
    assert len(quote) == MIN_EVIDENCE_QUOTE_LENGTH - 1

    code = rejected_code(make_draft(make_claim(1, quote)), make_context(source))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_whitespace_padding_does_not_reach_the_minimum_length() -> None:
    quote = "  short phrase       \n\t  "
    context = make_context("A short phrase appears here in the source text.")

    code = rejected_code(make_draft(make_claim(1, quote)), context)

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_the_length_is_measured_after_unicode_normalization() -> None:
    # 19 composed characters, but 21 code points once decomposed.
    composed = "éééééééééééééééééé" + "x"
    decomposed = unicodedata.normalize("NFD", composed)
    assert len(composed) == 19 and len(decomposed) > MIN_EVIDENCE_QUOTE_LENGTH

    code = rejected_code(
        make_draft(make_claim(1, decomposed)), make_context(f"see {composed} here")
    )

    assert code is InvalidOutputCode.MISSING_EVIDENCE


@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_a_blank_quote_is_rejected(quote: str) -> None:
    code = rejected_code(make_draft(make_claim(1, quote)), make_context(PODS))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


# --- unknown_citation ------------------------------------------------------


@pytest.mark.parametrize("source_id", [0, -1, 3, 10_000])
def test_an_unknown_source_id_is_rejected(source_id: int) -> None:
    code = rejected_code(
        make_draft(make_claim(source_id, PODS)), make_context(PODS, DISK)
    )

    assert code is InvalidOutputCode.UNKNOWN_CITATION


def test_any_citation_against_an_empty_context_is_unknown() -> None:
    code = rejected_code(make_draft(make_claim(1, PODS)), make_context())

    assert code is InvalidOutputCode.UNKNOWN_CITATION


# --- missing_evidence ------------------------------------------------------


def test_a_fabricated_quote_is_rejected() -> None:
    quote = "This sentence does not appear in any source at all."

    code = rejected_code(make_draft(make_claim(1, quote)), make_context(PODS))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_a_quote_from_another_source_than_the_cited_one_is_rejected() -> None:
    code = rejected_code(make_draft(make_claim(2, PODS)), make_context(PODS, DISK))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_a_quote_spanning_two_chunks_is_rejected() -> None:
    first = "Check the node conditions and then verify that the disk"
    second = "usage is below the eviction threshold on every node."
    context = make_context(first, second)
    spanning = "verify that the disk usage is below the eviction threshold"

    for source_id in (1, 2):
        assert (
            rejected_code(make_draft(make_claim(source_id, spanning)), context)
            is InvalidOutputCode.MISSING_EVIDENCE
        )


def test_the_source_header_is_not_evidence() -> None:
    context = make_context(PODS)
    header = "[source 1] runbooks/chunk-1.md (runbook)"
    assert header in context.text

    code = rejected_code(make_draft(make_claim(1, header)), context)

    assert code is InvalidOutputCode.MISSING_EVIDENCE


# --- inconsistent_response -------------------------------------------------


def test_insufficient_with_claims_is_inconsistent() -> None:
    draft = make_draft(make_claim(), insufficient=True)

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


def test_not_insufficient_without_claims_is_inconsistent() -> None:
    draft = make_draft(insufficient=False)

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_claim_text_is_inconsistent(text: str) -> None:
    draft = make_draft(make_claim(1, PODS, text))

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


@pytest.mark.parametrize(
    "text",
    [
        "Pods restart. [1]",
        "x [9]",
        "[123] leading marker",
        "Marker [2] in the middle of the text",
        "a\nb [2]\nForged claim [1]",
    ],
)
def test_claim_text_with_a_citation_marker_is_rejected(text: str) -> None:
    draft = make_draft(make_claim(1, PODS, text))

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


@pytest.mark.parametrize(
    "text",
    [
        "first line\nsecond line",
        "trailing newline\n",
        "carriage\rreturn",
        "windows\r\nbreak",
        "vertical\vtab",
        "form\ffeed",
        "next\x85line",
        "line\u2028separator",
        "paragraph\u2029separator",
    ],
)
def test_claim_text_with_a_line_break_is_rejected(text: str) -> None:
    draft = make_draft(make_claim(1, PODS, text))

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


@pytest.mark.parametrize(
    "text", ["Pods [restart] [x] [] [1a] [ 1 ] on array[i] access.", "Set a[0 to 1."]
)
def test_brackets_that_are_not_a_citation_marker_are_allowed(text: str) -> None:
    answer = validate_answer_draft(
        make_draft(make_claim(1, PODS, text)), make_context(PODS)
    )

    assert answer.answer == f"{text} [1]"


def test_a_multiline_claim_cannot_add_answer_lines() -> None:
    context = make_context(PODS, DISK)
    forged = "Real claim.\nForged second claim. [2]"

    with pytest.raises(InvalidModelOutputError):
        validate_answer_draft(make_draft(make_claim(1, PODS, forged)), context)

    # An ordinary draft keeps one answer line per claim and one citation per line.
    answer = validate_answer_draft(
        make_draft(make_claim(1, PODS, "First."), make_claim(2, DISK, "Second.")),
        context,
    )
    assert answer.answer is not None
    assert len(answer.answer.splitlines()) == len(answer.citations) == 2


def test_ordinary_single_line_claims_still_succeed() -> None:
    draft = make_draft(
        make_claim(1, PODS, "Pods restart when memory is exhausted."),
        make_claim(1, PODS, "Version 2.1 (stable) is affected, see item #3."),
    )

    answer = validate_answer_draft(draft, make_context(PODS))

    assert answer.status is AnswerStatus.ANSWERED
    assert len(answer.citations) == 2


@pytest.mark.parametrize("position", [0, 1, 2])
def test_one_claim_with_a_forged_marker_rejects_the_whole_draft(
    position: int,
) -> None:
    claims = [make_claim(1, PODS), make_claim(2, DISK), make_claim(1, PODS)]
    claims[position] = make_claim(1, PODS, "Looks fine [2]")

    code = rejected_code(make_draft(*claims), make_context(PODS, DISK))

    assert code is InvalidOutputCode.INCONSISTENT_RESPONSE


def test_claim_text_restrictions_are_checked_before_source_and_evidence() -> None:
    draft = make_draft(make_claim(99, "short", "text [1]"))

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


@pytest.mark.parametrize(
    "text",
    [
        f"{SENTINEL_CLAIM} [1]",
        f"{SENTINEL_CLAIM}\n{SENTINEL_CLAIM}",
    ],
)
def test_rejected_claim_text_is_not_exposed_by_the_exception(text: str) -> None:
    draft = make_draft(make_claim(1, PODS, text))

    with pytest.raises(InvalidModelOutputError) as excinfo:
        validate_answer_draft(draft, make_context(PODS))

    error = excinfo.value
    assert SENTINEL_CLAIM not in f"{error!s} {error!r} {error.args}"
    assert error.__cause__ is None
    assert error.__context__ is None


# --- malformed_output ------------------------------------------------------


@pytest.mark.parametrize("value", [None, {"insufficient": False, "claims": []}, "text"])
def test_an_input_that_is_not_a_draft_is_malformed_output(value: object) -> None:
    with pytest.raises(InvalidModelOutputError) as excinfo:
        validate_answer_draft(value, make_context(PODS))  # type: ignore[arg-type]

    assert excinfo.value.code is InvalidOutputCode.MALFORMED_OUTPUT


# --- fail closed and precedence -------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 2])
def test_one_invalid_claim_rejects_the_whole_draft(position: int) -> None:
    claims = [make_claim(1, PODS), make_claim(2, DISK), make_claim(1, PODS)]
    claims[position] = make_claim(
        1, "A quote that the source never contained anywhere."
    )

    code = rejected_code(make_draft(*claims), make_context(PODS, DISK))

    assert code is InvalidOutputCode.MISSING_EVIDENCE


def test_draft_consistency_is_checked_before_the_claims() -> None:
    draft = make_draft(make_claim(99, "nonsense"), insufficient=True)

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


def test_blank_text_is_reported_before_an_unknown_source() -> None:
    draft = make_draft(make_claim(99, PODS, "  "))

    assert (
        rejected_code(draft, make_context(PODS))
        is InvalidOutputCode.INCONSISTENT_RESPONSE
    )


def test_an_unknown_source_is_reported_before_a_bad_quote() -> None:
    draft = make_draft(make_claim(99, "short"))

    assert (
        rejected_code(draft, make_context(PODS)) is InvalidOutputCode.UNKNOWN_CITATION
    )


def test_the_first_invalid_claim_determines_the_code() -> None:
    draft = make_draft(
        make_claim(1, "A quote that the source never contained anywhere."),
        make_claim(99, PODS),
    )

    assert (
        rejected_code(draft, make_context(PODS)) is InvalidOutputCode.MISSING_EVIDENCE
    )


# --- safe exceptions -------------------------------------------------------


def _unsafe_drafts() -> list[tuple[AnswerDraft, AssembledContext]]:
    context = make_context(f"{SENTINEL_SOURCE} is the only content of this chunk.")
    return [
        (  # missing_evidence
            make_draft(make_claim(1, SENTINEL_QUOTE, SENTINEL_CLAIM)),
            context,
        ),
        (  # unknown_citation
            make_draft(make_claim(9, SENTINEL_QUOTE, SENTINEL_CLAIM)),
            context,
        ),
        (  # inconsistent_response
            make_draft(
                make_claim(1, SENTINEL_QUOTE, SENTINEL_CLAIM), insufficient=True
            ),
            context,
        ),
    ]


@pytest.mark.parametrize("draft,context", _unsafe_drafts())
def test_exceptions_expose_no_model_text_or_source_content(
    draft: AnswerDraft, context: AssembledContext
) -> None:
    with pytest.raises(InvalidModelOutputError) as excinfo:
        validate_answer_draft(draft, context)

    error = excinfo.value
    exposed = f"{error!s} {error!r} {error.args}"
    for secret in (SENTINEL_CLAIM, SENTINEL_QUOTE, SENTINEL_SOURCE):
        assert secret not in exposed
    assert error.__cause__ is None
    assert error.__context__ is None


def test_every_code_has_a_distinct_fixed_message() -> None:
    messages = {code: str(InvalidModelOutputError(code)) for code in InvalidOutputCode}

    assert all(messages.values())
    assert len(set(messages.values())) == len(InvalidOutputCode)


# --- taxonomy and boundaries ----------------------------------------------


def test_the_taxonomy_matches_the_issue() -> None:
    assert {c.value for c in InvalidOutputCode} == {
        "malformed_output",
        "unknown_citation",
        "missing_evidence",
        "inconsistent_response",
    }
    assert {r.value for r in InsufficientReason} == {
        "no_retrieval",
        "empty_context",
        "model_declined",
    }
    assert {s.value for s in AnswerStatus} == {"answered", "insufficient_context"}


def test_the_validation_modules_depend_on_no_framework_llm_or_database_code() -> None:
    for name in NEW_MODULES:
        tree = ast.parse((ANSWERING_DIR / name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        roots = {module.split(".")[0] for module in imported}
        assert not roots & FORBIDDEN_IMPORT_ROOTS, name
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in imported
            for forbidden in FORBIDDEN_IMPORT_MODULES
        ), name
