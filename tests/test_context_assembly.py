import ast
from pathlib import Path

import pytest

from app.answering.context import (
    DEFAULT_CONTEXT_CHARACTER_BUDGET,
    EMPTY_CONTEXT,
    SOURCE_BLOCK_SEPARATOR,
    AssembledContext,
    ContextSource,
    assemble_context,
)
from app.ingestion.models import DocumentType
from app.retrieval.models import RetrievedChunk

CONTEXT_MODULE = (
    Path(__file__).resolve().parent.parent / "app" / "answering" / "context.py"
)

# Large enough that no fixture in this module is affected by the budget.
UNBOUNDED = 100_000

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


def make_chunk(
    chunk_id: str = "chunk-a",
    *,
    score: float = 0.9,
    document_id: str = "doc-a",
    chunk_index: int = 0,
    source_path: str = "runbooks/example.md",
    document_type: DocumentType = DocumentType.RUNBOOK,
    content: str = "Check the pod logs.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=source_path,
        document_type=document_type,
        content=content,
        similarity_score=score,
    )


def chunk_ids(context: AssembledContext) -> list[str]:
    return [source.chunk_id for source in context.sources]


def rendered_length(chunk: RetrievedChunk) -> int:
    """Length of `chunk` rendered as the only source, measured by the module itself."""
    return len(assemble_context([chunk], UNBOUNDED).text)


def test_sources_are_ranked_and_numbered_from_one() -> None:
    # The input order is deliberately not the ranked order.
    chunks = [
        make_chunk("b", score=0.5),
        make_chunk("a", score=0.9),
        make_chunk("c", score=0.1),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    assert chunk_ids(context) == ["a", "b", "c"]
    assert [source.source_id for source in context.sources] == [1, 2, 3]


def test_equal_scores_are_ordered_by_chunk_id() -> None:
    chunks = [
        make_chunk("c", score=0.4),
        make_chunk("a", score=0.4),
        make_chunk("b", score=0.4),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    assert chunk_ids(context) == ["a", "b", "c"]


def test_input_order_does_not_change_the_result() -> None:
    chunks = [
        make_chunk("a", score=0.9),
        make_chunk("b", score=0.4),
        make_chunk("c", score=0.4),
    ]

    assert assemble_context(list(reversed(chunks)), UNBOUNDED) == assemble_context(
        chunks, UNBOUNDED
    )


def test_repeated_assembly_of_the_same_input_is_identical() -> None:
    chunks = [make_chunk("a", score=0.9), make_chunk("b", score=0.5)]

    first = assemble_context(chunks, UNBOUNDED)
    second = assemble_context(chunks, UNBOUNDED)

    assert first == second
    assert first.text == second.text


def test_input_sequence_is_not_mutated() -> None:
    chunks = [make_chunk("b", score=0.2), make_chunk("a", score=0.8)]
    snapshot = list(chunks)

    assemble_context(chunks, UNBOUNDED)

    assert chunks == snapshot


def test_duplicate_chunk_id_is_included_once() -> None:
    chunks = [
        make_chunk("a", score=0.9),
        make_chunk("a", score=0.9),
        make_chunk("b", score=0.5),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    assert chunk_ids(context) == ["a", "b"]
    assert [source.source_id for source in context.sources] == [1, 2]


def test_duplicate_chunk_id_keeps_the_highest_ranked_occurrence() -> None:
    chunks = [
        make_chunk("a", score=0.2, content="stale copy"),
        make_chunk("a", score=0.8, content="best copy"),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    assert [
        (source.content, source.similarity_score) for source in context.sources
    ] == [("best copy", 0.8)]


@pytest.mark.parametrize("blank", ["", " ", "\n\t ", "\n\n"])
def test_blank_content_is_skipped(blank: str) -> None:
    chunks = [make_chunk("a", score=0.9, content=blank), make_chunk("b", score=0.5)]

    context = assemble_context(chunks, UNBOUNDED)

    assert chunk_ids(context) == ["b"]
    assert [source.source_id for source in context.sources] == [1]


def test_only_blank_chunks_yield_an_empty_context() -> None:
    chunks = [
        make_chunk("a", score=0.9, content=" "),
        make_chunk("b", score=0.5, content=""),
    ]

    assert assemble_context(chunks, UNBOUNDED) == EMPTY_CONTEXT


def test_a_blank_duplicate_does_not_hide_a_non_blank_occurrence() -> None:
    # Regression: the blank check runs before the chunk id is recorded as seen,
    # so a higher-ranked blank copy must not drop the usable one.
    chunks = [
        make_chunk("a", score=0.9, content="  \n "),
        make_chunk("a", score=0.5, content="Restart the payments pod."),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    assert [
        (source.source_id, source.chunk_id, source.content)
        for source in context.sources
    ] == [(1, "a", "Restart the payments pod.")]


def test_a_blank_chunk_does_not_consume_the_budget_or_stop_assembly() -> None:
    first = make_chunk("a", score=0.9, content="A" * 50)
    blank = make_chunk("b", score=0.5, content=" " * 5000)
    last = make_chunk("c", score=0.1, content="C" * 50)
    exact_budget = len(assemble_context([first, last], UNBOUNDED).text)

    context = assemble_context([first, blank, last], exact_budget)

    assert chunk_ids(context) == ["a", "c"]


def test_many_small_chunks_all_fit() -> None:
    chunks = [
        make_chunk(f"chunk-{index}", score=1.0 - index / 10, content="x")
        for index in range(9)
    ]

    context = assemble_context(chunks, DEFAULT_CONTEXT_CHARACTER_BUDGET)

    assert len(context.sources) == 9
    assert len(context.text) <= DEFAULT_CONTEXT_CHARACTER_BUDGET


def test_the_tenth_source_is_numbered_and_its_wider_header_is_charged() -> None:
    # `source_id` 10 renders one character wider than 9, so the budget must be
    # charged for it. Every expectation below is written out independently of
    # the module: the headers are spelled out and the separator is a literal.
    source_path = "runbooks/example.md"
    chunks = [
        make_chunk(
            f"chunk-{index}",
            score=1.0 - index / 100,
            source_path=source_path,
            content=f"c{index}",
        )
        for index in range(10)
    ]
    expected_blocks = [
        f"[source {index + 1}] {source_path} (runbook)\nc{index}" for index in range(10)
    ]
    expected_text = "\n\n".join(expected_blocks)
    exact_budget = sum(len(block) for block in expected_blocks) + len("\n\n") * 9
    assert len(expected_blocks[9]) == len(expected_blocks[8]) + 1
    assert exact_budget == len(expected_text)

    context = assemble_context(chunks, exact_budget)

    assert [source.source_id for source in context.sources] == list(range(1, 11))
    assert context.sources[9].chunk_id == "chunk-9"
    assert context.text == expected_text
    assert "[source 10] runbooks/example.md (runbook)\nc9" in context.text
    assert len(context.text) == exact_budget

    tight = assemble_context(chunks, exact_budget - 1)

    assert [source.source_id for source in tight.sources] == list(range(1, 10))
    assert "[source 10]" not in tight.text
    assert tight.text == "\n\n".join(expected_blocks[:9])


def test_a_context_that_is_exactly_at_the_budget_is_kept_whole() -> None:
    chunks = [
        make_chunk("a", score=0.9, content="A" * 100),
        make_chunk("b", score=0.5, content="B" * 100),
    ]
    unbounded = assemble_context(chunks, UNBOUNDED)

    context = assemble_context(chunks, len(unbounded.text))

    assert context == unbounded
    assert len(context.text) == len(unbounded.text)


def test_one_character_over_the_budget_excludes_the_chunk() -> None:
    chunks = [
        make_chunk("a", score=0.9, content="A" * 100),
        make_chunk("b", score=0.5, content="B" * 100),
    ]
    exact_budget = len(assemble_context(chunks, UNBOUNDED).text)

    context = assemble_context(chunks, exact_budget - 1)

    assert chunk_ids(context) == ["a"]
    assert context == assemble_context([chunks[0]], UNBOUNDED)


def test_the_block_separator_is_counted_against_the_budget() -> None:
    first = make_chunk("a", score=0.9, content="A" * 100)
    second = make_chunk("b", score=0.5, content="B" * 100)
    # Room for both rendered blocks, but not for the separator between them.
    budget = rendered_length(first) + rendered_length(second)

    context = assemble_context([first, second], budget)

    assert chunk_ids(context) == ["a"]
    assert (
        len(
            assemble_context(
                [first, second], budget + len(SOURCE_BLOCK_SEPARATOR)
            ).sources
        )
        == 2
    )


def test_a_first_chunk_larger_than_the_budget_yields_an_empty_context() -> None:
    oversized = make_chunk("a", score=0.9, content="A" * 500)

    context = assemble_context([oversized], rendered_length(oversized) - 1)

    assert context == EMPTY_CONTEXT
    assert context.is_empty


def test_assembly_stops_at_the_first_chunk_that_does_not_fit() -> None:
    first = make_chunk("a", score=0.9, content="A" * 50)
    oversized = make_chunk("b", score=0.5, content="B" * 5000)
    small = make_chunk("c", score=0.1, content="C")
    budget = len(assemble_context([first, small], UNBOUNDED).text) + 1000

    context = assemble_context([first, oversized, small], budget)

    assert chunk_ids(context) == ["a"]
    assert "C" not in context.text


RANKED_TRIPLE = [
    make_chunk("a", score=0.9, content="A" * 20),
    make_chunk("b", score=0.5, content="B" * 20),
    make_chunk("c", score=0.1, content="C" * 20),
]


@pytest.mark.parametrize("fitting_count", [1, 2, 3])
def test_the_budget_that_exactly_fits_a_prefix_includes_that_prefix(
    fitting_count: int,
) -> None:
    prefix = RANKED_TRIPLE[:fitting_count]
    budget = len(assemble_context(prefix, UNBOUNDED).text)

    context = assemble_context(RANKED_TRIPLE, budget)

    assert chunk_ids(context) == [chunk.chunk_id for chunk in prefix]
    assert len(context.text) == budget


@pytest.mark.parametrize("fitting_count", [1, 2, 3])
def test_one_character_less_drops_the_last_chunk_of_the_prefix(
    fitting_count: int,
) -> None:
    prefix = RANKED_TRIPLE[:fitting_count]
    budget = len(assemble_context(prefix, UNBOUNDED).text) - 1

    context = assemble_context(RANKED_TRIPLE, budget)

    assert chunk_ids(context) == [chunk.chunk_id for chunk in prefix[:-1]]
    assert len(context.text) <= budget


def test_empty_input_yields_an_empty_context() -> None:
    context = assemble_context([])

    assert context == EMPTY_CONTEXT
    assert context.is_empty
    assert context.sources == ()
    assert context.text == ""


def test_a_non_empty_context_is_not_reported_as_empty() -> None:
    context = assemble_context([make_chunk("a")])

    assert not context.is_empty
    assert context.text


def test_source_metadata_is_copied_from_the_chunk() -> None:
    chunk = make_chunk(
        "chunk-x",
        score=0.42,
        document_id="doc-x",
        chunk_index=3,
        source_path="incidents/inc-1.md",
        document_type=DocumentType.INCIDENT,
        content="Multi-line\ncontent with ünicode.",
    )

    context = assemble_context([chunk], UNBOUNDED)

    assert context.sources == (
        ContextSource(
            source_id=1,
            chunk_id="chunk-x",
            document_id="doc-x",
            chunk_index=3,
            source_path="incidents/inc-1.md",
            document_type=DocumentType.INCIDENT,
            content="Multi-line\ncontent with ünicode.",
            similarity_score=0.42,
        ),
    )


def test_each_block_renders_the_header_and_then_the_content() -> None:
    chunks = [
        make_chunk(
            "a",
            score=0.9,
            source_path="runbooks/pods.md",
            content="Check the pod logs.",
        ),
        make_chunk(
            "b",
            score=0.5,
            source_path="incidents/inc-7.md",
            document_type=DocumentType.INCIDENT,
            content="The payments pod restarted.",
        ),
    ]

    text = assemble_context(chunks, UNBOUNDED).text

    assert text == (
        "[source 1] runbooks/pods.md (runbook)\n"
        "Check the pod logs."
        f"{SOURCE_BLOCK_SEPARATOR}"
        "[source 2] incidents/inc-7.md (incident)\n"
        "The payments pod restarted."
    )


def test_content_is_rendered_verbatim_and_never_truncated() -> None:
    chunks = [
        make_chunk(
            "a", score=0.9, content="  leading and trailing  \n\n  with blank lines  "
        ),
        make_chunk("b", score=0.5, content="```\ncode fence\n```"),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    for source, chunk in zip(context.sources, chunks, strict=True):
        assert source.content == chunk.content
        assert chunk.content in context.text


def test_every_source_id_resolves_back_to_its_source() -> None:
    chunks = [
        make_chunk("a", score=0.9),
        make_chunk("b", score=0.5),
        make_chunk("c", score=0.1),
    ]

    context = assemble_context(chunks, UNBOUNDED)

    for source in context.sources:
        assert context.source_by_id(source.source_id) is source


@pytest.mark.parametrize("source_id", [-1, 0, 3, 100])
def test_an_unknown_source_id_resolves_to_none(source_id: int) -> None:
    context = assemble_context(
        [make_chunk("a", score=0.9), make_chunk("b", score=0.5)], UNBOUNDED
    )

    assert context.source_by_id(source_id) is None


def test_an_empty_context_resolves_no_source_id() -> None:
    assert EMPTY_CONTEXT.source_by_id(1) is None


@pytest.mark.parametrize("budget", [0, -1, -100])
def test_a_budget_below_one_character_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="character_budget"):
        assemble_context([make_chunk("a")], budget)


def test_the_default_budget_holds_a_typical_retrieval_result() -> None:
    chunks = [
        make_chunk(f"chunk-{index}", score=1.0 - index / 10, content="x" * 1000)
        for index in range(5)
    ]

    context = assemble_context(chunks)

    assert len(context.sources) == 5
    assert len(context.text) <= DEFAULT_CONTEXT_CHARACTER_BUDGET


def test_the_context_module_depends_on_no_framework_llm_or_database_code() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(CONTEXT_MODULE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offending = {
        name
        for name in imported
        if name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS
        or any(
            name == module or name.startswith(f"{module}.")
            for module in FORBIDDEN_IMPORT_MODULES
        )
    }
    assert not offending, f"context assembly imports forbidden modules: {offending}"
