import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.models import (
    DocumentType,
    KnowledgeDocument,
    build_chunk_id,
    build_document_id,
)


def make_document(
    content: str,
    source_path: str = "runbooks/a.md",
    document_type: DocumentType = DocumentType.RUNBOOK,
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=build_document_id(source_path),
        source_path=source_path,
        document_type=document_type,
        content=content,
    )


def chunk_contents(content: str, max_chunk_size: int) -> list[str]:
    chunks = chunk_document(make_document(content), max_chunk_size)
    return [chunk.content for chunk in chunks]


# --- configuration ---


def test_default_limit_is_1000_characters() -> None:
    assert len(chunk_document(make_document("a" * 1000))) == 1
    assert len(chunk_document(make_document("a" * 1001))) == 2
    assert len(chunk_document(make_document("a" * 1001), 2000)) == 1


@pytest.mark.parametrize("max_chunk_size", [0, -1])
def test_non_positive_max_chunk_size_is_rejected(max_chunk_size: int) -> None:
    with pytest.raises(ValueError, match="max_chunk_size"):
        chunk_document(make_document("text"), max_chunk_size)


def test_max_chunk_size_of_one_is_valid() -> None:
    assert chunk_contents("ab\n\nc", 1) == ["a", "b", "c"]


# --- short documents and preserved whitespace ---


def test_short_document_produces_one_unchanged_chunk() -> None:
    document = make_document("# Title\n\nBody text.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize(
    "content",
    [
        "# T\n\n\n\nA  \n \t\n  B",
        "  indented first line\n\n\nnext  ",
        "para one\n\n\n\n\npara two\n\t\n\npara three",
    ],
)
def test_short_document_keeps_its_content_exactly(content: str) -> None:
    assert chunk_contents(content, 1000) == [content]


def test_separators_between_blocks_in_one_chunk_are_preserved() -> None:
    assert chunk_contents("aaa\n\n\nbbb", 9) == ["aaa\n\n\nbbb"]


def test_separator_length_counts_toward_the_limit() -> None:
    assert chunk_contents("aaa\n\n\nbbb", 8) == ["aaa", "bbb"]


# --- packing and natural boundaries ---


def test_blocks_are_packed_greedily_up_to_the_limit() -> None:
    content = "\n\n".join(["a" * 10, "b" * 10, "c" * 10])

    assert chunk_contents(content, 22) == ["a" * 10 + "\n\n" + "b" * 10, "c" * 10]
    assert chunk_contents(content, 21) == ["a" * 10, "b" * 10, "c" * 10]


def test_block_of_exactly_the_limit_is_not_split() -> None:
    assert chunk_contents("a" * 10, 10) == ["a" * 10]
    assert chunk_contents("a" * 11, 10) == ["a" * 10, "a"]


def test_chunks_cut_only_at_paragraph_boundaries_when_paragraphs_fit() -> None:
    paragraphs = [f"Paragraph {i} " + " ".join(["word"] * i) for i in range(1, 12)]
    limit = max(len(p) for p in paragraphs) + 20

    contents = chunk_contents("\n\n".join(paragraphs), limit)

    assert len(contents) > 1
    assert [p for content in contents for p in content.split("\n\n")] == paragraphs
    assert all(len(content) <= limit for content in contents)


def test_headings_lists_and_tables_without_blank_lines_stay_together() -> None:
    block = "## Heading\n- one\n- two\n| a | b |\n|---|---|"

    assert chunk_contents(block, len(block)) == [block]


# --- ordering, indexes, identity, metadata ---


def test_chunks_are_in_document_order_with_sequential_zero_based_indexes() -> None:
    content = "\n\n".join(f"paragraph number {i:02d}" for i in range(10))

    chunks = chunk_document(make_document(content), 40)

    assert len(chunks) > 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert "\n\n".join(chunk.content for chunk in chunks) == content


def test_chunk_ids_are_derived_from_document_id_and_index_and_are_unique() -> None:
    document = make_document("\n\n".join(["x" * 10] * 6))

    chunks = chunk_document(document, 10)

    assert len(chunks) == 6
    assert [chunk.chunk_id for chunk in chunks] == [
        build_chunk_id(document.document_id, index) for index in range(6)
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == 6


def test_source_metadata_is_preserved_on_every_chunk() -> None:
    document = make_document(
        "\n\n".join(["x" * 10] * 4),
        source_path="incidents/inc-1.md",
        document_type=DocumentType.INCIDENT,
    )

    chunks = chunk_document(document, 10)

    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.source_path == "incidents/inc-1.md"
        assert chunk.document_type is DocumentType.INCIDENT


def test_repeated_chunking_produces_identical_results() -> None:
    document = make_document("\n\n".join(f"paragraph {i}" for i in range(20)))

    assert chunk_document(document, 30) == chunk_document(document, 30)


def test_different_documents_with_same_content_get_different_chunk_ids() -> None:
    content = "\n\n".join(["same text"] * 4)
    first = chunk_document(make_document(content, "runbooks/a.md"), 9)
    second = chunk_document(make_document(content, "runbooks/b.md"), 9)

    assert [c.content for c in first] == [c.content for c in second]
    assert {c.chunk_id for c in first}.isdisjoint({c.chunk_id for c in second})


# --- no empty chunks ---


def test_blank_lines_do_not_produce_empty_chunks() -> None:
    content = "\n\n  \n\nA\n   \n\n\nB\n\n   \n"

    assert chunk_contents(content, 100) == ["A\n   \n\n\nB"]
    assert chunk_contents(content, 1) == ["A", "B"]


@pytest.mark.parametrize("content", ["", "\n", "  \n \t\n\n"])
def test_content_without_text_produces_no_chunks(content: str) -> None:
    assert chunk_contents(content, 10) == []


# --- oversized blocks ---


def test_oversized_list_is_split_at_lines_and_keeps_hard_breaks() -> None:
    items = ["- aaaa  ", "- bbbb  ", "- cccc  ", "- dddd  "]
    content = "intro\n\n" + "\n".join(items) + "\n\noutro"

    assert chunk_contents(content, 17) == [
        "intro",
        "- aaaa  \n- bbbb  ",
        "- cccc  \n- dddd  ",
        "outro",
    ]


def test_single_overlong_line_is_cut_into_limit_sized_slices() -> None:
    assert chunk_contents("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


def test_overlong_line_among_other_lines_is_cut_between_line_pieces() -> None:
    content = "short\n" + "y" * 25 + "\nafter"

    assert chunk_contents(content, 10) == [
        "short",
        "y" * 10,
        "y" * 10,
        "y" * 5,
        "after",
    ]


def test_hard_split_keeps_spaces_next_to_the_cut() -> None:
    assert chunk_contents("abcd  efgh  ", 6) == ["abcd  ", "efgh  "]


@pytest.mark.parametrize(
    ("line", "limit", "expected"),
    [
        # The trailing "  " forms a whitespace-only slice, which cannot be a chunk.
        ("abcdefgh  ", 8, ["abcdefgh"]),
        # Only the second of the two trailing spaces forms such a slice.
        ("abcdefg  ", 8, ["abcdefg "]),
    ],
)
def test_hard_split_drops_whitespace_only_slices(
    line: str, limit: int, expected: list[str]
) -> None:
    assert chunk_contents(line, limit) == expected


def test_whitespace_run_longer_than_the_limit_never_becomes_a_chunk() -> None:
    contents = chunk_contents("a" + " " * 30 + "b", 5)

    assert contents == ["a    ", " b"]
    assert all(content.strip() for content in contents)


# --- fenced code blocks ---


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_fenced_block_with_blank_line_is_not_split_when_it_fits(marker: str) -> None:
    fence = f"{marker}\na\n\nb\n{marker}"

    assert chunk_contents(f"intro\n\n{fence}\n\noutro", 12) == ["intro", fence, "outro"]
    assert chunk_contents(f"intro\n\n{fence}\n\noutro", 1000) == [
        f"intro\n\n{fence}\n\noutro"
    ]


def test_indented_fence_keeps_its_indentation() -> None:
    content = "1. Step\n\n   ```bash\n   kubectl get pods\n   ```\n\n2. Next"

    assert chunk_contents(content, 1000) == [content]
    assert chunk_contents(content, 38) == [
        "1. Step",
        "   ```bash\n   kubectl get pods\n   ```",
        "2. Next",
    ]


def test_shorter_marker_does_not_close_a_fence() -> None:
    fence = "````\na\n```\n\nb\n````"

    assert chunk_contents(f"p\n\n{fence}", 18) == ["p", fence]


def test_unclosed_fence_extends_to_the_end_of_the_document() -> None:
    fence = "```\na\n\nb\n\nc"

    assert chunk_contents(f"p\n\n{fence}", 11) == ["p", fence]


def test_inline_triple_backticks_do_not_open_a_fence() -> None:
    assert chunk_contents("p\n\n```x``` y\n\nz", 12) == ["p\n\n```x``` y", "z"]


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        # The blank line ends the first piece and is dropped.
        (25, ["```bash\n  indented  ", "    deeper\n   \nlast\n```"]),
        # The blank line would start the second piece and is dropped.
        (20, ["```bash\n  indented  ", "    deeper\n   \nlast", "```"]),
    ],
)
def test_oversized_fence_is_split_at_lines_preserving_whitespace(
    limit: int, expected: list[str]
) -> None:
    fence = "```bash\n  indented  \n\n    deeper\n   \nlast\n```"

    contents = chunk_contents(fence, limit)

    assert contents == expected
    assert all(len(content) <= limit and content.strip() for content in contents)


# --- unicode ---


def test_size_is_measured_in_characters_not_bytes() -> None:
    text = "é" * 10  # 20 bytes in UTF-8

    assert chunk_contents(text, 10) == [text]


def test_unicode_text_is_split_by_code_points() -> None:
    assert chunk_contents("😀" * 5, 4) == ["😀" * 4, "😀"]
    assert chunk_contents("日本語\n\nテスト", 3) == ["日本語", "テスト"]


# --- exactness invariants ---


MIXED_DOCUMENT = "\n".join(
    [
        "# Title w001",
        "",
        "Intro w002 w003  ",
        "line w004  ",
        "",
        "",
        "",
        "- item w005",
        "- item w006",
        "  - nested w007",
        "",
        "| h w008 | h w009 |",
        "|---|---|",
        "| w010 | w011 |",
        "",
        "```bash",
        "  cmd w012  ",
        "",
        "    cmd w013",
        "   ",
        "```",
        "",
        " ".join(f"word{i:03d}" for i in range(30)),
        "",
        "Ünïcödé 日本語 😀 w014",
        "\t",
        "last w015  ",
    ]
)


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200, 1000])
def test_chunks_are_exact_ordered_substrings_with_only_whitespace_between(
    limit: int,
) -> None:
    source = MIXED_DOCUMENT
    chunks = chunk_document(make_document(source), limit)

    position = 0
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.content.strip(), "chunk must not be empty or whitespace-only"
        assert len(chunk.content) <= limit
        found = source.find(chunk.content, position)
        assert found != -1, "chunk must be an in-order substring of the source"
        assert source[position:found].strip() == "", "only whitespace may be omitted"
        position = found + len(chunk.content)
    assert source[position:].strip() == ""
