"""Deterministic splitting of knowledge documents into chunks.

Size unit: Python `str` characters (Unicode code points), measured with
`len(chunk.content)`. It is not a byte count and not a token count.

Algorithm:

1. The document is split into *blocks*: runs of non-blank lines (paragraphs,
   lists, tables, ...). Blank lines inside a fenced code block do not end a
   block. A blank line is a line that is empty or whitespace-only.
2. Consecutive blocks are packed greedily into one chunk while the chunk stays
   within the limit. A chunk is a slice of the original text, so the
   separators between blocks inside a chunk (blank lines, whitespace-only
   lines) are preserved exactly, as are indentation and trailing spaces
   (Markdown hard line breaks). A block that fits is never cut.
3. A block larger than the limit is split on its own, see `_split_oversized`.
   Its pieces are standalone chunks and are never merged with other blocks.

Unavoidable whitespace losses (a size-bounded, never-empty chunk cannot hold
them):

- the separators at chunk boundaries (the newline and any blank or
  whitespace-only lines between two chunks);
- whitespace-only slices produced by cutting a single over-long line;
- leading and trailing blank lines of a document that was not normalized by the
  loader.

Oversized fenced code blocks are split at line boundaries without re-adding the
fence markers: the pieces after the first one lack the opening fence, and only
the last one carries the closing fence. Combining characters of one grapheme
may end up in different chunks when a single line is cut by character count.

Input is expected to use `\\n` line endings, as produced by the document loader.
"""

from app.ingestion.models import DocumentChunk, KnowledgeDocument, build_chunk_id

DEFAULT_MAX_CHUNK_SIZE = 1000

_FENCE_MARKERS = ("```", "~~~")


def chunk_document(
    document: KnowledgeDocument, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[DocumentChunk]:
    """Split `document` into ordered chunks of at most `max_chunk_size` characters.

    Every chunk is an exact, contiguous slice of `document.content`, chunks do
    not overlap, and chunks are never empty or whitespace-only. Chunk indexes
    are zero-based and sequential. A document without content yields no chunks.

    Raises `ValueError` if `max_chunk_size` is smaller than 1.
    """
    if max_chunk_size < 1:
        raise ValueError(f"max_chunk_size must be at least 1, got {max_chunk_size}")

    return [
        DocumentChunk(
            chunk_id=build_chunk_id(document.document_id, index),
            document_id=document.document_id,
            chunk_index=index,
            source_path=document.source_path,
            document_type=document.document_type,
            content=content,
        )
        for index, content in enumerate(_chunk_texts(document.content, max_chunk_size))
    ]


def _chunk_texts(content: str, max_size: int) -> list[str]:
    texts: list[str] = []
    current: tuple[int, int] | None = None  # span of the chunk being built

    for start, end in _find_blocks(content):
        if end - start > max_size:
            if current is not None:
                texts.append(content[current[0] : current[1]])
                current = None
            texts.extend(_split_oversized(content[start:end], max_size))
        elif current is None:
            current = (start, end)
        elif end - current[0] <= max_size:
            current = (current[0], end)
        else:
            texts.append(content[current[0] : current[1]])
            current = (start, end)

    if current is not None:
        texts.append(content[current[0] : current[1]])
    return texts


def _find_blocks(content: str) -> list[tuple[int, int]]:
    """Return the `(start, end)` spans of the blocks of `content`.

    A span starts at the first character of the block's first line (indentation
    included) and ends after the last non-blank line (trailing spaces included).
    """
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end = 0
    fence: tuple[str, int] | None = None
    offset = 0

    for line in content.split("\n"):
        line_end = offset + len(line)
        stripped = line.strip()
        if fence is not None:
            if stripped:
                block_end = line_end
                if _closes_fence(stripped, fence):
                    fence = None
        elif stripped:
            if block_start is None:
                block_start = offset
            block_end = line_end
            fence = _opens_fence(stripped)
        elif block_start is not None:
            blocks.append((block_start, block_end))
            block_start = None
        offset = line_end + 1

    if block_start is not None:
        blocks.append((block_start, block_end))
    return blocks


def _opens_fence(stripped_line: str) -> tuple[str, int] | None:
    if not stripped_line.startswith(_FENCE_MARKERS):
        return None
    marker = stripped_line[0]
    length = len(stripped_line) - len(stripped_line.lstrip(marker))
    if marker == "`" and "`" in stripped_line[length:]:
        return None  # inline code such as ```code```, not a fence
    return marker, length


def _closes_fence(stripped_line: str, fence: tuple[str, int]) -> bool:
    marker, length = fence
    return len(stripped_line) >= length and stripped_line == marker * len(stripped_line)


def _split_oversized(block: str, max_size: int) -> list[str]:
    """Split one block that is longer than `max_size` into pieces.

    1. Whole lines are packed greedily into pieces, joined with the original
       `\\n`. Lines are kept verbatim.
    2. Whitespace-only lines are dropped at the start and end of a piece.
       Non-blank lines are never trimmed, so trailing hard-break spaces stay.
    3. A single line longer than `max_size` is cut into consecutive slices of
       exactly `max_size` characters (the last may be shorter). Slices that are
       whitespace-only are dropped.
    """
    pieces: list[str] = []
    lines: list[str] = []
    size = 0  # length of "\n".join(lines)

    def flush() -> None:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            pieces.append("\n".join(lines))
        lines.clear()

    for line in block.split("\n"):
        if len(line) > max_size:
            flush()
            pieces.extend(_cut_line(line, max_size))
        elif lines and size + 1 + len(line) <= max_size:
            lines.append(line)
            size += 1 + len(line)
        else:
            flush()
            if line.strip():
                lines.append(line)
                size = len(line)

    flush()
    return pieces


def _cut_line(line: str, max_size: int) -> list[str]:
    slices = (line[i : i + max_size] for i in range(0, len(line), max_size))
    return [piece for piece in slices if piece.strip()]
