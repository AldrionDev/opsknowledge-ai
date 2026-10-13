"""Assembly of retrieved chunks into a numbered, bounded answering context.

`assemble_context` is a pure function: ranked `RetrievedChunk` items in, an
`AssembledContext` out. It performs no retrieval, no prompt construction and no
model call, and it depends on no web framework, LLM client or database.

    ranked chunks
        -> sort, drop blank content, deduplicate
        -> fit into the character budget, whole chunks only
        -> numbered sources + one rendered context text

Size unit: Python `str` characters (Unicode code points), measured with `len()`,
the same unit the document chunker uses. It is not a byte count and not a token
count.

Pipeline, in this order:

1. **Ordering.** By `similarity_score` descending, then `chunk_id` ascending,
   the same key `SemanticRetrievalService.retrieve()` applies. The input is
   already ranked; sorting again makes this function deterministic on its own,
   independent of the caller. The caller's sequence is never mutated. See
   *Ordering limits* below for what "independent of the caller" assumes.
2. **Blank content is skipped**, before the `chunk_id` is recorded as seen:
   a blank duplicate must not hide a non-blank occurrence of the same chunk.
   A skipped chunk costs no budget and never stops assembly.
3. **Deduplication** by `chunk_id`, keeping the highest-ranked non-blank
   occurrence.
4. **Budget fitting.** Whole chunks only, in ranked order; a chunk is never
   truncated. Assembly **stops at the first chunk that does not fit**, so the
   result is always a ranked prefix: once a chunk is excluded, no lower-ranked
   chunk can enter. A first chunk larger than the budget therefore yields an
   empty context.
5. **Numbering.** `source_id` is 1-based and contiguous over the selected
   chunks, so it is also the position in `AssembledContext.sources`.
6. **Rendering.** Each source becomes a header line followed by its content
   verbatim; blocks are joined with `SOURCE_BLOCK_SEPARATOR`.

The budget is accounted against the **rendered text**, not the raw chunk
contents: a chunk costs the length of its rendered block plus the separator that
joins it to the previous block (the first block pays no separator). So
`len(context.text) <= character_budget` always holds for the text that a prompt
will actually carry.

Ordering limits. Equal inputs always produce equal results. Ordering that is
independent of the *input order* additionally assumes finite similarity scores:
a NaN score makes the sort key a non-total order, so the result then depends on
where the chunk appeared in the input. The same holds for two records that share
a `chunk_id` and a `similarity_score` but carry different content: the stable
sort keeps whichever came first. Neither case arises in the retrieval contract
(`search_similar` returns distinct rows keyed by `chunk_id`, and the scores come
from pgvector distances), so this module validates nothing, normalizes no score
and adds no further tie-breaker.

`DEFAULT_CONTEXT_CHARACTER_BUDGET` is a module constant, deliberately not
environment-configurable; token-based budgeting is out of scope.

Content is stored and rendered unchanged: nothing is stripped, normalized or
truncated. Citation validation matches evidence quotes against
`ContextSource.content`, so rewriting content here would silently change what
counts as grounded.

Rendering states only the source number and its origin. Delimiting the context
as untrusted data, and every other instruction to the model, belongs to the
prompt, not here, and nothing in this module assumes a model provider.

An empty input, an input whose chunks are all blank, and an input where not even
the first chunk fits all return `EMPTY_CONTEXT`: a normal value, not an error.
A consumer must treat it as "no grounded context available" and not call a
model.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.ingestion.models import DocumentType
from app.retrieval.models import RetrievedChunk

# A typical retrieval is 5 chunks of at most 1000 characters (the `POST /search`
# default limit and the default chunk size), about 5000 characters. 8000 leaves
# headroom for the source headers and for larger result sets while keeping the
# context bounded at roughly 2000 tokens, far inside the context window of the
# configured chat models.
DEFAULT_CONTEXT_CHARACTER_BUDGET = 8000

SOURCE_HEADER_TEMPLATE = "[source {source_id}] {source_path} ({document_type})"
SOURCE_BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ContextSource:
    """One numbered source of the assembled context.

    `source_id` is 1-based and equals the position of this source in
    `AssembledContext.sources`. Every other field is copied unchanged from the
    retrieved chunk, so a consumer can resolve a cited `source_id` back to the
    chunk it came from.
    """

    source_id: int
    chunk_id: str
    document_id: str
    chunk_index: int
    source_path: str
    document_type: DocumentType
    content: str
    similarity_score: float


@dataclass(frozen=True)
class AssembledContext:
    """Numbered sources and the rendered context text built from them.

    Invariants:

    - `len(text) <= character_budget` of the call that produced it;
    - `sources` is a ranked prefix of the deduplicated, non-blank input;
    - `sources[i].source_id == i + 1`;
    - `sources` is empty if and only if `text` is empty;
    - every source's `content` occurs verbatim in `text`.

    `sources` is a tuple: the value object stays immutable, like the frozen
    dataclasses it is built from.
    """

    sources: tuple[ContextSource, ...]
    text: str

    @property
    def is_empty(self) -> bool:
        """Whether the context carries no source at all."""
        return not self.sources

    def source_by_id(self, source_id: int) -> ContextSource | None:
        """Return the source numbered `source_id`, or `None` if there is none."""
        if 1 <= source_id <= len(self.sources):
            return self.sources[source_id - 1]
        return None


EMPTY_CONTEXT = AssembledContext(sources=(), text="")


def assemble_context(
    chunks: Sequence[RetrievedChunk],
    character_budget: int = DEFAULT_CONTEXT_CHARACTER_BUDGET,
) -> AssembledContext:
    """Assemble `chunks` into a numbered context of at most `character_budget` characters.

    Equal inputs produce equal results, and `chunks` is left untouched. Chunks
    are included whole, in ranked order, and assembly stops at the first chunk
    that does not fit; `EMPTY_CONTEXT` is returned when nothing fits or there is
    nothing to include. See the module docstring for the full pipeline.

    Raises `ValueError` if `character_budget` is smaller than 1.
    """
    if character_budget < 1:
        raise ValueError(f"character_budget must be at least 1, got {character_budget}")

    ranked = sorted(chunks, key=lambda chunk: (-chunk.similarity_score, chunk.chunk_id))

    sources: list[ContextSource] = []
    blocks: list[str] = []
    seen_chunk_ids: set[str] = set()
    used_characters = 0

    for chunk in ranked:
        if not chunk.content.strip():
            continue
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)

        source = _to_source(len(sources) + 1, chunk)
        block = _render_block(source)
        # The separator only exists between two blocks, so the first block is
        # free of it; the header length depends on `source_id`, which is why the
        # cost is measured on the rendered block.
        cost = len(block) + (len(SOURCE_BLOCK_SEPARATOR) if blocks else 0)
        if used_characters + cost > character_budget:
            break

        used_characters += cost
        sources.append(source)
        blocks.append(block)

    if not sources:
        return EMPTY_CONTEXT
    return AssembledContext(
        sources=tuple(sources), text=SOURCE_BLOCK_SEPARATOR.join(blocks)
    )


def _to_source(source_id: int, chunk: RetrievedChunk) -> ContextSource:
    return ContextSource(
        source_id=source_id,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        source_path=chunk.source_path,
        document_type=chunk.document_type,
        content=chunk.content,
        similarity_score=chunk.similarity_score,
    )


def _render_block(source: ContextSource) -> str:
    header = SOURCE_HEADER_TEMPLATE.format(
        source_id=source.source_id,
        source_path=source.source_path,
        document_type=source.document_type,
    )
    return f"{header}\n{source.content}"
