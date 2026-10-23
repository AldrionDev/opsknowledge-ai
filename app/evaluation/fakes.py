"""Offline fake components of the evaluation harness: synthetic retrieval and a scripted LLM.

They back `--dry-run` and the evaluation tests. Neither needs a database, an
embedding model, credentials or network access, and this module imports nothing
that could create one. Nothing they produce measures the real system.

Synthetic retrieval
-------------------

`SyntheticRetrievalService` ranks a fixed list of chunks, by default the chunks of
the repository corpus as the real chunker produces them, with a hash score
instead of an embedding:

    score(query, chunk) = int(sha256(query + "\\0" + chunk_id)[:8], 16) / 2**32

The score lies in `[0, 1)`. Results are ordered by score descending, then by
`chunk_id`, like the real service, and the request validation (blank query,
limit outside `1..MAX_RESULT_LIMIT`) is the same. The ranking is deterministic and
independent of the input order, and it carries no semantic signal at all.

Scripted LLM
------------

`ScriptedFakeChain` is called with the prompt variables that `AnswerService`
passes to the real chain and acts according to a `FakeBehavior` chosen per
question text. It records every call.

`QUOTE_FIRST_SOURCE` answers with one claim citing source 1 and an evidence quote
taken from source 1's rendered block. The rendered context starts with the header
line of source 1, and the real block of source 2 starts with
`SOURCE_BLOCK_SEPARATOR` followed by the source 2 header prefix. A document may
contain that same text, but then it occurs *before* the real boundary, never
after it. Cutting at the **first** occurrence therefore always yields a prefix of
source 1's actual content. The quote is that prefix normalized like the grounding
validator normalizes (NFC, whitespace collapsed and trimmed), cut to
`QUOTE_LENGTH` characters, so it is a substring of the normalized content.
If the prefix is shorter than `MIN_EVIDENCE_QUOTE_LENGTH`, the behavior falls
back to `DECLINE`. The production context rendering and validation are used as
they are.
"""

import hashlib
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_core.runnables import Runnable, RunnableLambda

from app.answering.context import SOURCE_BLOCK_SEPARATOR, SOURCE_HEADER_TEMPLATE
from app.answering.models import AnswerDraft, Claim
from app.answering.validation import MIN_EVIDENCE_QUOTE_LENGTH
from app.evaluation.dataset import CORPUS_ROOT, EvalQuestion
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import MAX_RESULT_LIMIT, InvalidRetrievalRequestError

SYNTHETIC_RETRIEVAL = "synthetic_hash"
SCRIPTED_FAKE_MODEL = "scripted_fake"

QUOTE_LENGTH = 60
SYNTHETIC_CLAIM = "Synthetic claim quoting source 1."
ABSENT_QUOTE = "This scripted evidence quote does not occur in any source."

# The header of source 2 up to its path, e.g. "[source 2] ".
_SECOND_HEADER_PREFIX = SOURCE_HEADER_TEMPLATE.partition("{source_path}")[0].format(
    source_id=2
)
_FIRST_HEADER_PREFIX = SOURCE_HEADER_TEMPLATE.partition("{source_path}")[0].format(
    source_id=1
)


class SyntheticRetrievalService:
    """Deterministic hash-ranked retrieval over fixed chunks; see the module docstring."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._chunks = tuple(chunks)

    @classmethod
    def from_corpus(
        cls, corpus_root: Path = CORPUS_ROOT
    ) -> "SyntheticRetrievalService":
        """Build from the chunks of the corpus at `corpus_root`; reads files only."""
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source_path=chunk.source_path,
                document_type=chunk.document_type,
                content=chunk.content,
                similarity_score=0.0,
            )
            for document in load_knowledge_documents(corpus_root)
            for chunk in chunk_document(document)
        ]
        return cls(chunks)

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        if not query.strip():
            raise InvalidRetrievalRequestError("query is empty or whitespace-only")
        if not 1 <= limit <= MAX_RESULT_LIMIT:
            raise InvalidRetrievalRequestError(
                f"limit must be between 1 and {MAX_RESULT_LIMIT}, got {limit}"
            )
        scored = [
            replace(chunk, similarity_score=synthetic_score(query, chunk.chunk_id))
            for chunk in self._chunks
        ]
        scored.sort(key=lambda chunk: (-chunk.similarity_score, chunk.chunk_id))
        return scored[:limit]


def synthetic_score(query: str, chunk_id: str) -> float:
    digest = hashlib.sha256(f"{query}\0{chunk_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32


class FakeBehavior(StrEnum):
    DECLINE = "decline"
    QUOTE_FIRST_SOURCE = "quote_first_source"
    UNKNOWN_CITATION = "unknown_citation"
    MISSING_EVIDENCE = "missing_evidence"
    INCONSISTENT = "inconsistent"
    MALFORMED = "malformed"
    TRANSIENT_PROVIDER_ERROR = "transient_provider_error"
    NONRECOVERABLE_PROVIDER_ERROR = "nonrecoverable_provider_error"


# The dry-run cycle: every outcome class and most invalid-output codes appear,
# and transient provider errors are never consecutive.
DRY_RUN_CYCLE = (
    FakeBehavior.QUOTE_FIRST_SOURCE,
    FakeBehavior.DECLINE,
    FakeBehavior.UNKNOWN_CITATION,
    FakeBehavior.MISSING_EVIDENCE,
    FakeBehavior.INCONSISTENT,
    FakeBehavior.MALFORMED,
    FakeBehavior.TRANSIENT_PROVIDER_ERROR,
)


def build_dry_run_script(
    questions: Sequence[EvalQuestion],
) -> dict[str, FakeBehavior]:
    """Assign `DRY_RUN_CYCLE` to the questions in order, keyed by question text."""
    script: dict[str, FakeBehavior] = {}
    for index, question in enumerate(questions):
        if question.question in script:
            raise ValueError(f"duplicate question text: {question.id}")
        script[question.question] = DRY_RUN_CYCLE[index % len(DRY_RUN_CYCLE)]
    return script


class ScriptedFakeChain:
    """Stand-in for the answer chain; behaves per question text, records every call."""

    def __init__(
        self,
        script: Mapping[str, FakeBehavior],
        default: FakeBehavior = FakeBehavior.DECLINE,
    ) -> None:
        self._script = dict(script)
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def __call__(self, inputs: dict[str, Any]) -> Any:
        self.calls.append(dict(inputs))
        behavior = self._script.get(inputs["question"], self._default)
        return _act(behavior, inputs)

    def as_runnable(self) -> Runnable[dict[str, Any], Any]:
        call: Callable[[dict[str, Any]], Any] = self
        return RunnableLambda(call)


def _act(behavior: FakeBehavior, inputs: dict[str, Any]) -> Any:
    if behavior is FakeBehavior.QUOTE_FIRST_SOURCE:
        quote = first_source_quote(inputs["sources"])
        if quote is None:
            return _declined()
        return _draft(1, quote)
    if behavior is FakeBehavior.UNKNOWN_CITATION:
        return _draft(int(inputs["source_count"]) + 1, ABSENT_QUOTE)
    if behavior is FakeBehavior.MISSING_EVIDENCE:
        return _draft(1, ABSENT_QUOTE)
    if behavior is FakeBehavior.INCONSISTENT:
        return AnswerDraft(
            insufficient=True,
            claims=[
                Claim(text=SYNTHETIC_CLAIM, source_id=1, evidence_quote=ABSENT_QUOTE)
            ],
        )
    if behavior is FakeBehavior.MALFORMED:
        return {"insufficient": "not a draft"}
    if behavior is FakeBehavior.TRANSIENT_PROVIDER_ERROR:
        raise ModelRateLimitError("scripted transient provider failure")
    if behavior is FakeBehavior.NONRECOVERABLE_PROVIDER_ERROR:
        raise ModelAuthenticationError("scripted nonrecoverable provider failure")
    return _declined()


def first_source_quote(sources: str) -> str | None:
    """A verbatim quote of source 1's content, or `None`; see the module docstring."""
    if not sources.startswith(_FIRST_HEADER_PREFIX) or "\n" not in sources:
        return None
    content_start = sources.index("\n") + 1
    boundary = sources.find(
        SOURCE_BLOCK_SEPARATOR + _SECOND_HEADER_PREFIX, content_start
    )
    prefix = (
        sources[content_start:] if boundary == -1 else sources[content_start:boundary]
    )
    quote = _normalize(prefix)[:QUOTE_LENGTH].strip()
    if len(quote) < MIN_EVIDENCE_QUOTE_LENGTH:
        return None
    return quote


def _normalize(text: str) -> str:
    # The same normalization as `app.answering.validation`.
    return " ".join(unicodedata.normalize("NFC", text).split())


def _draft(source_id: int, quote: str) -> AnswerDraft:
    return AnswerDraft(
        insufficient=False,
        claims=[Claim(text=SYNTHETIC_CLAIM, source_id=source_id, evidence_quote=quote)],
    )


def _declined() -> AnswerDraft:
    return AnswerDraft(insufficient=True, claims=[])
