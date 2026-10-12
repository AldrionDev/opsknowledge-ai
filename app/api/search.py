"""HTTP boundary of semantic search.

`POST /search` validates the request, delegates to `SemanticRetrievalService`,
and serializes the ranked chunks. It contains no embedding, SQL, pgvector,
ranking or answer-generation logic.

Request validation mirrors the retrieval service contract: the query must not be
empty or whitespace-only, and `limit` must be within `1..MAX_RESULT_LIMIT`, the
bound imported from the service so the two cannot drift apart. Invalid input is
therefore rejected by Pydantic with HTTP 422 before the service is called.

The result order is the order the retrieval service returned; this module never
reorders it.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.api.dependencies import get_retrieval_service
from app.ingestion.models import DocumentType
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import MAX_RESULT_LIMIT, SemanticRetrievalService

# Typical retrieval size, well inside the range the retrieval service supports.
DEFAULT_RESULT_LIMIT = 5


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Natural-language search query.",
        examples=["Why is the payments pod repeatedly restarting?"],
    )
    limit: int = Field(
        default=DEFAULT_RESULT_LIMIT,
        ge=1,
        le=MAX_RESULT_LIMIT,
        description=f"Maximum number of results, between 1 and {MAX_RESULT_LIMIT}.",
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        # The query itself is passed on unchanged: trimming it here would change
        # what gets embedded.
        if not value.strip():
            raise ValueError("query is empty or whitespace-only")
        return value


class SearchResult(BaseModel):
    """One retrieved chunk with its source metadata and similarity score."""

    chunk_id: str
    document_id: str
    chunk_index: int
    source_path: str
    document_type: DocumentType
    content: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


router = APIRouter(tags=["search"])


@router.post("/search", summary="Semantic search over the knowledge corpus")
def search(
    request: SearchRequest,
    retrieval_service: Annotated[
        SemanticRetrievalService, Depends(get_retrieval_service)
    ],
) -> SearchResponse:
    """Return the chunks most similar to the query, most similar first."""
    results = retrieval_service.retrieve(request.query, request.limit)
    return SearchResponse(
        query=request.query,
        results=[_to_search_result(result) for result in results],
    )


def _to_search_result(chunk: RetrievedChunk) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        source_path=chunk.source_path,
        document_type=chunk.document_type,
        content=chunk.content,
        similarity_score=chunk.similarity_score,
    )
