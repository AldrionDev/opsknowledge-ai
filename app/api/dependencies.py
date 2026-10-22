"""Composition root of the HTTP layer.

This is the only place where the API wires concrete implementations together,
like `app.ingestion.__main__` does for the ingestion CLI. Nothing is created at
import time: the retrieval service is built on first use, so importing the
application requires neither a database nor an embedding model.

The service is a process-wide singleton. Creating it loads the embedding model
and creates a database engine, both of which are expensive and meant to be
shared by every request. The lock makes concurrent first requests wait for one
initialization instead of each building its own model and engine.

Lifecycle:

- A failed initialization disposes the engine it created and caches nothing, so
  the next request starts over with a clean state.
- `dispose_retrieval_service()` releases the engine and clears the cache; the
  application calls it on shutdown, and a later request would build the service
  again.

Answering
---------

`get_answer_service` builds an `AnswerService` per request from the retrieval
singleton, which FastAPI resolves through `Depends(get_retrieval_service)`, and
the cached answer chain. The service itself is not cached: it is cheap and
stateless, and a cached instance would keep a retrieval service that a shutdown
or a dependency override has since replaced.

The chain (prompt and structured chat model) is created on the first `/ask` of an
application lifespan and reused until shutdown, where `clear_answer_chain()`
drops it, so the next lifespan reads the LLM configuration again. Creating it
makes no network call. It has its own lock, `_answer_lock`, taken only after the
retrieval dependency has returned and released `_lock`; no code path holds both
locks or takes one of them twice. A failed creation, such as a missing API key
(`LLMConfigurationError`), caches nothing and leaves the retrieval singleton as
it is; the next request tries again. The chain owns no resource that needs
explicit release, and the engine stays disposed by `dispose_retrieval_service()`
only.
"""

import threading
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine

from app.answering.prompt import build_answer_chain
from app.answering.service import AnswerChain, AnswerService
from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.session import create_db_engine, create_session_factory
from app.embeddings.fastembed_service import FastEmbedService
from app.llm.factory import create_chat_model
from app.retrieval.service import SemanticRetrievalService

_lock = threading.Lock()
_engine: Engine | None = None
_retrieval_service: SemanticRetrievalService | None = None

_answer_lock = threading.Lock()
_answer_chain: AnswerChain | None = None


def get_retrieval_service() -> SemanticRetrievalService:
    """Return the process-wide semantic retrieval service, creating it once."""
    global _engine, _retrieval_service
    with _lock:
        if _retrieval_service is None:
            engine = create_db_engine()
            try:
                repository = SqlAlchemyChunkRepository(create_session_factory(engine))
                service = SemanticRetrievalService(FastEmbedService(), repository)
            except BaseException:
                engine.dispose()
                raise
            _engine = engine
            _retrieval_service = service
        return _retrieval_service


def dispose_retrieval_service() -> None:
    """Release the cached engine and clear the cached service.

    Safe to call when nothing was created yet, and safe to call more than once.
    """
    global _engine, _retrieval_service
    with _lock:
        engine, _engine, _retrieval_service = _engine, None, None
        if engine is not None:
            engine.dispose()


def get_answer_service(
    retrieval_service: Annotated[
        SemanticRetrievalService, Depends(get_retrieval_service)
    ],
) -> AnswerService:
    """Return an answer service over the retrieval singleton and the cached chain.

    Raises `LLMConfigurationError` if the chain cannot be created because the LLM
    configuration is missing or invalid.
    """
    return AnswerService(retrieval_service, _get_answer_chain())


def _get_answer_chain() -> AnswerChain:
    global _answer_chain
    with _answer_lock:
        if _answer_chain is None:
            _answer_chain = build_answer_chain(create_chat_model())
        return _answer_chain


def clear_answer_chain() -> None:
    """Drop the cached chain; the next request builds it from the current configuration.

    Safe to call when nothing was created yet, and safe to call more than once.
    """
    global _answer_chain
    with _answer_lock:
        _answer_chain = None
