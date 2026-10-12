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
"""

import threading

from sqlalchemy import Engine

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.session import create_db_engine, create_session_factory
from app.embeddings.fastembed_service import FastEmbedService
from app.retrieval.service import SemanticRetrievalService

_lock = threading.Lock()
_engine: Engine | None = None
_retrieval_service: SemanticRetrievalService | None = None


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
