import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.embeddings.service import EmbeddingError
from app.main import app
from app.retrieval.service import SemanticRetrievalService

THREAD_COUNT = 8


class StubEngine:
    """Stand-in for a SQLAlchemy engine that records its disposal."""

    def __init__(self) -> None:
        self.disposals = 0

    def dispose(self) -> None:
        self.disposals += 1


class EngineFactory:
    """Hands out a fresh `StubEngine` per call and keeps them for inspection."""

    def __init__(self) -> None:
        self.engines: list[StubEngine] = []

    def __call__(self, *args: Any, **kwargs: Any) -> StubEngine:
        engine = StubEngine()
        self.engines.append(engine)
        return engine


class CountingFactory:
    """Records how often it was called and returns a distinct marker object.

    `delay` simulates a slow construction, so a concurrency test cannot pass by
    luck when threads happen not to interleave. `failures` makes the first calls
    raise, simulating a model or database that is unavailable at first.
    """

    def __init__(
        self, delay: float = 0.0, failures: int = 0, failure: Exception | None = None
    ) -> None:
        self.calls = 0
        self.delay = delay
        self.failures = failures
        self.failure = failure or EmbeddingError("Cannot load embedding model")

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        self.calls += 1
        time.sleep(self.delay)
        if self.calls <= self.failures:
            raise self.failure
        return object()


class Factories:
    """The replaced dependencies of the composition root."""

    def __init__(self, embedding_failures: int = 0) -> None:
        self.create_db_engine = EngineFactory()
        self.create_session_factory = CountingFactory()
        self.SqlAlchemyChunkRepository = CountingFactory()
        # Loading the real embedding model takes seconds.
        self.FastEmbedService = CountingFactory(delay=0.02, failures=embedding_failures)

    @property
    def engines(self) -> list[StubEngine]:
        return self.create_db_engine.engines

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "Factories":
        monkeypatch.setattr(dependencies, "_engine", None)
        monkeypatch.setattr(dependencies, "_retrieval_service", None)
        for name in (
            "create_db_engine",
            "create_session_factory",
            "SqlAlchemyChunkRepository",
            "FastEmbedService",
        ):
            monkeypatch.setattr(dependencies, name, getattr(self, name))
        return self


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> Iterator[Factories]:
    """Replace the expensive dependencies and reset the cached service."""
    yield Factories().install(monkeypatch)
    dependencies.dispose_retrieval_service()


def test_retrieval_service_is_created_on_first_use(factories: Factories) -> None:
    service = dependencies.get_retrieval_service()

    assert isinstance(service, SemanticRetrievalService)
    assert len(factories.engines) == 1
    assert factories.FastEmbedService.calls == 1


def test_repeated_calls_reuse_the_same_service(factories: Factories) -> None:
    first = dependencies.get_retrieval_service()
    second = dependencies.get_retrieval_service()

    assert second is first
    assert factories.FastEmbedService.calls == 1
    assert len(factories.engines) == 1


def test_concurrent_first_calls_share_one_embedding_model_and_engine(
    factories: Factories,
) -> None:
    # Without a lock each thread could build its own model and engine, and every
    # engine but one would be leaked.
    start = threading.Barrier(THREAD_COUNT)
    services: list[SemanticRetrievalService] = []
    services_lock = threading.Lock()

    def call() -> None:
        start.wait()
        service = dependencies.get_retrieval_service()
        with services_lock:
            services.append(service)

    threads = [threading.Thread(target=call) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(services) == THREAD_COUNT
    assert all(service is services[0] for service in services)
    assert len(factories.engines) == 1
    assert factories.FastEmbedService.calls == 1


def test_failed_initialization_disposes_the_engine_and_caches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories = Factories(embedding_failures=1).install(monkeypatch)

    with pytest.raises(EmbeddingError):
        dependencies.get_retrieval_service()

    assert [engine.disposals for engine in factories.engines] == [1]
    assert dependencies._retrieval_service is None
    assert dependencies._engine is None


def test_initialization_is_retried_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories = Factories(embedding_failures=1).install(monkeypatch)
    with pytest.raises(EmbeddingError):
        dependencies.get_retrieval_service()

    service = dependencies.get_retrieval_service()

    assert isinstance(service, SemanticRetrievalService)
    assert len(factories.engines) == 2
    assert [engine.disposals for engine in factories.engines] == [1, 0]
    dependencies.dispose_retrieval_service()


def test_dispose_releases_the_engine_and_the_next_call_builds_a_new_service(
    factories: Factories,
) -> None:
    first = dependencies.get_retrieval_service()

    dependencies.dispose_retrieval_service()
    second = dependencies.get_retrieval_service()

    assert factories.engines[0].disposals == 1
    assert second is not first
    assert len(factories.engines) == 2


def test_dispose_without_a_created_service_does_nothing(factories: Factories) -> None:
    dependencies.dispose_retrieval_service()
    dependencies.dispose_retrieval_service()

    assert factories.engines == []
    assert factories.FastEmbedService.calls == 0


def test_application_shutdown_disposes_the_engine(factories: Factories) -> None:
    with TestClient(app):
        dependencies.get_retrieval_service()
        assert factories.engines[0].disposals == 0

    assert factories.engines[0].disposals == 1
    assert dependencies._retrieval_service is None
