import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.answering.service import AnswerService
from app.api import dependencies
from app.embeddings.service import EmbeddingError
from app.ingestion.chunk_repository import ChunkMatch
from app.llm import config
from app.llm.errors import LLMConfigurationError
from app.llm.factory import create_chat_model
from app.main import app
from app.retrieval.service import SemanticRetrievalService
from tests.fake_embedding import FakeEmbeddingService
from tests.fake_llm import (
    FakeStructuredChatModel,
    block_network,
    clear_llm_environment,
)

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


# --- answer service -------------------------------------------------------------

QUESTION = "Why is the payments pod repeatedly restarting?"
JOIN_TIMEOUT_SECONDS = 10.0


class ProducingFactory(CountingFactory):
    """A `CountingFactory` that returns `produce()` instead of a marker object."""

    def __init__(self, produce: Callable[[], object], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.produce = produce
        self.products: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        super().__call__(*args, **kwargs)
        product = self.produce()
        self.products.append(product)
        return product


class RecordingRepository:
    """Chunk repository that records searches and stores nothing."""

    def __init__(self) -> None:
        self.searches = 0

    def search_similar(
        self, query_embedding: Sequence[float], limit: int
    ) -> list[ChunkMatch]:
        self.searches += 1
        return []


class ObservingEngine(StubEngine):
    """Records whether the answer lock was held while the engine was disposed."""

    def __init__(self) -> None:
        super().__init__()
        self.answer_lock_held_on_dispose: list[bool] = []

    def dispose(self) -> None:
        self.answer_lock_held_on_dispose.append(dependencies._answer_lock.locked())
        super().dispose()


class ObservingEngineFactory(EngineFactory):
    def __call__(self, *args: Any, **kwargs: Any) -> StubEngine:
        engine = ObservingEngine()
        self.engines.append(engine)
        return engine


class AnsweringFactories(Factories):
    """Factories under which `/ask` works end to end without model or database.

    Each factory also records which lock is held while it runs, so a test can
    see that neither lock is taken inside the other one.
    """

    def __init__(self, chat_model_failures: int = 0) -> None:
        super().__init__()
        self.create_db_engine = ObservingEngineFactory()
        self.answer_lock_held_on_embedding: list[bool] = []
        self.retrieval_lock_held_on_chat_model: list[bool] = []
        self.FastEmbedService = ProducingFactory(self._embedding, delay=0.02)
        self.SqlAlchemyChunkRepository = ProducingFactory(RecordingRepository)
        self.create_chat_model = ProducingFactory(
            self._chat_model,
            delay=0.02,
            failures=chat_model_failures,
            failure=LLMConfigurationError("ANTHROPIC_API_KEY is not set or is blank"),
        )

    def _embedding(self) -> FakeEmbeddingService:
        self.answer_lock_held_on_embedding.append(dependencies._answer_lock.locked())
        return FakeEmbeddingService()

    def _chat_model(self) -> FakeStructuredChatModel:
        self.retrieval_lock_held_on_chat_model.append(dependencies._lock.locked())
        return FakeStructuredChatModel()

    @property
    def repositories(self) -> list[RecordingRepository]:
        return self.SqlAlchemyChunkRepository.products

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "AnsweringFactories":
        super().install(monkeypatch)
        monkeypatch.setattr(dependencies, "_answer_chain", None)
        monkeypatch.setattr(dependencies, "create_chat_model", self.create_chat_model)
        return self


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM configuration from the developer environment and no network."""
    clear_llm_environment(monkeypatch)
    block_network(monkeypatch)


@pytest.fixture
def answering(offline: None, monkeypatch: pytest.MonkeyPatch) -> AnsweringFactories:
    # No cleanup call that takes a lock: after a deadlock it would hang the run
    # instead of failing it. `install` lets monkeypatch restore the cached state.
    return AnsweringFactories().install(monkeypatch)


def ask(client: TestClient) -> int:
    return client.post("/ask", json={"question": QUESTION}).status_code


def test_answer_service_is_built_per_request_on_one_cached_chain(
    answering: AnsweringFactories,
) -> None:
    retrieval = dependencies.get_retrieval_service()

    first = dependencies.get_answer_service(retrieval)
    second = dependencies.get_answer_service(retrieval)

    assert isinstance(first, AnswerService)
    assert second is not first
    assert answering.create_chat_model.calls == 1
    assert answering.FastEmbedService.calls == 1
    assert len(answering.engines) == 1


def test_requests_of_one_lifespan_reuse_the_chain_and_the_retrieval_service(
    answering: AnsweringFactories,
) -> None:
    with TestClient(app) as client:
        statuses = [ask(client) for _ in range(3)]
        client.post("/search", json={"query": QUESTION})

    assert statuses == [200, 200, 200]
    assert answering.create_chat_model.calls == 1
    assert answering.FastEmbedService.calls == 1
    assert len(answering.engines) == 1
    assert [repository.searches for repository in answering.repositories] == [4]


def test_concurrent_first_requests_create_one_model_engine_and_chain(
    answering: AnsweringFactories,
) -> None:
    # The factories are slow, so the threads really meet inside both
    # initializations; a nested or re-entrant lock would hang a thread.
    start = threading.Barrier(THREAD_COUNT)
    statuses: list[int] = []
    statuses_lock = threading.Lock()

    def call() -> None:
        client = TestClient(app)
        start.wait()
        status = ask(client)
        with statuses_lock:
            statuses.append(status)

    threads = [threading.Thread(target=call, daemon=True) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + JOIN_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    assert not any(thread.is_alive() for thread in threads), "deadlock"
    assert statuses == [200] * THREAD_COUNT
    assert answering.FastEmbedService.calls == 1
    assert len(answering.engines) == 1
    assert answering.create_chat_model.calls == 1


def test_answer_and_retrieval_locks_are_separate_and_never_nested(
    answering: AnsweringFactories,
) -> None:
    with TestClient(app) as client:
        assert ask(client) == 200

    assert dependencies._answer_lock is not dependencies._lock
    assert answering.answer_lock_held_on_embedding == [False]
    assert answering.retrieval_lock_held_on_chat_model == [False]
    [engine] = answering.engines
    assert isinstance(engine, ObservingEngine)
    assert engine.answer_lock_held_on_dispose == [False]


def test_failed_chain_creation_caches_nothing_and_keeps_the_retrieval_service(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories = AnsweringFactories(chat_model_failures=1).install(monkeypatch)
    with TestClient(app) as client:
        first = ask(client)
        assert dependencies._answer_chain is None
        second = ask(client)

    assert (first, second) == (503, 200)
    assert factories.create_chat_model.calls == 2
    assert factories.FastEmbedService.calls == 1
    assert [engine.disposals for engine in factories.engines] == [1]


def test_shutdown_clears_the_chain_and_disposes_the_engine_once(
    answering: AnsweringFactories,
) -> None:
    with TestClient(app) as client:
        ask(client)
        assert dependencies._answer_chain is not None
        assert answering.engines[0].disposals == 0

    assert dependencies._answer_chain is None
    assert dependencies._retrieval_service is None
    assert answering.engines[0].disposals == 1


def test_next_lifespan_builds_a_fresh_chain_and_retrieval_service(
    answering: AnsweringFactories,
) -> None:
    with TestClient(app) as client:
        ask(client)
    with TestClient(app) as client:
        ask(client)
        assert [engine.disposals for engine in answering.engines] == [1, 0]

    assert answering.create_chat_model.calls == 2
    assert answering.FastEmbedService.calls == 2
    assert [engine.disposals for engine in answering.engines] == [1, 1]
    # Each lifespan searched through its own retrieval service.
    assert [repository.searches for repository in answering.repositories] == [1, 1]


def test_shutdown_disposes_the_engine_even_if_clearing_the_chain_fails(
    answering: AnsweringFactories, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_clear() -> None:
        raise RuntimeError("clear failed")

    monkeypatch.setattr("app.main.clear_answer_chain", failing_clear)

    with pytest.raises(RuntimeError, match="clear failed"):
        with TestClient(app) as client:
            ask(client)

    assert answering.engines[0].disposals == 1


@pytest.fixture
def real_chat_model_factory(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> AnsweringFactories:
    """Like `answering`, but with the real `create_chat_model`.

    The repository returns no chunks, so `/ask` answers `no_retrieval` and the
    real chain is built but never invoked; the socket guard backs that up.
    """
    factories = AnsweringFactories().install(monkeypatch)
    monkeypatch.setattr(dependencies, "create_chat_model", create_chat_model)
    return factories


def test_configuration_removed_between_lifespans_is_observed(
    real_chat_model_factory: AnsweringFactories, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.API_KEY_ENV, "sk-ant-fake-test-key")
    with TestClient(app) as client:
        first = ask(client)

    monkeypatch.delenv(config.API_KEY_ENV)
    with TestClient(app) as client:
        second = ask(client)

    assert (first, second) == (200, 503)


def test_configuration_added_between_lifespans_is_observed(
    real_chat_model_factory: AnsweringFactories, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(app) as client:
        first = ask(client)

    monkeypatch.setenv(config.API_KEY_ENV, "sk-ant-fake-test-key")
    with TestClient(app) as client:
        second = ask(client)

    assert (first, second) == (503, 200)


def test_configuration_change_within_a_lifespan_keeps_the_cached_chain(
    real_chat_model_factory: AnsweringFactories, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.API_KEY_ENV, "sk-ant-fake-test-key")
    with TestClient(app) as client:
        first = ask(client)
        monkeypatch.delenv(config.API_KEY_ENV)
        second = ask(client)

    assert (first, second) == (200, 200)


def test_clear_answer_chain_without_a_cached_chain_does_nothing(
    answering: AnsweringFactories,
) -> None:
    dependencies.clear_answer_chain()
    dependencies.clear_answer_chain()

    assert dependencies._answer_chain is None
    assert answering.create_chat_model.calls == 0
    assert answering.engines == []
