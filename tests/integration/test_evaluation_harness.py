"""The evaluation harness on PostgreSQL + pgvector, with fake embeddings and a fake LLM.

The repository corpus is ingested into a disposable database with
`FakeEmbeddingService(dimension=384)`. Retrieval, the chunk repository, the
read-only engine and the corpus verification are real; the embedding model and
the chat model are fakes, so no model is downloaded and no provider request can
be made.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from langchain_core.exceptions import ModelAuthenticationError
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import DBAPIError

from app.answering.models import AnswerDraft
from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.db.session import create_session_factory
from app.evaluation import __main__ as cli
from app.evaluation.dataset import CORPUS_ROOT, load_dataset
from app.evaluation.fakes import build_dry_run_script
from app.evaluation.metrics import expected_rank
from app.evaluation.provenance import (
    GitIdentity,
    compare_corpus,
    corpus_fingerprint,
    corpus_manifest,
    create_read_only_engine,
    read_database_corpus,
    read_migration_revision,
)
from app.evaluation.runner import Outcome
from app.ingestion.service import IngestionService
from app.llm.config import LLMConfig
from app.llm.preflight import ENDPOINT_OVERRIDE_VARS, TRACING_VARS
from app.retrieval.service import SemanticRetrievalService
from tests.corpus import write_corpus
from tests.evaluation.support import Harness
from tests.fake_embedding import FakeEmbeddingService
from tests.fake_llm import FakeStructuredChatModel, clear_llm_environment

pytestmark = pytest.mark.integration

KEY = "sk-ant-test-key-SENTINEL-int-4a7e"
CLEAN_GIT = GitIdentity("a" * 40, "b" * 40, "c" * 40, worktree_clean=True)
DECLINED = AnswerDraft(insufficient=True, claims=[])


def repository(engine: Engine) -> SqlAlchemyChunkRepository:
    return SqlAlchemyChunkRepository(create_session_factory(engine))


def ingest(engine: Engine, corpus_root: Path) -> None:
    embedder = FakeEmbeddingService(dimension=EMBEDDING_DIMENSION)
    IngestionService(embedder, repository(engine)).ingest(corpus_root)


@pytest.fixture
def ingested(migrated_engine: Engine) -> Engine:
    """The writable engine of a disposable database holding the repository corpus."""
    ingest(migrated_engine, CORPUS_ROOT)
    return migrated_engine


@pytest.fixture
def read_only_engine(ingested: Engine, temp_database_url: str) -> Iterator[Engine]:
    engine = create_read_only_engine(temp_database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def chunk_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(DocumentChunkRecord)
        ).scalar_one()


# --- harness over real retrieval --------------------------------------------------


def test_the_harness_runs_the_dataset_over_pgvector_retrieval(
    read_only_engine: Engine,
) -> None:
    dataset = load_dataset()
    retrieval = SemanticRetrievalService(
        FakeEmbeddingService(dimension=EMBEDDING_DIMENSION),
        repository(read_only_engine),
    )
    harness = Harness(retrieval, build_dry_run_script(dataset.questions))

    run = harness.run(dataset.questions, limit=5)

    assert run.complete
    assert all(r.retrieved is not None and len(r.retrieved) == 5 for r in run.results)
    assert all(r.outcome is not None for r in run.results)
    invoked = sum(1 for r in run.results if r.llm_invoked)
    assert harness.limiter.invocations == invoked == len(harness.fake.calls) == 24
    assert any(r.outcome is Outcome.ANSWERED for r in run.results)
    for result in run.results:
        retrieved = result.retrieved or ()
        positions = [
            position
            for position, chunk in enumerate(retrieved, start=1)
            if chunk.source_path in result.question.expected_sources
        ]
        rank = expected_rank(retrieved, result.question.expected_sources, 5)
        assert rank == (positions[0] if positions else None)


# --- corpus verification and read-only access ---------------------------------------


def alembic_head(config: Config) -> str | None:
    return ScriptDirectory.from_config(config).get_current_head()


def test_a_fully_ingested_corpus_is_verified(
    read_only_engine: Engine, alembic_config: Config
) -> None:
    expected = corpus_manifest(CORPUS_ROOT)

    stored = read_database_corpus(read_only_engine)

    assert compare_corpus(expected, stored).matches
    assert corpus_fingerprint(stored) == corpus_fingerprint(expected)
    assert read_migration_revision(read_only_engine) == alembic_head(alembic_config)


def test_stale_chunks_of_another_document_are_unexpected(
    ingested: Engine, read_only_engine: Engine, tmp_path: Path
) -> None:
    stale = write_corpus(
        tmp_path / "stale", {"runbooks/deleted-runbook.md": "# Deleted\n\nOld content."}
    )
    ingest(ingested, stale)

    diff = compare_corpus(
        corpus_manifest(CORPUS_ROOT), read_database_corpus(read_only_engine)
    )

    assert {c.source_path for c in diff.unexpected} == {"runbooks/deleted-runbook.md"}
    assert not diff.missing and not diff.changed


def test_deleted_chunks_are_missing(ingested: Engine, read_only_engine: Engine) -> None:
    expected = corpus_manifest(CORPUS_ROOT)
    victim = next(iter(expected.values()))
    repository(ingested).delete_document_chunks(victim.document_id)

    diff = compare_corpus(expected, read_database_corpus(read_only_engine))

    assert {c.source_path for c in diff.missing} == {victim.source_path}
    assert not diff.unexpected and not diff.changed


def test_edited_content_is_changed(ingested: Engine, read_only_engine: Engine) -> None:
    expected = corpus_manifest(CORPUS_ROOT)
    victim = sorted(expected)[0]
    with ingested.begin() as connection:
        connection.execute(
            text(
                "UPDATE document_chunks SET content = content || ' edited' WHERE chunk_id = :id"
            ),
            {"id": victim},
        )

    diff = compare_corpus(expected, read_database_corpus(read_only_engine))

    assert [(e.chunk_id, a.chunk_id) for e, a in diff.changed] == [(victim, victim)]
    assert not diff.missing and not diff.unexpected


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM document_chunks",
        "UPDATE document_chunks SET content = 'x'",
        "TRUNCATE document_chunks",
        "CREATE TABLE harness_write_probe (id integer)",
    ],
)
def test_the_read_only_engine_rejects_writes(
    ingested: Engine, read_only_engine: Engine, statement: str
) -> None:
    before = chunk_count(ingested)

    with pytest.raises(DBAPIError, match="read-only transaction"):
        with read_only_engine.begin() as connection:
            connection.execute(text(statement))

    assert chunk_count(ingested) == before
    with read_only_engine.connect() as connection:
        assert (
            connection.execute(text("SHOW default_transaction_read_only")).scalar_one()
            == "on"
        )


# --- the live composition root with fake models ------------------------------------------


class Captured:
    def __init__(self, model: FakeStructuredChatModel) -> None:
        self.model = model
        self.configs: list[LLMConfig] = []

    def create_chat_model(self, config: LLMConfig) -> FakeStructuredChatModel:
        self.configs.append(config)
        return self.model


@pytest.fixture
def live_wiring(
    ingested: Engine, temp_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> Captured:
    clear_llm_environment(monkeypatch)
    for name in (*ENDPOINT_OVERRIDE_VARS, *TRACING_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", temp_database_url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(cli, "git_identity", lambda root: CLEAN_GIT)
    monkeypatch.setattr(
        cli,
        "FastEmbedService",
        lambda: FakeEmbeddingService(dimension=EMBEDDING_DIMENSION),
    )
    captured = Captured(FakeStructuredChatModel(result=DECLINED))
    monkeypatch.setattr(cli, "create_chat_model", captured.create_chat_model)
    return captured


def test_the_live_composition_root_runs_on_a_verified_database(
    live_wiring: Captured,
    alembic_config: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "live"

    status = cli.main(
        [
            "--live",
            "--max-retries",
            "0",
            "--max-output-tokens",
            "256",
            "--output-dir",
            str(target),
        ]
    )

    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    [config] = live_wiring.configs
    assert status == 0
    assert (config.max_retries, config.max_output_tokens) == (0, 256)
    assert report["database"]["corpus_verified"] is True
    assert report["database"]["migration_revision"] == alembic_head(alembic_config)
    assert report["database"]["chunk_count"] == len(corpus_manifest(CORPUS_ROOT))
    assert report["usage"]["llm_invocations"] == len(live_wiring.model.prompts) == 24
    assert (
        "24 requests x (1 + 0 retries) x 256 output tokens = 6144"
        in capsys.readouterr().out
    )


def test_the_live_request_cap_on_pgvector(
    live_wiring: Captured, tmp_path: Path
) -> None:
    status = cli.main(
        ["--live", "--max-requests", "2", "--output-dir", str(tmp_path / "x")]
    )

    assert status == 1
    assert len(live_wiring.model.prompts) == 2


def test_a_nonrecoverable_provider_error_stops_the_live_run(
    live_wiring: Captured, tmp_path: Path
) -> None:
    live_wiring.model.failure = ModelAuthenticationError("scripted")

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    report = json.loads((tmp_path / "x" / "report.json").read_text(encoding="utf-8"))
    assert status == 1
    assert report["run"]["abort_reason"] == "provider_nonrecoverable"
    assert len(live_wiring.model.prompts) == 1


def test_a_stale_database_refuses_the_live_run(
    live_wiring: Captured, ingested: Engine, tmp_path: Path
) -> None:
    ingest(
        ingested,
        write_corpus(tmp_path / "stale", {"incidents/old.md": "# Old\n\nGone."}),
    )

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    assert status == 2
    assert live_wiring.configs == []
    assert not (tmp_path / "x").exists()
