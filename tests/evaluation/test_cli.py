"""The evaluation CLI: mode gating, the offline dry-run and the live preflight.

Network access is blocked for every test (conftest). Live-mode tests replace the
database, the embedding model and the chat model factory with fakes; no provider
request can be made.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.exc import ArgumentError, OperationalError

from app.answering.models import AnswerDraft
from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.service import EmbeddingError
from app.evaluation import __main__ as cli
from app.evaluation.dataset import CORPUS_ROOT
from app.evaluation.provenance import CorpusChunk, GitIdentity, corpus_manifest
from app.evaluation.report import ReportWriteError
from app.evaluation.runner import AbortReason, RunResult
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from app.llm.config import LLMConfig
from app.llm.errors import LLMConfigurationError
from tests.fake_chunk_repository import InMemoryChunkRepository
from tests.fake_embedding import FakeEmbeddingService
from tests.fake_llm import FakeStructuredChatModel

KEY = "sk-ant-test-key-SENTINEL-71d3"
DB_PASSWORD = "SENTINEL-DB-PASSWORD-c2e9"
DATABASE_URL = (
    f"postgresql+psycopg://user:{DB_PASSWORD}@localhost:5432/opsknowledge_eval_test"
)
REDIRECT = "https://SENTINEL-redirect-5b1a.example"
CLEAN_GIT = GitIdentity("a" * 40, "b" * 40, "c" * 40, worktree_clean=True)
DECLINED = AnswerDraft(insufficient=True, claims=[])

REAL_FACTORIES = (
    "create_read_only_engine",
    "get_database_url",
    "verify_embedding_dimension",
    "read_database_corpus",
    "read_migration_revision",
    "FastEmbedService",
    "SqlAlchemyChunkRepository",
    "SemanticRetrievalService",
    "load_llm_config",
    "create_chat_model",
    "build_answer_chain",
)


def forbid(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:

        def forbidden(*args: object, _name: str = name, **kwargs: object) -> None:
            raise AssertionError(f"{_name} must not be called")

        monkeypatch.setattr(cli, name, forbidden)


def outputs(capsys: pytest.CaptureFixture[str], directory: Path | None = None) -> str:
    captured = capsys.readouterr()
    text = captured.out + captured.err
    if directory is not None and directory.exists():
        text += "".join(
            path.read_text(encoding="utf-8") for path in directory.iterdir()
        )
    return text


def read_report(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


# --- mode gating -------------------------------------------------------------------


def test_without_a_mode_nothing_is_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    forbid(monkeypatch, *REAL_FACTORIES, "run_evaluation", "git_identity")

    status = cli.main(["--output-dir", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert status == 2
    assert "24 questions" in captured.out
    assert "24 requests x (1 + 1 retries) x 1024 output tokens = 49152" in captured.out
    assert "not a total API cost bound" in captured.out
    assert "nothing was done" in captured.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--live", "--dry-run"],
        ["--dry-run", "--max-requests", "31"],
        ["--dry-run", "--max-requests", "0"],
        ["--dry-run", "--max-retries", "3"],
        ["--dry-run", "--max-output-tokens", "2049"],
        ["--dry-run", "--max-output-tokens", "many"],
        ["--dry-run", "--limit", "51"],
        ["--dry"],
    ],
)
def test_invalid_arguments_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)

    assert caught.value.code == 2


# --- dry-run --------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["sentinels", "empty"])
def test_the_dry_run_is_fully_offline_and_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    environment: str,
) -> None:
    if environment == "sentinels":
        monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
        monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", REDIRECT)
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    forbid(monkeypatch, *REAL_FACTORIES)
    target = tmp_path / "dry"

    status = cli.main(["--dry-run", "--output-dir", str(target)])

    text = outputs(capsys, target)
    report = read_report(target)
    assert status == 0
    assert report["run"]["mode"] == "dry_run"
    assert report["run"]["synthetic"] is True
    assert report["run"]["complete"] is True
    assert report["database"] == {"used": False}
    assert report["retrieval"]["implementation"] == "synthetic_hash"
    assert report["llm"]["implementation"] == "scripted_fake"
    assert report["llm"]["sampling_parameters"] == "none"
    assert report["usage"]["input_tokens"] == "unavailable"
    assert {q["outcome"] for q in report["questions"]} == {
        "answered",
        "insufficient_context",
        "invalid_output",
        "provider_error",
    }
    assert "NOT A BASELINE" in (target / "report.md").read_text(encoding="utf-8")
    assert "NOT A BASELINE" in text
    for sentinel in (KEY, DB_PASSWORD, REDIRECT):
        assert sentinel not in text


def test_the_dry_run_is_reproducible_apart_from_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forbid(monkeypatch, *REAL_FACTORIES)
    for name in ("one", "two"):
        assert cli.main(["--dry-run", "--output-dir", str(tmp_path / name)]) == 0

    first, second = read_report(tmp_path / "one"), read_report(tmp_path / "two")
    for report in (first, second):
        del report["run"]["started_at"], report["run"]["finished_at"]
    assert first == second


def test_an_existing_output_directory_refuses_before_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forbid(monkeypatch, *REAL_FACTORIES, "run_evaluation")
    (tmp_path / "report.json").write_text("previous", encoding="utf-8")

    assert cli.main(["--dry-run", "--output-dir", str(tmp_path)]) == 2
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == "previous"


def test_a_failed_report_write_prints_the_json_instead(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def failing_write(report: Any, output_dir: Path) -> None:
        raise ReportWriteError(f"cannot write the report to {output_dir}: OSError")

    monkeypatch.setattr(cli, "write_report", failing_write)

    status = cli.main(["--dry-run", "--output-dir", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert status == 1
    assert "cannot write the report" in captured.err
    dumped = json.loads(captured.out[captured.out.index("{") :])
    assert dumped["persisted"] is False
    assert dumped["report"]["schema_version"] == 1


def test_an_interrupted_run_exits_130_after_writing_the_partial_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *args: RunResult(
            results=(), llm_invocations=0, abort_reason=AbortReason.INTERRUPTED
        ),
    )

    assert cli.main(["--dry-run", "--output-dir", str(tmp_path / "out")]) == 130
    assert read_report(tmp_path / "out")["run"]["abort_reason"] == "interrupted"


def test_dotenv_files_are_never_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        f"DATABASE_URL={DATABASE_URL}\nANTHROPIC_API_KEY={KEY}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli, "git_identity", lambda root: CLEAN_GIT)
    forbid(
        monkeypatch, "create_read_only_engine", "load_llm_config", "create_chat_model"
    )

    live_status = cli.main(["--live", "--output-dir", str(tmp_path / "live")])
    dry_status = cli.main(["--dry-run", "--output-dir", str(tmp_path / "dry")])

    text = outputs(capsys, tmp_path / "dry")
    assert live_status == 2
    assert "DATABASE_URL is not set" in text
    assert dry_status == 0
    assert KEY not in text and DB_PASSWORD not in text


# --- live preflight ----------------------------------------------------------------------


@dataclass
class LiveFakes:
    """What the live branch built while a test ran, with fakes at the boundaries."""

    configs: list[LLMConfig] = field(default_factory=list)
    model: FakeStructuredChatModel = field(
        default_factory=lambda: FakeStructuredChatModel(result=DECLINED)
    )
    disposed: int = 0
    stored: dict[str, CorpusChunk] = field(
        default_factory=lambda: corpus_manifest(CORPUS_ROOT)
    )


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> LiveFakes:
    fakes = LiveFakes()
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(cli, "git_identity", lambda root: CLEAN_GIT)

    class Engine:
        def dispose(self) -> None:
            fakes.disposed += 1

    def repository(session_factory: object) -> InMemoryChunkRepository:
        store = InMemoryChunkRepository(dimension=EMBEDDING_DIMENSION)
        embedder = FakeEmbeddingService(dimension=EMBEDDING_DIMENSION)
        for document in load_knowledge_documents(CORPUS_ROOT):
            chunks = chunk_document(document)
            store.add_chunks(
                chunks, embedder.embed_documents([c.content for c in chunks])
            )
        return store

    def create_chat_model(config: LLMConfig) -> FakeStructuredChatModel:
        fakes.configs.append(config)
        return fakes.model

    monkeypatch.setattr(cli, "create_read_only_engine", lambda url: Engine())
    monkeypatch.setattr(cli, "verify_embedding_dimension", lambda engine, dim: None)
    monkeypatch.setattr(cli, "read_database_corpus", lambda engine: fakes.stored)
    monkeypatch.setattr(cli, "read_migration_revision", lambda engine: "0001")
    monkeypatch.setattr(
        cli, "FastEmbedService", lambda: FakeEmbeddingService(EMBEDDING_DIMENSION)
    )
    monkeypatch.setattr(cli, "SqlAlchemyChunkRepository", repository)
    monkeypatch.setattr(cli, "create_chat_model", create_chat_model)
    return fakes


def test_the_live_branch_passes_the_cli_bounds_to_the_model(
    live: LiveFakes, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "live"

    status = cli.main(
        [
            "--live",
            "--max-requests", "24",
            "--max-retries", "0",
            "--max-output-tokens", "512",
            "--output-dir", str(target),
        ]
    )  # fmt: skip

    text = outputs(capsys, target)
    report = read_report(target)
    [config] = live.configs
    assert status == 0
    assert (config.max_retries, config.max_output_tokens) == (0, 512)
    assert "24 requests x (1 + 0 retries) x 512 output tokens = 12288" in text
    assert report["run"]["synthetic"] is False
    assert report["llm"]["max_retries"] == 0
    assert report["llm"]["max_output_tokens"] == 512
    assert report["llm"]["sampling_parameters"] == "none"
    assert report["llm"]["structured_output_method"] == "json_schema"
    assert report["database"]["corpus_verified"] is True
    assert report["database"]["name"] == "opsknowledge_eval_test"
    assert report["database"]["migration_revision"] == "0001"
    assert report["database"]["read_only_transactions"] is True
    assert report["database"]["embedding_provenance_verified"] is False
    assert report["usage"]["max_possible_transport_attempts"] == (
        report["usage"]["llm_invocations"] * 1
    )
    assert report["usage"]["llm_invocations"] == len(live.model.prompts) <= 24
    assert live.disposed == 1
    assert KEY not in text and DB_PASSWORD not in text


def test_the_live_request_cap_stops_the_run(live: LiveFakes, tmp_path: Path) -> None:
    status = cli.main(
        ["--live", "--max-requests", "2", "--output-dir", str(tmp_path / "x")]
    )

    report = read_report(tmp_path / "x")
    assert status == 1
    assert report["run"]["abort_reason"] == "max_requests"
    assert len(live.model.prompts) == 2


def test_unsafe_environment_refuses_before_anything_else(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", REDIRECT)
    forbid(monkeypatch, "git_identity", "get_database_url", "load_llm_config")

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    text = outputs(capsys)
    assert status == 2
    assert "ANTHROPIC_BASE_URL" in text
    assert REDIRECT not in text


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (None, "Git identity of the repository is unavailable"),
        (GitIdentity("a", "b", "c", worktree_clean=False), "worktree is not clean"),
    ],
)
def test_live_requires_a_clean_git_worktree(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    identity: GitIdentity | None,
    message: str,
) -> None:
    monkeypatch.setattr(cli, "git_identity", lambda root: identity)
    forbid(
        monkeypatch, "get_database_url", "create_read_only_engine", "load_llm_config"
    )

    assert cli.main(["--live", "--output-dir", str(tmp_path / "x")]) == 2
    assert message in capsys.readouterr().err


def test_a_corpus_mismatch_refuses_before_the_key_is_read(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    stored = dict(live.stored)
    removed = stored.pop(sorted(stored)[0])
    stale = CorpusChunk(
        "stale", "doc-stale", 0, "runbooks/deleted.md", "runbook", "0" * 64
    )
    stored["stale"] = stale
    changed_id = sorted(stored)[1]
    stored[changed_id] = CorpusChunk(
        **(vars(stored[changed_id]) | {"content_sha256": "f" * 64})
    )
    live.stored = stored
    forbid(monkeypatch, "load_llm_config", "FastEmbedService", "create_chat_model")

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    err = capsys.readouterr().err
    assert status == 2
    assert f"corpus missing chunks: 1 in {removed.source_path}" in err
    assert "corpus unexpected chunks: 1 in runbooks/deleted.md" in err
    assert "corpus changed chunks: 1 in" in err
    assert live.disposed == 1
    assert not (tmp_path / "x").exists()


def test_a_missing_key_refuses_before_any_model_is_built(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    forbid(monkeypatch, "FastEmbedService", "create_chat_model")

    assert cli.main(["--live", "--output-dir", str(tmp_path / "x")]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert live.disposed == 1


def test_database_errors_are_reported_by_type_only(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def unreachable(engine: object, dimension: int) -> None:
        raise OperationalError(
            f"connect to {DATABASE_URL}", None, Exception(DB_PASSWORD)
        )

    monkeypatch.setattr(cli, "verify_embedding_dimension", unreachable)
    forbid(monkeypatch, "load_llm_config", "create_chat_model")

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    err = capsys.readouterr().err
    assert status == 2
    assert "database preflight failed (OperationalError)" in err
    assert DB_PASSWORD not in err
    assert live.disposed == 1


# --- database configuration and engine construction ------------------------------


MODEL_FACTORIES = ("load_llm_config", "FastEmbedService", "create_chat_model")


def test_a_malformed_database_url_refuses_safely(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    malformed = f"postgresql+psycopg://user:{DB_PASSWORD}@localhost:notaport/eval"
    monkeypatch.setenv("DATABASE_URL", malformed)
    forbid(monkeypatch, "create_read_only_engine", *MODEL_FACTORIES)

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    text = outputs(capsys)
    assert status == 2
    assert "error: DATABASE_URL is not a valid database URL (ValueError)" in text
    assert "Traceback" not in text
    assert DB_PASSWORD not in text and "notaport" not in text
    assert live.configs == [] and live.model.prompts == []
    assert live.disposed == 0


@pytest.mark.parametrize(
    "failure",
    [
        ArgumentError(f"Could not parse SQLAlchemy URL from string '{DATABASE_URL}'"),
        ValueError(f"invalid literal for int() in {DATABASE_URL}"),
    ],
)
def test_a_failed_engine_construction_refuses_safely(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    failure: Exception,
) -> None:
    def failing_engine(url: str) -> None:
        raise failure

    monkeypatch.setattr(cli, "create_read_only_engine", failing_engine)
    forbid(monkeypatch, "verify_embedding_dimension", *MODEL_FACTORIES)

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    text = outputs(capsys)
    assert status == 2
    assert f"cannot create the database engine ({type(failure).__name__})" in text
    assert "Traceback" not in text
    assert DB_PASSWORD not in text and "Could not parse" not in text
    assert live.configs == [] and live.model.prompts == []
    assert live.disposed == 0


def _raise(failure: Exception) -> Any:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    return fail


@pytest.mark.parametrize(
    ("factory", "failure"),
    [
        (
            "verify_embedding_dimension",
            OperationalError("SELECT", None, Exception(DB_PASSWORD)),
        ),
        ("read_database_corpus", OperationalError("SELECT", None, Exception("x"))),
        ("read_migration_revision", OperationalError("SELECT", None, Exception("x"))),
        ("load_llm_config", LLMConfigurationError("ANTHROPIC_API_KEY is not set")),
        ("FastEmbedService", EmbeddingError("Cannot load embedding model")),
        ("create_chat_model", LLMConfigurationError("LLM_MODEL must be valid")),
    ],
)
def test_a_failed_preflight_disposes_the_engine_exactly_once(
    live: LiveFakes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    factory: str,
    failure: Exception,
) -> None:
    monkeypatch.setattr(cli, factory, _raise(failure))

    status = cli.main(["--live", "--output-dir", str(tmp_path / "x")])

    text = outputs(capsys)
    assert status == 2
    assert live.disposed == 1
    assert live.model.prompts == []
    assert DB_PASSWORD not in text and "Traceback" not in text
    assert not (tmp_path / "x").exists()
