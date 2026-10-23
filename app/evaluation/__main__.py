"""Offline evaluation harness: run the evaluation dataset through `AnswerService`.

    uv run python -m app.evaluation             # describe only: prints the bounds, runs nothing
    uv run python -m app.evaluation --dry-run   # synthetic and offline, NOT A BASELINE
    uv run python -m app.evaluation --live --max-requests 24 --max-retries 1 --max-output-tokens 1024

Modes
-----

- **No mode flag:** loads the dataset, validates the bounds, prints the dataset
  size and the worst-case output-token bound, and exits with status 2. Nothing
  else is read, built or contacted.
- **`--dry-run`:** fully offline. Synthetic hash-ranked retrieval over the
  repository corpus files and a scripted fake LLM (`app.evaluation.fakes`). It
  reads no environment variable and needs no database, no embedding model, no
  credentials and no network. The report is marked SYNTHETIC / NOT A BASELINE.
- **`--live`:** the real stack (PostgreSQL + pgvector, FastEmbed, Anthropic). This
  is the only branch that builds real components, and only after the preflight:

  1. no endpoint-redirection or tracing variable is set (`app.llm.preflight`);
  2. Git identity is available and the worktree is clean;
  3. `DATABASE_URL` is set; the engine opens read-only transactions only;
  4. the embedding column dimension matches the model;
  5. the stored chunks equal the repository corpus chunk by chunk;
  6. the LLM configuration is valid (`ANTHROPIC_API_KEY`, `LLM_*`), with the
     transport retries and output tokens replaced by the CLI bounds.

  Any failure exits with status 2 before a model request is possible. No
  environment variable can enable live mode: only the flag does.

Configuration comes from the process environment only; this module never reads
or writes `.env`.

Bounds (absolute ceilings in `app.evaluation.guards`): `--max-requests`
(default: the dataset size), `--max-retries` (default 1), `--max-output-tokens`
(default 1024), and `--limit` (retrieval `k`, default as `POST /ask`). Every
question costs at most one LLM invocation, with no application-level retry;
questions without retrieval or context cost none.

Exit status: 0 run completed; 1 run stopped early (partial report written) or the
report could not be written (the unsaved report is then printed to stdout inside
`{"persisted": false, "report": ...}`); 2 nothing was run; 130 interrupted
(partial report written).

Diagnostics never contain credentials, the database URL, raw provider messages
or other exception messages that could carry them; unexpected errors are
reported by exception type only.
"""

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from app.answering.prompt import build_answer_chain
from app.answering.service import AnswerChain, AnswerService, RetrievalService
from app.api.search import DEFAULT_RESULT_LIMIT
from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.config import DatabaseConfigError, get_database_url
from app.db.session import create_session_factory
from app.db.verification import (
    EmbeddingDimensionMismatchError,
    verify_embedding_dimension,
)
from app.embeddings.fastembed_service import (
    MODEL_DIMENSION,
    MODEL_NAME,
    FastEmbedService,
)
from app.embeddings.service import EmbeddingError
from app.evaluation.dataset import (
    CORPUS_ROOT,
    REPO_ROOT,
    Dataset,
    DatasetError,
    load_dataset,
)
from app.evaluation.fakes import (
    SCRIPTED_FAKE_MODEL,
    SYNTHETIC_RETRIEVAL,
    ScriptedFakeChain,
    SyntheticRetrievalService,
    build_dry_run_script,
)
from app.evaluation.guards import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    MAX_OUTPUT_TOKENS_CEILING,
    MAX_REQUESTS_CEILING,
    MAX_RETRIES_CEILING,
    WORST_CASE_NOTE,
    BoundsError,
    CostBounds,
    RecordingRetrievalService,
    RequestLimiter,
    UsageRecorder,
)
from app.evaluation.provenance import (
    CorpusChunk,
    CorpusDiff,
    GitIdentity,
    compare_corpus,
    corpus_fingerprint,
    corpus_manifest,
    create_read_only_engine,
    document_count,
    git_identity,
    package_versions,
    read_database_corpus,
    read_migration_revision,
)
from app.evaluation.report import (
    NOT_APPLICABLE_FAKE,
    SYNTHETIC_BANNER,
    ReportInputs,
    ReportWriteError,
    build_report,
    require_new_output_dir,
    to_json,
    write_report,
)
from app.evaluation.runner import AbortReason, run_evaluation
from app.ingestion.loader import DocumentLoadError
from app.llm.config import LLMConfig, load_llm_config
from app.llm.errors import LLMConfigurationError
from app.llm.factory import create_chat_model
from app.llm.preflight import find_unsafe_environment
from app.retrieval.service import MAX_RESULT_LIMIT, SemanticRetrievalService

EXIT_COMPLETED = 0
EXIT_ABORTED = 1
EXIT_NOT_STARTED = 2
EXIT_INTERRUPTED = 130

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation-output"
MAX_LISTED_DIFFERENCES = 10

DATABASE_PREPARATION_NOTE = (
    "Expected: a separately provisioned, empty evaluation database, migrated with "
    "Alembic and ingested from the recorded corpus tree with `python -m "
    "app.ingestion` (see docs/evaluation.md). The harness verified the stored chunks "
    "against the repository corpus and accessed the database read-only."
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dataset = load_dataset()
        bounds = CostBounds(
            max_requests=(
                len(dataset.questions)
                if args.max_requests is None
                else args.max_requests
            ),
            max_retries=args.max_retries,
            max_output_tokens=args.max_output_tokens,
        )
    except (DatasetError, BoundsError) as exc:
        return _refuse(str(exc))

    mode = "live" if args.live else "dry_run" if args.dry_run else None
    _print_plan(dataset, bounds, args.limit, mode)
    if mode is None:
        return _refuse(
            "nothing was done: pass --dry-run for a synthetic offline run or --live "
            "for a real, bounded run"
        )

    output_dir = args.output_dir or _default_output_dir(mode)
    try:
        require_new_output_dir(output_dir)
    except ReportWriteError as exc:
        return _refuse(str(exc))

    if args.live:
        return _run_live(dataset, bounds, args.limit, output_dir)
    return _run_dry(dataset, bounds, args.limit, output_dir)


def _run_dry(dataset: Dataset, bounds: CostBounds, limit: int, output_dir: Path) -> int:
    """Synthetic and offline; builds fakes only and reads no environment variable."""
    try:
        corpus = corpus_manifest(CORPUS_ROOT)
        retrieval = SyntheticRetrievalService.from_corpus(CORPUS_ROOT)
        chain = ScriptedFakeChain(build_dry_run_script(dataset.questions)).as_runnable()
    except (DocumentLoadError, ValueError) as exc:
        return _refuse(f"cannot prepare the synthetic run ({type(exc).__name__})")

    versions = package_versions()
    return _execute(
        mode="dry_run",
        synthetic=True,
        dataset=dataset,
        bounds=bounds,
        limit=limit,
        output_dir=output_dir,
        retrieval=retrieval,
        chain=chain,
        git=git_identity(REPO_ROOT),
        corpus=corpus,
        database={"used": False},
        retrieval_info={
            "implementation": SYNTHETIC_RETRIEVAL,
            "embedding_model": "not used: synthetic SHA-256 scores",
            "embedding_dimension": "not used",
            "fastembed_version": "not used",
        },
        llm_info={
            "implementation": SCRIPTED_FAKE_MODEL,
            "model": SCRIPTED_FAKE_MODEL,
            "structured_output_method": NOT_APPLICABLE_FAKE,
            "sampling_parameters": "none",
            "max_output_tokens": NOT_APPLICABLE_FAKE,
            "max_retries": NOT_APPLICABLE_FAKE,
            "timeout_seconds": NOT_APPLICABLE_FAKE,
            **_llm_versions(versions),
        },
    )


def _run_live(
    dataset: Dataset, bounds: CostBounds, limit: int, output_dir: Path
) -> int:
    unsafe = find_unsafe_environment(os.environ)
    if unsafe:
        return _refuse(
            f"these environment variables are set: {', '.join(unsafe)} (values not "
            "shown); unset them in your shell"
        )
    git = git_identity(REPO_ROOT)
    if git is None:
        return _refuse("the Git identity of the repository is unavailable")
    if not git.worktree_clean:
        return _refuse("the worktree is not clean; commit or remove changes first")
    try:
        database_url = get_database_url()
    except DatabaseConfigError as exc:
        # Fixed messages that never contain the URL.
        return _refuse(str(exc))
    except ValueError as exc:
        # SQLAlchemy URL parsing, e.g. a non-numeric port. The message can quote
        # parts of the URL, so only the exception type is shown.
        return _refuse(
            f"DATABASE_URL is not a valid database URL ({type(exc).__name__})"
        )

    try:
        engine = create_read_only_engine(database_url)
    except (SQLAlchemyError, ValueError) as exc:
        return _refuse(f"cannot create the database engine ({type(exc).__name__})")
    # From here on the engine exists, so it is disposed on every path.
    try:
        return _prepare_and_run_live(
            engine, database_url, dataset, bounds, limit, output_dir, git
        )
    finally:
        engine.dispose()


def _prepare_and_run_live(
    engine: Engine,
    database_url: str,
    dataset: Dataset,
    bounds: CostBounds,
    limit: int,
    output_dir: Path,
    git: GitIdentity,
) -> int:
    try:
        verify_embedding_dimension(engine, MODEL_DIMENSION)
        corpus = corpus_manifest(CORPUS_ROOT)
        stored = read_database_corpus(engine)
        difference = compare_corpus(corpus, stored)
        if not difference.matches:
            _print_difference(difference)
            return _refuse(
                "the database does not contain exactly the repository corpus; use a "
                "clean evaluation database (docs/evaluation.md)"
            )
        revision = read_migration_revision(engine)
        config = replace(
            load_llm_config(),
            max_retries=bounds.max_retries,
            max_output_tokens=bounds.max_output_tokens,
        )
        embedding_service = FastEmbedService()
        retrieval = SemanticRetrievalService(
            embedding_service, SqlAlchemyChunkRepository(create_session_factory(engine))
        )
        chain = build_answer_chain(create_chat_model(config))
    except SQLAlchemyError as exc:
        return _refuse(f"database preflight failed ({type(exc).__name__})")
    except (
        EmbeddingDimensionMismatchError,
        EmbeddingError,
        LLMConfigurationError,
        DocumentLoadError,
    ) as exc:
        # These messages are fixed texts that carry no credentials or URLs.
        return _refuse(str(exc))

    print(f"Preflight passed; starting the live run with model {config.model}.")
    versions = package_versions()
    return _execute(
        mode="live",
        synthetic=False,
        dataset=dataset,
        bounds=bounds,
        limit=limit,
        output_dir=output_dir,
        retrieval=retrieval,
        chain=chain,
        git=git,
        corpus=corpus,
        database={
            "used": True,
            "name": _database_name(database_url),
            "migration_revision": revision or "unavailable",
            "corpus_verified": True,
            "chunk_count": len(stored),
            "document_count": document_count(stored),
            "corpus_fingerprint": corpus_fingerprint(stored),
            "read_only_transactions": True,
            "embedding_provenance_verified": False,
            "preparation_note": DATABASE_PREPARATION_NOTE,
        },
        retrieval_info={
            "implementation": "semantic_pgvector",
            "embedding_model": MODEL_NAME,
            "embedding_dimension": MODEL_DIMENSION,
            "fastembed_version": versions["fastembed"],
        },
        llm_info=_live_llm_info(config, versions),
    )


def _execute(
    *,
    mode: str,
    synthetic: bool,
    dataset: Dataset,
    bounds: CostBounds,
    limit: int,
    output_dir: Path,
    retrieval: RetrievalService,
    chain: AnswerChain,
    git: GitIdentity | None,
    corpus: Mapping[str, CorpusChunk],
    database: dict[str, Any],
    retrieval_info: dict[str, Any],
    llm_info: dict[str, Any],
) -> int:
    recorder = RecordingRetrievalService(retrieval)
    usage = UsageRecorder()
    limiter = RequestLimiter(chain.with_config(callbacks=[usage]), bounds.max_requests)
    service = AnswerService(recorder, limiter.as_runnable())

    started_at = datetime.now(UTC)
    run = run_evaluation(dataset.questions, service, recorder, limiter, usage, limit)
    finished_at = datetime.now(UTC)

    report = build_report(
        ReportInputs(
            mode=mode,
            synthetic=synthetic,
            started_at=started_at,
            finished_at=finished_at,
            dataset=dataset,
            bounds=bounds,
            limit=limit,
            git=git,
            corpus=corpus,
            database=database,
            retrieval=retrieval_info,
            llm=llm_info,
            run=run,
        )
    )
    try:
        write_report(report, output_dir)
    except ReportWriteError as exc:
        print(
            f"error: {exc}. The unsaved report follows on stdout, marked "
            '"persisted": false, so the results are not lost.',
            file=sys.stderr,
        )
        print(to_json({"persisted": False, "report": report}), end="")
        return EXIT_ABORTED

    _print_summary(report, output_dir)
    if run.abort_reason is AbortReason.INTERRUPTED:
        return EXIT_INTERRUPTED
    return EXIT_COMPLETED if run.complete else EXIT_ABORTED


def _live_llm_info(config: LLMConfig, versions: Mapping[str, str]) -> dict[str, Any]:
    return {
        "implementation": "anthropic",
        "model": config.model,
        "structured_output_method": "json_schema",
        "sampling_parameters": "none",
        "max_output_tokens": config.max_output_tokens,
        "max_retries": config.max_retries,
        "timeout_seconds": config.timeout_seconds,
        **_llm_versions(versions),
    }


def _llm_versions(versions: Mapping[str, str]) -> dict[str, str]:
    return {
        "langchain_core_version": versions["langchain-core"],
        "langchain_anthropic_version": versions["langchain-anthropic"],
        "anthropic_version": versions["anthropic"],
    }


def _database_name(database_url: str) -> str:
    return make_url(database_url).database or "unavailable"


def _print_plan(
    dataset: Dataset, bounds: CostBounds, limit: int, mode: str | None
) -> None:
    print(
        f"Evaluation dataset: {dataset.path} ({len(dataset.questions)} questions, "
        f"{dataset.version})"
    )
    print(f"Retrieval limit (k): {limit}")
    print(
        f"Bounds: max_requests={bounds.max_requests} (ceiling {MAX_REQUESTS_CEILING}), "
        f"max_retries={bounds.max_retries} (ceiling {MAX_RETRIES_CEILING}), "
        f"max_output_tokens={bounds.max_output_tokens} "
        f"(ceiling {MAX_OUTPUT_TOKENS_CEILING})"
    )
    print(
        f"Worst-case output tokens: {bounds.max_requests} requests x "
        f"(1 + {bounds.max_retries} retries) x {bounds.max_output_tokens} output tokens "
        f"= {bounds.worst_case_output_tokens}"
    )
    print(f"  {WORST_CASE_NOTE}")
    if mode == "dry_run":
        print(
            "Mode: --dry-run (synthetic, offline): no database, embedding model, "
            "credentials or network are used, and retries and output tokens do not "
            "apply to the scripted fake."
        )
    elif mode == "live":
        print(
            "Mode: --live: real Anthropic requests within these bounds, after the "
            "preflight."
        )


def _print_difference(difference: CorpusDiff) -> None:
    def paths(chunks: Sequence[CorpusChunk]) -> str:
        listed = sorted({chunk.source_path for chunk in chunks})
        shown = ", ".join(listed[:MAX_LISTED_DIFFERENCES])
        more = len(listed) - MAX_LISTED_DIFFERENCES
        return shown + (f" (+{more} more)" if more > 0 else "")

    changed = [expected for expected, _ in difference.changed]
    for label, chunks in (
        ("missing", difference.missing),
        ("unexpected", difference.unexpected),
        ("changed", changed),
    ):
        if chunks:
            print(
                f"corpus {label} chunks: {len(chunks)} in {paths(chunks)}",
                file=sys.stderr,
            )


def _print_summary(report: Mapping[str, Any], output_dir: Path) -> None:
    run = report["run"]
    if run["synthetic"]:
        print(SYNTHETIC_BANNER)
    state = "complete" if run["complete"] else f"stopped early: {run['abort_reason']}"
    if run["abort_detail"]:
        state += f" ({run['abort_detail']})"
    print(
        f"Run {state}; evaluated {run['evaluated_questions']}, not evaluated "
        f"{run['not_evaluated_questions']}."
    )
    usage = report["usage"]
    print(
        f"LLM invocations: {usage['llm_invocations']}; reported tokens: input "
        f"{usage['input_tokens']}, output {usage['output_tokens']}; invocations with "
        f"unknown usage: {usage['invocations_with_unknown_usage']}."
    )
    print(f"Report written to {output_dir}")


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_NOT_STARTED


def _default_output_dir(mode: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{mode}"


def _bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}, got {number}"
            )
        return number

    return parse


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description=(
            "Evaluate the answer service on the evaluation dataset. Without a mode "
            "flag nothing is run."
        ),
        allow_abbrev=False,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic, fully offline run with fake retrieval and a scripted LLM",
    )
    modes.add_argument(
        "--live",
        action="store_true",
        help="real run against the database and the Anthropic API, within the bounds",
    )
    parser.add_argument(
        "--max-requests",
        type=_bounded_int(1, MAX_REQUESTS_CEILING),
        default=None,
        help=f"cap on LLM requests (default: dataset size; at most {MAX_REQUESTS_CEILING})",
    )
    parser.add_argument(
        "--max-retries",
        type=_bounded_int(0, MAX_RETRIES_CEILING),
        default=DEFAULT_MAX_RETRIES,
        help=f"transport retries per request (default {DEFAULT_MAX_RETRIES}; at most "
        f"{MAX_RETRIES_CEILING})",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_bounded_int(1, MAX_OUTPUT_TOKENS_CEILING),
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"output tokens per attempt (default {DEFAULT_MAX_OUTPUT_TOKENS}; at most "
        f"{MAX_OUTPUT_TOKENS_CEILING})",
    )
    parser.add_argument(
        "--limit",
        type=_bounded_int(1, MAX_RESULT_LIMIT),
        default=DEFAULT_RESULT_LIMIT,
        help=f"retrieved chunks per question, the k of hit@k (default {DEFAULT_RESULT_LIMIT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new directory for report.json and report.md (default: "
        "evaluation-output/<UTC time>-<mode>/); must not exist",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
