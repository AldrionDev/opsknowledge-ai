"""Local entry point that ingests the knowledge corpus into the database.

Run it from the repository root, with the database from `DATABASE_URL`:

    uv run --env-file .env python -m app.ingestion
    uv run --env-file .env python -m app.ingestion --corpus-root path/to/corpus

By default the repository corpus in `data/knowledge` is ingested. The first run
downloads the embedding model. Exits with status 1 and an `error:` message on
stderr if the ingestion cannot be completed.

This module is the composition root: it is the only place that wires the
ingestion service to the PostgreSQL repository and the FastEmbed model.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.db.chunk_repository import SqlAlchemyChunkRepository
from app.db.config import DatabaseConfigError
from app.db.session import create_db_engine, create_session_factory
from app.db.verification import (
    EmbeddingDimensionMismatchError,
    verify_embedding_dimension,
)
from app.embeddings.fastembed_service import FastEmbedService
from app.embeddings.service import EmbeddingError
from app.ingestion.loader import DocumentLoadError
from app.ingestion.service import IngestionError, IngestionService

DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge"

_EXPECTED_ERRORS = (
    DatabaseConfigError,
    DocumentLoadError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    IngestionError,
    SQLAlchemyError,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    engine = None
    try:
        engine = create_db_engine()
        repository = SqlAlchemyChunkRepository(create_session_factory(engine))
        embedding_service = FastEmbedService()
        verify_embedding_dimension(engine, embedding_service.dimension)

        print(f"Ingesting corpus: {args.corpus_root}")
        summary = IngestionService(embedding_service, repository).ingest(
            args.corpus_root
        )
    except _EXPECTED_ERRORS as exc:
        _report_error(exc)
        return 1
    finally:
        if engine is not None:
            engine.dispose()

    print(
        f"Ingestion completed: documents={summary.documents_processed} "
        f"chunks_generated={summary.chunks_generated} "
        f"chunks_persisted={summary.chunks_persisted}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion",
        description="Ingest the knowledge corpus into the configured database.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="corpus directory containing runbooks/ and incidents/ "
        "(default: data/knowledge of this repository)",
    )
    return parser.parse_args(argv)


def _report_error(exc: Exception) -> None:
    print(f"error: {_first_line(exc)}", file=sys.stderr)
    for note in getattr(exc, "__notes__", []):
        print(f"  {note}", file=sys.stderr)
    if exc.__cause__ is not None:
        print(
            f"  caused by: {type(exc.__cause__).__name__}: "
            f"{_first_line(exc.__cause__)}",
            file=sys.stderr,
        )


def _first_line(exc: BaseException) -> str:
    lines = str(exc).splitlines()
    return lines[0] if lines else type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
