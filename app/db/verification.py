"""Runtime check of the database embedding column dimension."""

from sqlalchemy import Engine, text

from app.db.models import DocumentChunkRecord


class EmbeddingDimensionMismatchError(Exception):
    """Raised when the embedding model and the database column disagree."""


def verify_embedding_dimension(engine: Engine, expected_dimension: int) -> None:
    """Raise `EmbeddingDimensionMismatchError` unless the `embedding` column of
    the chunk table has exactly `expected_dimension` dimensions.

    Call it with the embedding service dimension before relying on persistence.
    For `vector(n)` columns pgvector stores `n` in `pg_attribute.atttypmod`.
    """
    table = DocumentChunkRecord.__tablename__
    with engine.connect() as connection:
        actual = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
            ),
            {"table": table},
        ).scalar_one_or_none()

    if actual is None:
        raise EmbeddingDimensionMismatchError(
            f"Table '{table}' has no 'embedding' column; run the Alembic migrations"
        )
    if actual != expected_dimension:
        raise EmbeddingDimensionMismatchError(
            f"Embedding model dimension is {expected_dimension}, but the "
            f"database column has {actual} dimensions"
        )
