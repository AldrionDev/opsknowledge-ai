import ast
from pathlib import Path

from pgvector.sqlalchemy import Vector

from app.db.models import EMBEDDING_DIMENSION, DocumentChunkRecord
from app.embeddings.fastembed_service import MODEL_DIMENSION

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations" / "versions"

REQUIRED_COLUMNS = {
    "chunk_id",
    "document_id",
    "chunk_index",
    "source_path",
    "document_type",
    "content",
    "embedding",
}


def test_table_stores_chunk_data_and_embedding() -> None:
    table = DocumentChunkRecord.__table__

    assert table.name == "document_chunks"
    assert {column.name for column in table.columns} == REQUIRED_COLUMNS


def test_all_columns_are_required() -> None:
    assert all(not column.nullable for column in DocumentChunkRecord.__table__.columns)


def test_chunk_id_is_the_primary_key() -> None:
    primary_key = DocumentChunkRecord.__table__.primary_key

    assert [column.name for column in primary_key.columns] == ["chunk_id"]


def test_embedding_is_a_vector_of_the_database_dimension() -> None:
    embedding_type = DocumentChunkRecord.__table__.c.embedding.type

    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == EMBEDDING_DIMENSION


def test_database_dimension_matches_the_selected_embedding_model() -> None:
    assert EMBEDDING_DIMENSION == MODEL_DIMENSION


def test_migrations_do_not_import_application_code() -> None:
    migrations = sorted(MIGRATIONS_DIR.glob("*.py"))
    assert migrations

    for migration in migrations:
        imported: set[str] = set()
        for node in ast.walk(ast.parse(migration.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        offending = {name for name in imported if name.split(".")[0] == "app"}
        assert not offending, f"{migration.name} imports application code: {offending}"
