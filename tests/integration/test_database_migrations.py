import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from app.db.models import Base

pytestmark = pytest.mark.integration


def _schema_snapshot(url: str) -> dict[str, object]:
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        return {
            "columns": [
                (c["name"], str(c["type"]), c["nullable"])
                for c in inspector.get_columns("document_chunks")
            ],
            "pk": inspector.get_pk_constraint("document_chunks")["name"],
            "unique": sorted(
                u["name"] or ""
                for u in inspector.get_unique_constraints("document_chunks")
            ),
            "checks": sorted(
                c["name"] or ""
                for c in inspector.get_check_constraints("document_chunks")
            ),
        }
    finally:
        engine.dispose()


def _alembic_versions(url: str) -> list[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(text("SELECT version_num FROM alembic_version"))
            return [row[0] for row in rows]
    finally:
        engine.dispose()


def test_migrations_apply_on_a_fresh_database(
    alembic_config: Config, temp_database_url: str
) -> None:
    command.upgrade(alembic_config, "head")

    engine = create_engine(temp_database_url)
    try:
        with engine.connect() as connection:
            extensions = connection.execute(text("SELECT extname FROM pg_extension"))
            assert "vector" in {row[0] for row in extensions}
        assert "document_chunks" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_upgrade_is_repeatable(alembic_config: Config, temp_database_url: str) -> None:
    command.upgrade(alembic_config, "head")
    first = _schema_snapshot(temp_database_url)

    command.upgrade(alembic_config, "head")

    assert _schema_snapshot(temp_database_url) == first
    assert _alembic_versions(temp_database_url) == ["0001"]


def test_downgrade_and_upgrade_reproduce_the_same_schema(
    alembic_config: Config, temp_database_url: str
) -> None:
    command.upgrade(alembic_config, "head")
    first = _schema_snapshot(temp_database_url)

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    assert _schema_snapshot(temp_database_url) == first


def test_migrated_schema_matches_the_models(migrated_engine) -> None:  # type: ignore[no-untyped-def]
    with migrated_engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)

    assert differences == []
