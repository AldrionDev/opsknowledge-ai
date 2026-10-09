import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from app.db.session import create_db_engine
from tests.integration.temp_database import (
    create_database,
    drop_database,
    new_test_database_name,
)

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


@pytest.fixture
def admin_url() -> str:
    """Server URL used only to create and drop disposable databases."""
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip(
            "TEST_DATABASE_URL is not set; PostgreSQL integration tests were NOT executed"
        )

    developer_url = os.environ.get("DATABASE_URL", "").strip()
    if developer_url:
        if make_url(url).database == make_url(developer_url).database:
            pytest.fail(
                "TEST_DATABASE_URL must not point to the same database as DATABASE_URL",
                pytrace=False,
            )
    return url


@pytest.fixture
def temp_database_url(admin_url: str) -> Iterator[str]:
    """URL of a new, empty, uniquely named database, dropped after the test."""
    name = new_test_database_name()
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        create_database(admin_engine, name)
        try:
            url = make_url(admin_url).set(database=name)
            yield url.render_as_string(hide_password=False)
        finally:
            drop_database(admin_engine, name)
    finally:
        admin_engine.dispose()


@pytest.fixture
def alembic_config(temp_database_url: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Alembic config whose `DATABASE_URL` points to the disposable database."""
    monkeypatch.setenv("DATABASE_URL", temp_database_url)
    return Config(str(ALEMBIC_INI))


@pytest.fixture
def migrated_engine(alembic_config: Config, temp_database_url: str) -> Iterator[Engine]:
    command.upgrade(alembic_config, "head")
    engine = create_db_engine(temp_database_url)
    try:
        yield engine
    finally:
        engine.dispose()
