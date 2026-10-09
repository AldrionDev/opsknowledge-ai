import pytest
from fastapi.testclient import TestClient

from app.db.config import DatabaseConfigError
from app.db.session import create_db_engine, create_session_factory
from app.main import app


def test_engine_is_created_without_connecting() -> None:
    # Port 1 is unreachable: any connection attempt would fail.
    engine = create_db_engine("postgresql+psycopg://user:pw@127.0.0.1:1/appdb")

    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.pool.checkedout() == 0


def test_engine_uses_database_url_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pw@127.0.0.1:1/envdb")

    assert create_db_engine().url.database == "envdb"


def test_engine_creation_fails_clearly_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigError):
        create_db_engine()


def test_session_factory_is_bound_to_the_engine() -> None:
    engine = create_db_engine("postgresql+psycopg://user:pw@127.0.0.1:1/appdb")

    factory = create_session_factory(engine)

    with factory() as session:
        assert session.get_bind() is engine


def test_health_does_not_require_database_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
