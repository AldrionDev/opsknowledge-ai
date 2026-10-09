import pytest

from app.db.config import DatabaseConfigError, get_database_url

VALID_URL = "postgresql+psycopg://user:s3cret-pw@localhost:5432/appdb"


def test_returns_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_URL)

    assert get_database_url() == VALID_URL


def test_missing_url_fails_with_setup_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigError, match=r"DATABASE_URL is not set"):
        get_database_url()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_url_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DATABASE_URL", value)

    with pytest.raises(DatabaseConfigError, match=r"DATABASE_URL is not set"):
        get_database_url()


def test_unparsable_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "not a url")

    with pytest.raises(DatabaseConfigError, match=r"not a valid database URL"):
        get_database_url()


@pytest.mark.parametrize(
    "value",
    [
        "postgresql://user:s3cret-pw@localhost/appdb",
        "postgresql+psycopg2://user:s3cret-pw@localhost/appdb",
        "mysql+pymysql://user:s3cret-pw@localhost/appdb",
    ],
)
def test_other_drivers_are_rejected_without_leaking_the_password(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", value)

    with pytest.raises(DatabaseConfigError, match=r"postgresql\+psycopg") as exc_info:
        get_database_url()

    assert "s3cret-pw" not in str(exc_info.value)
