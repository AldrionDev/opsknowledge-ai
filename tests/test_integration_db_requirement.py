import pytest

from app.db.config import DATABASE_URL_ENV
from tests.integration.database_requirement import (
    REQUIRE_INTEGRATION_DB_ENV,
    TEST_DATABASE_URL_ENV,
    integration_db_is_required,
    require_admin_url,
)

ADMIN_URL = "postgresql+psycopg://tester:pw@localhost:5432/postgres"
DEVELOPER_URL = "postgresql+psycopg://tester:pw@localhost:5432/opsknowledge"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decide from the test's own values, not from the developer's `.env`."""
    for name in (
        TEST_DATABASE_URL_ENV,
        DATABASE_URL_ENV,
        REQUIRE_INTEGRATION_DB_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_configured_server_url_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEST_DATABASE_URL_ENV, ADMIN_URL)

    assert require_admin_url() == ADMIN_URL


def test_configured_server_url_is_returned_when_the_database_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEST_DATABASE_URL_ENV, ADMIN_URL)
    monkeypatch.setenv(REQUIRE_INTEGRATION_DB_ENV, "1")

    assert require_admin_url() == ADMIN_URL


def test_missing_url_skips_the_test() -> None:
    with pytest.raises(pytest.skip.Exception, match="were NOT executed"):
        require_admin_url()


@pytest.mark.parametrize("url", ["", "   "])
def test_blank_url_counts_as_missing(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv(TEST_DATABASE_URL_ENV, url)

    with pytest.raises(pytest.skip.Exception):
        require_admin_url()


def test_missing_url_fails_when_the_database_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REQUIRE_INTEGRATION_DB_ENV, "1")

    with pytest.raises(pytest.fail.Exception, match=REQUIRE_INTEGRATION_DB_ENV):
        require_admin_url()


def test_url_naming_the_developer_database_always_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEST_DATABASE_URL_ENV, DEVELOPER_URL)
    monkeypatch.setenv(DATABASE_URL_ENV, DEVELOPER_URL)

    with pytest.raises(pytest.fail.Exception, match="same database"):
        require_admin_url()


@pytest.mark.parametrize("value", ["", "   ", "0", "false", "FALSE"])
def test_disabling_values_keep_the_skip(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(REQUIRE_INTEGRATION_DB_ENV, value)

    assert integration_db_is_required() is False


@pytest.mark.parametrize("value", ["1", "true"])
def test_any_other_value_requires_the_database(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(REQUIRE_INTEGRATION_DB_ENV, value)

    assert integration_db_is_required() is True
