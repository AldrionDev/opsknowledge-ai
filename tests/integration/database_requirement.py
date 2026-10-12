"""Decides whether the PostgreSQL integration tests can run.

`require_admin_url()` is the single entry point: it returns the administrative
server URL for the disposable test databases, or ends the test with a skip or a
failure. The decision lives here rather than in `conftest.py` so that it can be
tested directly, like the disposable-database helpers in `temp_database.py`.

- `TEST_DATABASE_URL` unset or blank: the tests are skipped, because an
  ordinary local unit-test run has no integration database.
- `REQUIRE_INTEGRATION_DB` set to anything but a disabling value turns that
  skip into a failure. The dedicated CI integration run sets it, so that run
  cannot report success while every integration test was skipped.
- `TEST_DATABASE_URL` naming the same database as `DATABASE_URL` is always a
  failure; see `temp_database.py` for the guarantee this warning supplements.
"""

import os

import pytest
from sqlalchemy.engine import make_url

from app.db.config import DATABASE_URL_ENV

TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"
REQUIRE_INTEGRATION_DB_ENV = "REQUIRE_INTEGRATION_DB"
_DISABLING_VALUES = frozenset({"", "0", "false"})


def integration_db_is_required() -> bool:
    """True when a missing `TEST_DATABASE_URL` must fail instead of skip."""
    value = os.environ.get(REQUIRE_INTEGRATION_DB_ENV, "").strip().lower()
    return value not in _DISABLING_VALUES


def require_admin_url() -> str:
    """Return the integration server URL, or skip/fail the calling test."""
    url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if not url:
        message = (
            f"{TEST_DATABASE_URL_ENV} is not set; "
            "PostgreSQL integration tests were NOT executed"
        )
        if integration_db_is_required():
            pytest.fail(
                f"{message} ({REQUIRE_INTEGRATION_DB_ENV} is set)", pytrace=False
            )
        pytest.skip(message)

    developer_url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if developer_url and make_url(url).database == make_url(developer_url).database:
        pytest.fail(
            f"{TEST_DATABASE_URL_ENV} must not point to the same database as "
            f"{DATABASE_URL_ENV}",
            pytrace=False,
        )
    return url
