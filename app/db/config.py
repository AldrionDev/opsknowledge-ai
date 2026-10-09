"""Database configuration.

The connection is configured through the `DATABASE_URL` environment variable
only. There is no default: a missing or invalid value raises
`DatabaseConfigError`. Error messages never contain the URL, because it carries
the database password.
"""

import os

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

DATABASE_URL_ENV = "DATABASE_URL"
SUPPORTED_DRIVER = "postgresql+psycopg"


class DatabaseConfigError(Exception):
    """Raised when the database configuration is missing or invalid."""


def get_database_url() -> str:
    """Return the validated `DATABASE_URL`.

    Raises `DatabaseConfigError` if it is unset, blank, not a valid URL, or does
    not use the `postgresql+psycopg` driver.
    """
    value = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not value:
        raise DatabaseConfigError(
            f"{DATABASE_URL_ENV} is not set. Copy .env.example to .env and "
            "provide it, for example with `uv run --env-file .env ...`."
        )

    try:
        url = make_url(value)
    except ArgumentError:
        raise DatabaseConfigError(
            f"{DATABASE_URL_ENV} is not a valid database URL"
        ) from None

    if url.drivername != SUPPORTED_DRIVER:
        raise DatabaseConfigError(
            f"{DATABASE_URL_ENV} must use the '{SUPPORTED_DRIVER}' driver"
        )
    return value
