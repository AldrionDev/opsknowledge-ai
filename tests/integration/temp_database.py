"""Disposable PostgreSQL databases for integration tests.

Safety rules: these helpers only ever create or drop databases named
`opsknowledge_test_<32 hex characters>`, and only drop a database that
`create_database` created in the same process. A developer-owned database can
therefore never be affected, even if its name starts with the same prefix.
"""

import re
import uuid

from sqlalchemy import Engine, text

TEST_DATABASE_PREFIX = "opsknowledge_test_"
_DISPOSABLE_NAME = re.compile(rf"{TEST_DATABASE_PREFIX}[0-9a-f]{{32}}")

_created: set[str] = set()


def new_test_database_name() -> str:
    return f"{TEST_DATABASE_PREFIX}{uuid.uuid4().hex}"


def require_disposable_name(name: str) -> None:
    if not _DISPOSABLE_NAME.fullmatch(name):
        raise ValueError(
            f"Refusing to touch database '{name}': only generated "
            f"'{TEST_DATABASE_PREFIX}<32 hex characters>' databases are disposable"
        )


def create_database(admin_engine: Engine, name: str) -> None:
    require_disposable_name(name)
    quoted = admin_engine.dialect.identifier_preparer.quote(name)
    with admin_engine.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {quoted}"))
    _created.add(name)


def drop_database(admin_engine: Engine, name: str) -> None:
    require_disposable_name(name)
    if name not in _created:
        raise ValueError(
            f"Refusing to drop database '{name}': it was not created by this test run"
        )
    quoted = admin_engine.dialect.identifier_preparer.quote(name)
    with admin_engine.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {quoted} WITH (FORCE)"))
    _created.discard(name)
