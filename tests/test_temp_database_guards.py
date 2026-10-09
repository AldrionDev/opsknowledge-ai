from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine

from tests.integration.temp_database import (
    TEST_DATABASE_PREFIX,
    create_database,
    drop_database,
    new_test_database_name,
    require_disposable_name,
)

Operation = Callable[[Engine, str], None]


def fake_engine() -> MagicMock:
    engine = MagicMock()
    engine.dialect.identifier_preparer.quote.side_effect = lambda name: f'"{name}"'
    return engine


def executed_sql(engine: MagicMock) -> list[str]:
    connection = engine.connect.return_value.__enter__.return_value
    return [str(call.args[0]) for call in connection.execute.call_args_list]


def test_generated_names_are_prefixed_and_unique() -> None:
    first, second = new_test_database_name(), new_test_database_name()

    assert first.startswith(TEST_DATABASE_PREFIX)
    assert first != second
    require_disposable_name(first)


@pytest.mark.parametrize(
    "name",
    [
        "opsknowledge",
        "postgres",
        "template1",
        "",
        "opsknowledge_test_",
        "opsknowledge_test_foo",
        "opsknowledge_test_" + "A" * 32,
        "opsknowledge_test_" + "a" * 33,
        "x_opsknowledge_test_" + "a" * 32,
        "opsknowledge_test_" + "a" * 32 + '"; DROP DATABASE opsknowledge; --',
    ],
)
@pytest.mark.parametrize("operation", [create_database, drop_database])
def test_helpers_refuse_names_that_are_not_generated_before_connecting(
    operation: Operation, name: str
) -> None:
    engine = fake_engine()

    with pytest.raises(ValueError, match="Refusing to"):
        operation(engine, name)

    engine.connect.assert_not_called()


def test_drop_refuses_a_well_formed_name_that_was_not_created() -> None:
    engine = fake_engine()

    with pytest.raises(ValueError, match="not created by this test run"):
        drop_database(engine, new_test_database_name())

    engine.connect.assert_not_called()


def test_created_database_can_be_dropped_once() -> None:
    engine = fake_engine()
    name = new_test_database_name()

    create_database(engine, name)
    drop_database(engine, name)

    assert executed_sql(engine) == [
        f'CREATE DATABASE "{name}"',
        f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)',
    ]
    with pytest.raises(ValueError, match="not created by this test run"):
        drop_database(engine, name)
