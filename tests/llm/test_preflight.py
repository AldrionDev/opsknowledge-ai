import pytest

from app.llm.preflight import (
    ENDPOINT_OVERRIDE_VARS,
    TRACING_VARS,
    find_unsafe_environment,
)

GUARDED_VARS = (*ENDPOINT_OVERRIDE_VARS, *TRACING_VARS)
SECRET_VALUE = "https://SENTINEL-redirect-host-4d2f.example"


def test_all_blocked_names_are_reported_together() -> None:
    environ = dict.fromkeys(GUARDED_VARS, "true")

    assert sorted(find_unsafe_environment(environ)) == sorted(GUARDED_VARS)


@pytest.mark.parametrize("value", ["", "  ", "false", "False", "0", "no", "off"])
def test_tracing_explicitly_off_or_blank_is_allowed(value: str) -> None:
    environ = dict.fromkeys(TRACING_VARS, value)

    assert find_unsafe_environment(environ) == []


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "enabled"])
@pytest.mark.parametrize("name", TRACING_VARS)
def test_active_tracing_is_unsafe(name: str, value: str) -> None:
    assert find_unsafe_environment({name: value}) == [name]


@pytest.mark.parametrize("name", ENDPOINT_OVERRIDE_VARS)
def test_endpoint_override_is_unsafe_and_only_its_name_is_reported(name: str) -> None:
    result = find_unsafe_environment({name: SECRET_VALUE})

    assert result == [name]
    assert SECRET_VALUE not in "".join(result)


@pytest.mark.parametrize("name", ENDPOINT_OVERRIDE_VARS)
def test_blank_endpoint_variable_is_allowed(name: str) -> None:
    assert find_unsafe_environment({name: "  "}) == []


def test_unrelated_variables_are_ignored() -> None:
    assert (
        find_unsafe_environment({"HTTPS_PROXY": "http://proxy", "PATH": "/bin"}) == []
    )
