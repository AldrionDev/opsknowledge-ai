import dataclasses
import math

import pytest

from app.llm.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LLMConfig,
    load_llm_config,
)
from app.llm.errors import LLMConfigurationError

SECRET = "sk-ant-test-secret-value"


def test_defaults_apply_when_only_the_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)

    config = load_llm_config()

    assert config.api_key == SECRET
    assert config.model == DEFAULT_MODEL == "claude-haiku-5-5"
    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert config.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert config.max_retries == DEFAULT_MAX_RETRIES


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("LLM_MODEL", " claude-sonnet-5-5 ")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")

    config = load_llm_config()

    assert config.model == "claude-sonnet-5-5"
    assert config.timeout_seconds == 12.5
    assert config.max_output_tokens == 256
    assert config.max_retries == 0


def test_blank_optional_variables_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    for name in ("LLM_MODEL", "LLM_TIMEOUT_SECONDS", "LLM_MAX_OUTPUT_TOKENS"):
        monkeypatch.setenv(name, "  ")

    config = load_llm_config()

    assert config.model == DEFAULT_MODEL
    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert config.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_missing_key_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        load_llm_config()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_key_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        load_llm_config()


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("LLM_TIMEOUT_SECONDS", "soon-9", "LLM_TIMEOUT_SECONDS"),
        ("LLM_TIMEOUT_SECONDS", "0", "LLM_TIMEOUT_SECONDS"),
        ("LLM_TIMEOUT_SECONDS", "-3", "LLM_TIMEOUT_SECONDS"),
        ("LLM_TIMEOUT_SECONDS", "nan", "LLM_TIMEOUT_SECONDS"),
        ("LLM_TIMEOUT_SECONDS", "inf", "LLM_TIMEOUT_SECONDS"),
        ("LLM_MAX_OUTPUT_TOKENS", "many-9", "LLM_MAX_OUTPUT_TOKENS"),
        ("LLM_MAX_OUTPUT_TOKENS", "1.5", "LLM_MAX_OUTPUT_TOKENS"),
        ("LLM_MAX_OUTPUT_TOKENS", "0", "LLM_MAX_OUTPUT_TOKENS"),
        ("LLM_MAX_RETRIES", "x-9", "LLM_MAX_RETRIES"),
        ("LLM_MAX_RETRIES", "-1", "LLM_MAX_RETRIES"),
    ],
)
def test_invalid_environment_values_are_rejected_without_leaking_them(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv(name, value)

    with pytest.raises(LLMConfigurationError, match=expected) as exc_info:
        load_llm_config()

    assert SECRET not in str(exc_info.value)
    assert value not in str(exc_info.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": ""},
        {"api_key": "   "},
        {"model": ""},
        {"model": "  "},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1.0},
        {"timeout_seconds": math.nan},
        {"timeout_seconds": math.inf},
        {"timeout_seconds": True},
        {"timeout_seconds": "30"},
        {"max_output_tokens": 0},
        {"max_output_tokens": 10.5},
        {"max_output_tokens": True},
        {"max_retries": -1},
        {"max_retries": 1.0},
        {"max_retries": False},
    ],
)
def test_direct_construction_is_validated(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {"api_key": SECRET} | overrides

    with pytest.raises(LLMConfigurationError) as exc_info:
        LLMConfig(**values)  # type: ignore[arg-type]

    assert SECRET not in str(exc_info.value)


@pytest.mark.parametrize(
    ("overrides", "expected_variable"),
    [
        ({"api_key": " "}, "ANTHROPIC_API_KEY"),
        ({"model": " "}, "LLM_MODEL"),
        ({"timeout_seconds": 0}, "LLM_TIMEOUT_SECONDS"),
        ({"max_output_tokens": 0}, "LLM_MAX_OUTPUT_TOKENS"),
        ({"max_retries": -1}, "LLM_MAX_RETRIES"),
    ],
)
def test_validation_errors_name_the_environment_variable(
    overrides: dict[str, object], expected_variable: str
) -> None:
    values: dict[str, object] = {"api_key": SECRET} | overrides

    with pytest.raises(LLMConfigurationError, match=expected_variable):
        LLMConfig(**values)  # type: ignore[arg-type]


def test_replace_revalidates_overrides() -> None:
    config = LLMConfig(api_key=SECRET)

    assert dataclasses.replace(config, max_retries=0).max_retries == 0
    with pytest.raises(LLMConfigurationError):
        dataclasses.replace(config, max_output_tokens=0)


def test_repr_does_not_contain_the_key() -> None:
    assert SECRET not in repr(LLMConfig(api_key=SECRET))


def test_configuration_has_no_sampling_parameters() -> None:
    names = {field.name for field in dataclasses.fields(LLMConfig)}

    assert names.isdisjoint({"temperature", "top_p", "top_k"})
