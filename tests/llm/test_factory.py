import dataclasses
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic

from app.llm.config import LLMConfig, load_llm_config
from app.llm.errors import LLMConfigurationError
from app.llm.factory import create_chat_model

SECRET = "sk-ant-test-secret-value"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    yield


def test_model_is_built_from_the_configuration() -> None:
    config = LLMConfig(
        api_key=SECRET,
        model="claude-sonnet-5-5",
        timeout_seconds=7.5,
        max_output_tokens=321,
        max_retries=4,
    )

    model = create_chat_model(config)

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5-5"
    assert model.default_request_timeout == 7.5
    assert model.max_tokens == 321
    assert model.max_retries == 4
    assert model.anthropic_api_key.get_secret_value() == SECRET


def test_no_sampling_parameters_are_set() -> None:
    model = create_chat_model(LLMConfig(api_key=SECRET))

    assert model.temperature is None
    assert model.top_p is None
    assert model.top_k is None


def test_no_sampling_parameter_is_configured_on_the_model() -> None:
    """Covers `model_kwargs` passthrough, which the attribute checks would miss.

    Coverage limit: this asserts the model's own configuration, not the serialized
    request. Asserting the outgoing payload would require `_get_request_payload`,
    a private API; unset parameters are omitted from the request, so a model with
    none configured cannot send one.
    """
    model = create_chat_model(LLMConfig(api_key=SECRET))

    configured = model.model_dump(exclude_none=True)

    assert not model.model_kwargs
    assert {"temperature", "top_p", "top_k"}.isdisjoint(configured)


def test_environment_configuration_is_used_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "99")

    model = create_chat_model()

    assert model.model == "claude-haiku-5-5"
    assert model.max_tokens == 99


def test_caller_can_override_limits_of_the_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override path the evaluation harness in #40 uses."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "77")

    config = dataclasses.replace(load_llm_config(), max_retries=0, timeout_seconds=5)
    model = create_chat_model(config)

    assert model.max_retries == 0
    assert model.default_request_timeout == 5
    # Values the caller did not override still come from the environment.
    assert model.max_tokens == 77


def test_missing_key_fails_when_the_factory_is_called() -> None:
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        create_chat_model()


def test_modules_can_be_imported_without_llm_variables() -> None:
    """A fresh interpreter with no LLM variables must import the package.

    Run in a subprocess, because importing in-process cannot show what the
    already-imported modules did at import time.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "ANTHROPIC_API_KEY" and not name.startswith("LLM_")
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.llm.config, app.llm.errors, app.llm.factory",
        ],
        env=environment,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
