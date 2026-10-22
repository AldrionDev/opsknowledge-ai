import os
import socket
from typing import Any

import pytest
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from app.llm import config


def clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the API key and every `LLM_*` variable inherited from the shell."""
    for name in list(os.environ):
        if name == config.API_KEY_ENV or name.startswith("LLM_"):
            monkeypatch.delenv(name)


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outgoing socket connection fail the test."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)


class FakeStructuredChatModel:
    """Offline stand-in for the chat model at the structured-output boundary.

    `build_answer_chain` only calls `with_structured_output`, so the real prompt
    template still renders the request; the returned runnable records every
    rendered prompt and returns `result` or raises `failure`. Both can be changed
    between calls. Nothing is sent anywhere.
    """

    def __init__(self, result: object = None, failure: Exception | None = None) -> None:
        self.result = result
        self.failure = failure
        self.structured_output_calls: list[tuple[object, str | None]] = []
        self.prompts: list[PromptValue] = []

    def with_structured_output(
        self, schema: object, *, method: str | None = None, **kwargs: Any
    ) -> Runnable[Any, Any]:
        self.structured_output_calls.append((schema, method))
        return RunnableLambda(self._invoke)

    def _invoke(self, prompt: PromptValue) -> Any:
        self.prompts.append(prompt)
        if self.failure is not None:
            raise self.failure
        return self.result

    def prompt_text(self, index: int = -1) -> str:
        """All message contents of one recorded prompt, joined."""
        return "\n".join(str(m.content) for m in self.prompts[index].to_messages())
