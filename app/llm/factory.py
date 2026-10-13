"""Chat model factory.

`create_chat_model` builds one `ChatAnthropic` per call from a validated
`LLMConfig`. Nothing is created or read at import time, and construction makes
no network call. The factory passes no sampling parameters.

Callers that need other limits, such as the evaluation harness, derive a
configuration with `dataclasses.replace(load_llm_config(), max_retries=0)`.
"""

from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.llm.config import LLMConfig, load_llm_config


def create_chat_model(config: LLMConfig | None = None) -> ChatAnthropic:
    """Return a chat model for `config`, or for the environment if it is `None`.

    Raises `LLMConfigurationError` if the environment configuration is invalid.
    """
    config = config or load_llm_config()
    return ChatAnthropic(
        model_name=config.model,
        api_key=SecretStr(config.api_key),
        timeout=config.timeout_seconds,
        max_tokens_to_sample=config.max_output_tokens,
        max_retries=config.max_retries,
        stop=None,  # explicit only to satisfy mypy; no stop sequences are used
    )
