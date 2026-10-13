"""LLM configuration.

Values come from environment variables and are read when `load_llm_config()` is
called, never at import time:

- `ANTHROPIC_API_KEY`: required, no default.
- `LLM_MODEL`: default `claude-haiku-5-5`.
- `LLM_TIMEOUT_SECONDS`: default 60, must be positive.
- `LLM_MAX_OUTPUT_TOKENS`: default 1024, must be a positive integer.
- `LLM_MAX_RETRIES`: default 2, must be a non-negative integer.

Sampling parameters (`temperature`, `top_p`, `top_k`) are deliberately not
configurable: not every current model accepts them.

Error messages name the environment variable and the violated rule, never the
value, because the configuration carries the API key. Directly constructed
configurations report the same variable names: they are the canonical names of
the settings.

`api_key` is excluded from `repr`, so the key does not reach logs or tracebacks
through an accidental `repr(config)`. That is not a serialization boundary:
`dataclasses.asdict(config)`, `vars()` and attribute access all return the key in
plaintext, as they do for any other value.
"""

import math
import os
from dataclasses import dataclass, field

from app.llm.errors import LLMConfigurationError

API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL_ENV = "LLM_MODEL"
TIMEOUT_ENV = "LLM_TIMEOUT_SECONDS"
MAX_OUTPUT_TOKENS_ENV = "LLM_MAX_OUTPUT_TOKENS"
MAX_RETRIES_ENV = "LLM_MAX_RETRIES"

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class LLMConfig:
    """Validated LLM configuration; direct construction is validated as well."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise LLMConfigurationError(f"{API_KEY_ENV} is not set or is blank")
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMConfigurationError(f"{MODEL_ENV} must be a non-blank string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise LLMConfigurationError(
                f"{TIMEOUT_ENV} must be a positive, finite number"
            )
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise LLMConfigurationError(
                f"{MAX_OUTPUT_TOKENS_ENV} must be a positive integer"
            )
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise LLMConfigurationError(
                f"{MAX_RETRIES_ENV} must be a non-negative integer"
            )


def load_llm_config() -> LLMConfig:
    """Build `LLMConfig` from the environment.

    Raises `LLMConfigurationError` if the API key is missing or blank, or any
    value is invalid. Blank optional variables are treated as unset.
    """
    return LLMConfig(
        api_key=_read(API_KEY_ENV) or "",
        model=_read(MODEL_ENV) or DEFAULT_MODEL,
        timeout_seconds=_parse_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        max_output_tokens=_parse_int(MAX_OUTPUT_TOKENS_ENV, DEFAULT_MAX_OUTPUT_TOKENS),
        max_retries=_parse_int(MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES),
    )


def _read(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _parse_float(name: str, default: float) -> float:
    value = _read(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise LLMConfigurationError(f"{name} must be a number") from None


def _parse_int(name: str, default: int) -> int:
    value = _read(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise LLMConfigurationError(f"{name} must be an integer") from None
