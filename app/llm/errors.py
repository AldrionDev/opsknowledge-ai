"""Application-level LLM errors.

`LLMConfigurationError` signals an invalid or missing configuration.
`LLMError` signals a provider or transport failure of a model call.

Messages are fixed strings: they never contain the API key, configuration
values or the text of the provider exception.

Why this module imports the Anthropic SDK
-----------------------------------------

`langchain_anthropic` classifies most provider failures into the
`langchain_core.exceptions.ModelError` hierarchy, and those classes drive the
specific messages below. It does not classify all of them: its error handler
re-raises the remaining SDK exceptions unchanged, so HTTP 409, HTTP 422 and
`APIResponseValidationError` reach callers as plain `anthropic` exceptions whose
messages may echo request or response detail. Recognizing `anthropic.APIError`
is therefore required to keep the fixed-message guarantee above; the SDK is used
for nothing else. It is a hard dependency of the declared `langchain-anthropic`
package, not a second provider integration: no client, model or configuration
from it is used anywhere in the application.

`APIError` is the SDK's base for failures of a request to the provider. Its
siblings under `anthropic.AnthropicError`, such as credential-resolution and
toolset misuse errors, report local problems rather than provider failures, so
they stay unmapped like any other programming error.
"""

from anthropic import APIError
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class LLMError(Exception):
    """Raised when a call to the language model provider fails."""


class LLMConfigurationError(Exception):
    """Raised when the LLM configuration is missing or invalid."""


GENERIC_FAILURE_MESSAGE = "LLM provider request failed"

# Pairs of (provider error class, application message). The classes are
# independent siblings of `ModelError`, so the order does not matter; the test
# suite pins that property against the locked `langchain-core` version.
_MESSAGES: tuple[tuple[type[ModelError], str], ...] = (
    (ModelAuthenticationError, "LLM provider authentication failed"),
    (ModelPermissionDeniedError, "LLM provider denied permission for the request"),
    (ModelNotFoundError, "Configured LLM model was not found"),
    (ModelRateLimitError, "LLM provider rate limit exceeded"),
    (ModelTimeoutError, "LLM request timed out"),
    (ModelConnectionError, "LLM provider could not be reached"),
    (ModelAPIError, "LLM provider reported a server error"),
    (ContextOverflowError, "LLM input exceeds the model context limit"),
    (ModelInvalidRequestError, "LLM provider rejected the request as invalid"),
)


def map_llm_error(exc: Exception) -> LLMError | None:
    """Map a provider failure to `LLMError`.

    Returns `None` when `exc` is not a provider failure, for example an output
    parsing error or a programming error. The caller must re-raise such
    exceptions unchanged. An `LLMError` is returned as is.

    Classified `ModelError` failures keep their specific message. Any other
    Anthropic request failure, such as HTTP 409, HTTP 422 or a response
    validation error, maps to `GENERIC_FAILURE_MESSAGE`, so no provider text
    escapes.

    The returned error carries a fixed message and no cause. Callers should raise
    it with `raise mapped from None`, so the raw provider exception, which may
    echo request details, does not end up in tracebacks or logs.
    """
    if isinstance(exc, LLMError):
        return exc
    if not isinstance(exc, ModelError | APIError):
        return None
    for error_type, message in _MESSAGES:
        if isinstance(exc, error_type):
            return LLMError(message)
    return LLMError(GENERIC_FAILURE_MESSAGE)
