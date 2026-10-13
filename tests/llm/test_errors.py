import anthropic

# `httpx2` is the HTTP library the Anthropic SDK is built on. The tests need it
# only to build the response objects its status errors require.
import httpx2
import pytest
from langchain_anthropic import chat_models as anthropic_chat_models
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
    OutputParserException,
)

from app.llm.errors import GENERIC_FAILURE_MESSAGE, LLMError, map_llm_error

SECRET = "sk-ant-test-secret-value"
PROVIDER_TEXT = f"invalid x-api-key {SECRET} for request req_0123456789"

MODEL_ERROR_TYPES = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    ContextOverflowError,
    ModelError,
)


def _response(status_code: int) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status_code, request=request)


def _assert_safe(mapped: LLMError | None) -> str:
    assert isinstance(mapped, LLMError)
    assert mapped.__cause__ is None
    message = str(mapped)
    assert message
    assert SECRET not in message
    assert "req_0123456789" not in message
    assert "x-api-key" not in message
    return message


@pytest.mark.parametrize("error_type", MODEL_ERROR_TYPES)
def test_model_errors_map_to_a_fixed_message(error_type: type[ModelError]) -> None:
    assert _assert_safe(map_llm_error(error_type(PROVIDER_TEXT)))


@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        (anthropic_chat_models.AnthropicAuthenticationError, 401),
        (anthropic_chat_models.AnthropicPermissionDeniedError, 403),
        (anthropic_chat_models.AnthropicInvalidRequestError, 400),
        (anthropic_chat_models.AnthropicModelNotFoundError, 404),
        (anthropic_chat_models.AnthropicRateLimitError, 429),
        (anthropic_chat_models.AnthropicAPIError, 500),
        (anthropic_chat_models.AnthropicOverloadedError, 529),
        (anthropic_chat_models.AnthropicContextOverflowError, 400),
    ],
)
def test_classified_provider_errors_keep_their_specific_message(
    error_type: type[Exception], status_code: int
) -> None:
    """The classes the Anthropic integration actually raises, not just the bases."""
    source = error_type(
        message=PROVIDER_TEXT, response=_response(status_code), body=None
    )

    assert _assert_safe(map_llm_error(source)) != GENERIC_FAILURE_MESSAGE


@pytest.mark.parametrize(
    "source",
    [
        # HTTP 409 and 422: the Anthropic integration re-raises these unchanged,
        # so they never become `ModelError` instances.
        anthropic.ConflictError(PROVIDER_TEXT, response=_response(409), body=None),
        anthropic.UnprocessableEntityError(
            PROVIDER_TEXT, response=_response(422), body=None
        ),
        anthropic.APIStatusError(PROVIDER_TEXT, response=_response(418), body=None),
        anthropic.APIResponseValidationError(
            _response(200), None, message=PROVIDER_TEXT
        ),
        anthropic.APIError(PROVIDER_TEXT, request=_response(500).request, body=None),
    ],
)
def test_unclassified_provider_errors_map_to_the_generic_message(
    source: Exception,
) -> None:
    assert PROVIDER_TEXT in str(source), "the source must carry provider text"

    assert _assert_safe(map_llm_error(source)) == GENERIC_FAILURE_MESSAGE


@pytest.mark.parametrize(
    "source",
    [
        # Siblings of `APIError` under `anthropic.AnthropicError`: local
        # credential-resolution problems, not failures of a provider request.
        anthropic.CredentialsError(PROVIDER_TEXT),
        anthropic.RetryableError(PROVIDER_TEXT),
        anthropic.WorkloadIdentityError(PROVIDER_TEXT),
        anthropic.IdentityTokenFileError(PROVIDER_TEXT),
    ],
)
def test_sdk_errors_that_are_not_request_failures_are_not_mapped(
    source: Exception,
) -> None:
    assert not isinstance(source, anthropic.APIError), "guards the premise"

    assert map_llm_error(source) is None


def test_failure_kinds_have_distinct_messages() -> None:
    kinds = [
        ModelAuthenticationError,
        ModelRateLimitError,
        ModelTimeoutError,
        ModelConnectionError,
    ]

    messages = {str(map_llm_error(kind("x"))) for kind in kinds}

    assert len(messages) == len(kinds)


def test_message_table_order_is_irrelevant() -> None:
    """Every mapped class is a direct sibling, so no entry can shadow another."""
    from app.llm.errors import _MESSAGES

    for error_type, _ in _MESSAGES:
        others = [other for other, _ in _MESSAGES if other is not error_type]
        assert not issubclass(error_type, tuple(others))


def test_llm_error_is_returned_unchanged() -> None:
    error = LLMError("already mapped")

    assert map_llm_error(error) is error


@pytest.mark.parametrize(
    "exc",
    [
        OutputParserException("not json"),
        ValueError("bad input"),
        KeyError("missing"),
        RuntimeError("bug"),
        TypeError("Could not resolve authentication method"),
    ],
)
def test_other_exceptions_are_not_mapped(exc: Exception) -> None:
    assert map_llm_error(exc) is None
