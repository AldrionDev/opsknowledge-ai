"""FastAPI application: routes, lifespan and the generic error response.

The application creates no database engine and no embedding model at startup;
`app.api.dependencies` builds them on the first request that needs them. That
keeps `/health` and the OpenAPI documents available without a database, and the
whole application, `/search` included, available without an LLM configuration.
On shutdown the cached answer chain is dropped and the cached engine is released.

Error responses: request validation keeps FastAPI's HTTP 422 body. The answering
errors of `POST /ask` are `{"detail": "<fixed message>", "code": "<code>"}`:
HTTP 502 for an invalid model output or a provider failure, HTTP 503 when the LLM
is not configured. Every other failure is the generic HTTP 500 response.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.answering.errors import InvalidModelOutputError
from app.api import ask, search
from app.api.ask import (
    LLM_NOT_CONFIGURED,
    LLM_NOT_CONFIGURED_MESSAGE,
    LLM_UNAVAILABLE,
    LLM_UNAVAILABLE_MESSAGE,
    ServiceUnavailableResponse,
    UpstreamErrorResponse,
)
from app.api.dependencies import clear_answer_chain, dispose_retrieval_service
from app.llm.errors import LLMConfigurationError, LLMError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    try:
        clear_answer_chain()
    finally:
        # The engine is released even if dropping the chain ever failed.
        dispose_retrieval_service()


app = FastAPI(
    title="OpsKnowledge AI",
    description="Enterprise RAG knowledge assistant for technical and operational knowledge.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search.router)
app.include_router(ask.router)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Answer an unhandled failure with a structured, detail-free response.

    Starlette re-raises the exception afterwards, so the traceback still reaches
    the server log; only the generic message reaches the client. Request
    validation keeps its own HTTP 422 response.
    """
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# The answering errors carry fixed messages and no cause, and the handlers below
# never use the exception's provider or configuration detail. They are handled
# here, not re-raised, so nothing about them reaches a log.


@app.exception_handler(InvalidModelOutputError)
async def handle_invalid_model_output(
    request: Request, exc: InvalidModelOutputError
) -> JSONResponse:
    # The message is fixed per code and documented as safe for an HTTP body.
    body = UpstreamErrorResponse(detail=str(exc), code=exc.code)
    return JSONResponse(status_code=502, content=body.model_dump(mode="json"))


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    body = UpstreamErrorResponse(detail=LLM_UNAVAILABLE_MESSAGE, code=LLM_UNAVAILABLE)
    return JSONResponse(status_code=502, content=body.model_dump(mode="json"))


@app.exception_handler(LLMConfigurationError)
async def handle_llm_configuration_error(
    request: Request, exc: LLMConfigurationError
) -> JSONResponse:
    body = ServiceUnavailableResponse(
        detail=LLM_NOT_CONFIGURED_MESSAGE, code=LLM_NOT_CONFIGURED
    )
    return JSONResponse(status_code=503, content=body.model_dump(mode="json"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
