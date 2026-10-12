"""FastAPI application: routes, lifespan and the generic error response.

The application creates no database engine and no embedding model at startup;
`app.api.dependencies` builds them on the first request that needs them. That
keeps `/health` and the OpenAPI documents available without a database. On
shutdown the cached engine is released.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import search
from app.api.dependencies import dispose_retrieval_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    dispose_retrieval_service()


app = FastAPI(
    title="OpsKnowledge AI",
    description="Enterprise RAG knowledge assistant for technical and operational knowledge.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search.router)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Answer an unhandled failure with a structured, detail-free response.

    Starlette re-raises the exception afterwards, so the traceback still reaches
    the server log; only the generic message reaches the client. Request
    validation keeps its own HTTP 422 response.
    """
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
