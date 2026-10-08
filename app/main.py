from fastapi import FastAPI

app = FastAPI(
    title="OpsKnowledge AI",
    description="Enterprise RAG knowledge assistant for technical and operational knowledge.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
