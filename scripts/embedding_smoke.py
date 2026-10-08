"""Manual smoke verification of the real local embedding model.

Not part of pytest or CI: it loads the actual model, which is downloaded from
the Hugging Face Hub on the first run (about 67 MB, internet required). Run it
from the repository root:

    uv run python -m scripts.embedding_smoke

Optional: set `FASTEMBED_CACHE_PATH` to a persistent directory (the default is in
the system temp dir). To verify offline operation after the first run, repeat
with `HF_HUB_OFFLINE=1`.

Exits with a non-zero status if any check fails.
"""

import math
import time

from app.embeddings.fastembed_service import (
    MODEL_DIMENSION,
    MODEL_NAME,
    FastEmbedService,
)

DOCUMENTS = [
    "To restart a crashed Kubernetes pod, delete it and let the deployment recreate it.",
    "Postgres failover: promote the replica, then repoint the application connection string.",
    "Incident postmortem: the 502 errors were caused by an exhausted nginx upstream pool.",
    "Rotate the TLS certificate with certbot and reload the ingress controller.",
]
QUERY = "How do I recover from nginx 502 errors?"

NORM_TOLERANCE = 1e-3
BATCH_INVARIANCE_TOLERANCE = 1e-3


def check(condition: bool, message: str) -> None:
    print(f"[{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        raise SystemExit(1)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def main() -> None:
    started = time.perf_counter()
    service = FastEmbedService()
    load_seconds = time.perf_counter() - started
    print(f"model={MODEL_NAME} dimension={service.dimension} load={load_seconds:.2f}s")
    check(
        service.dimension == MODEL_DIMENSION, "service exposes the documented dimension"
    )

    started = time.perf_counter()
    vectors = service.embed_documents(DOCUMENTS)
    first_seconds = time.perf_counter() - started
    check(
        len(vectors) == len(DOCUMENTS),
        f"{len(vectors)} vectors for {len(DOCUMENTS)} documents",
    )
    check(
        all(len(v) == MODEL_DIMENSION for v in vectors),
        f"document vectors have {MODEL_DIMENSION} elements",
    )
    check(
        all(math.isfinite(x) for v in vectors for x in v), "document vectors are finite"
    )
    check(
        all(abs(norm(v) - 1.0) < NORM_TOLERANCE for v in vectors),
        f"document vectors are L2-normalized (tolerance {NORM_TOLERANCE})",
    )

    query = service.embed_query(QUERY)
    check(len(query) == MODEL_DIMENSION, "query vector has the document dimension")
    check(abs(norm(query) - 1.0) < NORM_TOLERANCE, "query vector is L2-normalized")

    single = service.embed_documents([DOCUMENTS[1]])[0]
    drift = max(abs(a - b) for a, b in zip(single, vectors[1], strict=True))
    check(
        drift < BATCH_INVARIANCE_TOLERANCE,
        f"input order preserved (batch vs single drift {drift:.2e})",
    )

    started = time.perf_counter()
    repeated = service.embed_documents(DOCUMENTS)
    second_seconds = time.perf_counter() - started
    check(
        repeated == vectors,
        "repeated call on the same instance returns identical vectors",
    )
    print(
        f"model reuse: load={load_seconds:.2f}s, first batch={first_seconds:.3f}s, "
        f"second batch={second_seconds:.3f}s"
    )
    print("smoke verification passed")


if __name__ == "__main__":
    main()
