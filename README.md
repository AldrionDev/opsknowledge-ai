# OpsKnowledge AI

OpsKnowledge AI is being developed as an enterprise RAG knowledge assistant
for technical documentation, operational runbooks, incident reports, and
operational knowledge.

The repository provides a full RAG answering pipeline: documents are loaded,
chunked, embedded locally, stored in PostgreSQL with pgvector, and searched by
meaning. Retrieved context is assembled, sent to a language model for structured
generation, and grounding-validated before the answer is returned.

```text
documents -> chunks -> embeddings -> pgvector -> semantic retrieval
                                                       |
                                              context assembly
                                                       |
                                              structured generation
                                                       |
                                              grounding validation
                                                       |
                                            grounded answer (or insufficient context)
```

`POST /search` returns ranked source chunks. `POST /ask` returns a grounded answer
with citations, or an explicit insufficient-context response when the corpus does
not contain the answer. See [Development status](#development-status) for details.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git
- [Docker](https://docs.docker.com/get-docker/) with Docker Compose (the
  `docker compose` command), for the local PostgreSQL + pgvector database
- Internet access on first use, to pull the database image and to download the
  embedding model from the Hugging Face Hub

The project is pinned to Python 3.12 through `.python-version`. uv uses the
required Python version for the project environment.

No cloud account or API key is required for ingestion, retrieval and
`POST /search`. Embeddings are computed locally on the CPU. An Anthropic API key
is needed for `POST /ask` and for the live evaluation run; both use the chat model
factory (`app/llm/`).

## Local setup

```bash
git clone https://github.com/AldrionDev/opsknowledge-ai.git
cd opsknowledge-ai
uv sync --locked
```

`uv sync --locked` creates the `.venv` virtual environment and installs the
runtime and development dependencies exactly as pinned in `uv.lock`.

Run project commands through `uv run`; activating the environment is not
required.

## Configuration

```bash
cp .env.example .env
```

| Variable                | Required for                                                                  |
| ----------------------- | ----------------------------------------------------------------------------- |
| `DATABASE_URL`          | Migrations, ingestion, `POST /search` and `POST /ask`                         |
| `TEST_DATABASE_URL`     | Database integration tests only ([Running tests](#running-tests))             |
| `ANTHROPIC_API_KEY`     | `POST /ask` and live evaluation; read when the chat model is created          |
| `LLM_MODEL`             | Optional, default `claude-haiku-5-5`                                          |
| `LLM_TIMEOUT_SECONDS`   | Optional, default `60`; must be positive                                      |
| `LLM_MAX_OUTPUT_TOKENS` | Optional, default `1024`; must be a positive integer                          |
| `LLM_MAX_RETRIES`       | Optional, default `2`; must be a non-negative integer (SDK transport retries) |

`.env.example` contains development-only values that match the database
credentials in `compose.yaml`; no edit is needed for the local setup. Its
`ANTHROPIC_API_KEY` is a placeholder: set a real key only in your local `.env`.
`.env` is ignored by Git. Never commit real credentials or secrets. No sampling
parameters (`temperature`, `top_p`, `top_k`) are configured.

`.env` is not loaded automatically. The commands below that need the database
pass it explicitly with `uv run --env-file .env`.

## Database

Start PostgreSQL with the pgvector extension and wait until it is healthy:

```bash
docker compose up -d --wait
docker compose ps
```

`docker compose ps` should report the `db` service as `healthy`. The database
listens on `127.0.0.1:5432`.

Apply the database migrations:

```bash
uv run --env-file .env alembic upgrade head
```

Alembic migrations are the only supported way to create or update the schema.
See [docs/database.md](docs/database.md) for the schema, the repository and
details of the local database workflow.

## Knowledge corpus

`data/knowledge/` contains the synthetic operational knowledge corpus used for
development and evaluation. All content is fictional and contains no
proprietary or production knowledge.

- `runbooks/` holds general, reusable procedures for diagnosing and resolving a
  class of problem.
- `incidents/` holds reports of specific past events: impact, timeline, root
  cause, resolution and follow-up actions.

See [data/knowledge/README.md](data/knowledge/README.md) for the full
description.

## Ingesting the corpus

With the database running and the migrations applied:

```bash
uv run --env-file .env python -m app.ingestion
```

Ingestion runs the following pipeline for every document of the corpus:

```text
load
 -> chunk
 -> embed
 -> persist
```

and prints a summary when it completes:

```text
Ingesting corpus: <repository>/data/knowledge
Ingestion completed: documents=<n> chunks_generated=<n> chunks_persisted=<n>
```

- The first run downloads the embedding model; see
  [docs/embeddings.md](docs/embeddings.md) for caching and offline use.
- Re-ingesting unchanged documents is designed to be idempotent: the stored
  chunks of each document are replaced, not duplicated. Chunks of documents that
  were deleted or renamed in the corpus are not removed.
- `--corpus-root <path>` ingests another directory with the same
  `runbooks/` and `incidents/` layout.
- On failure the command exits with status 1 and an `error:` message on stderr,
  for example when the migrations have not been applied.

Ingestion is a local command. The API has no ingestion endpoint.

## Running the API

```bash
uv run --env-file .env uvicorn app.main:app --reload
```

The API is served at <http://127.0.0.1:8000>. The server runs in the foreground:
run the search requests below in a second terminal, and stop the server with
`Ctrl+C`.

`--env-file .env` supplies `DATABASE_URL`, which `POST /search` needs. The plain
development command `uv run uvicorn app.main:app --reload` starts the same
application without any database configuration: `GET /health`, `GET /docs` and
`GET /openapi.json` work, but `POST /search` responds with HTTP 500. Use the
plain command only when search is not needed.

| Endpoint            | Description                                        |
| ------------------- | -------------------------------------------------- |
| `GET /health`       | Health check, returns `{"status": "ok"}`           |
| `POST /search`      | Semantic search over the ingested knowledge corpus |
| `POST /ask`         | Grounded answer from the knowledge corpus          |
| `GET /docs`         | Interactive API documentation (Swagger UI)         |
| `GET /openapi.json` | OpenAPI schema                                     |

## Semantic search

`POST /search` embeds the query and returns the most similar stored chunks.
Ingest the corpus first: a migrated but empty database is not an error, it
returns an empty `results` list.

Request body:

| Field   | Type    | Description                                                  |
| ------- | ------- | ------------------------------------------------------------ |
| `query` | string  | Required. Natural-language query; must not be blank.         |
| `limit` | integer | Optional. Maximum number of results, 1 to 50. Defaults to 5. |

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is the pod repeatedly restarting?",
    "limit": 5
  }'
```

The response contains the query and the ranked source chunks, most similar
first. It is not a generated answer:

```text
{
  "query": "Why is the pod repeatedly restarting?",
  "results": [
    {
      "chunk_id": "<chunk id>",
      "document_id": "<document id>",
      "chunk_index": <position of the chunk in its document>,
      "source_path": "<runbooks|incidents>/<document>.md",
      "document_type": "<runbook|incident>",
      "content": "<chunk text>",
      "similarity_score": <score>
    }
  ]
}
```

The block only illustrates the shape of the response. Values in angle brackets
are placeholders, not literal values, so it is not valid JSON.

| Field              | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `chunk_id`         | Identifier of the chunk                                       |
| `document_id`      | Identifier of the source document                             |
| `chunk_index`      | Zero-based position of the chunk within its document          |
| `source_path`      | Path of the source document, relative to the corpus root      |
| `document_type`    | `runbook` or `incident`                                       |
| `content`          | Text of the chunk                                             |
| `similarity_score` | Cosine similarity, -1 to 1; higher is more similar            |

A request with a blank `query` or a `limit` outside 1 to 50 is rejected with
HTTP 422. If the database is not configured, not reachable or not migrated, the
API responds with HTTP 500 and a generic error body. The first search request
after startup is slower because it loads the embedding model.

## Grounded answering

`POST /ask` retrieves the most similar chunks, assembles them into a numbered
context, sends the context and question to the language model for structured
generation, validates the model's output for grounding (structural validity and
evidence presence), and returns either an answer with citations or an explicit
insufficient-context response.

The service is started the same way as `POST /search` (see [Running the
API](#running-the-api)) and requires both `DATABASE_URL` and `ANTHROPIC_API_KEY`
in the environment.

Request body:

| Field      | Type    | Description                                                             |
| ---------- | ------- | ----------------------------------------------------------------------- |
| `question` | string  | Required. Natural-language question; must not be blank; max 2000 chars. |
| `limit`    | integer | Optional. Maximum chunks retrieved, 1 to 50. Defaults to 5.            |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why is the payments pod repeatedly restarting?",
    "limit": 5
  }'
```

**Answered response** (HTTP 200):

```json
{
  "question": "Why is the payments pod repeatedly restarting?",
  "status": "answered",
  "reason": null,
  "answer": "The pod is killed when it exceeds its memory limit. [1]",
  "citations": [
    {
      "source_id": 1,
      "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "document_id": "d1e2f3a4-b5c6-7890-abcd-ef1234567890",
      "source_path": "runbooks/oom-killed-pod.md",
      "document_type": "runbook",
      "chunk_index": 2,
      "evidence_quote": "container is killed when it exceeds its memory limit"
    }
  ]
}
```

Each line of `answer` ends with a `[source_id]` marker referring to the
corresponding entry in `citations`. `evidence_quote` is a contiguous substring
of the cited source, normalized for whitespace.

**Insufficient-context response** (HTTP 200):

```json
{
  "question": "How do I rotate the database encryption key?",
  "status": "insufficient_context",
  "reason": "model_declined",
  "answer": null,
  "citations": []
}
```

`reason` is one of `no_retrieval` (no chunks found), `empty_context` (no chunks
fit the context budget), or `model_declined` (the model determined it lacked
sufficient information).

**Error responses:**

| HTTP | `code`                 | Cause                                            |
| ---- | ---------------------- | ------------------------------------------------ |
| 422  | —                      | Blank question, `limit` outside 1–50             |
| 502  | `malformed_output`     | Model output could not be parsed                 |
| 502  | `unknown_citation`     | Model cited a source not in the retrieved context|
| 502  | `missing_evidence`     | Evidence quote absent from source or too short   |
| 502  | `inconsistent_response`| Structural contradiction in model output         |
| 502  | `llm_unavailable`      | Provider or transport failure                    |
| 503  | `llm_not_configured`   | `ANTHROPIC_API_KEY` missing or configuration invalid |
| 500  | —                      | Unexpected internal error                        |

502 and 503 error bodies have the shape `{"detail": "<fixed message>", "code": "<code>"}`.
HTTP 422 uses FastAPI's standard validation error format (`detail` is an array).
HTTP 500 returns `{"detail": "Internal server error"}` with no `code` field.
See [docs/answer-generation.md](docs/answer-generation.md) for the full pipeline,
grounding levels, evidence normalization, and known limitations.

## Evaluation

An offline evaluation harness runs the questions of `data/eval/questions.jsonl`
through the answer service and writes a JSON and Markdown report:

```bash
uv run python -m app.evaluation             # prints the bounds, runs nothing
uv run python -m app.evaluation --dry-run   # synthetic, fully offline, NOT A BASELINE
```

A real run (`--live`) calls the Anthropic API within explicit request, retry and
output-token bounds and requires a separately prepared evaluation database. A live
baseline was completed on 2026-11-02 using `claude-haiku-5-5` over 24 questions.
See [docs/evaluation.md](docs/evaluation.md) for the baseline results, modes,
metrics, cost bounds and the live-run procedure.

## Running tests

```bash
uv run pytest
```

This runs the unit and API tests, which need no database. The database
integration tests (`tests/integration/`, marker `integration`) are skipped, and
reported as skipped, when `TEST_DATABASE_URL` is not set. To run only the unit
and API tests, as the CI `checks` job does:

```bash
uv run pytest -m "not integration"
```

The database integration tests need a running PostgreSQL + pgvector instance
and `TEST_DATABASE_URL` from `.env`:

```bash
docker compose up -d --wait
uv run --env-file .env pytest -m integration -rs
```

The tests run in temporary databases of their own and do not modify the
development database. Skipped integration tests mean the database was not
exercised, so check the report for skips. See
[docs/database.md](docs/database.md#tests) for how the tests are isolated.

## Stopping the database

Optional final step, once you are done with the workflow above:

```bash
docker compose down
```

The data is kept in a Docker volume and survives `docker compose down`; the next
`docker compose up -d --wait` starts the database with the same data.

## Code quality

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

GitHub Actions runs on pull requests to, and pushes to, `main`: the `checks` job
runs the commands above and the unit and API tests without a database, and the
`integration` job runs the database integration tests against a PostgreSQL +
pgvector service container. See
[docs/database.md](docs/database.md#continuous-integration).

## Project structure

```text
.
├── app/
│   ├── main.py              # FastAPI application and /health endpoint
│   ├── api/                 # POST /search and POST /ask endpoints, dependency wiring
│   ├── answering/           # Context assembly, structured generation and grounding validation
│   ├── ingestion/           # Document loading, chunking and ingestion CLI
│   ├── embeddings/          # Local embedding service (FastEmbed)
│   ├── evaluation/          # Offline evaluation harness and report
│   ├── llm/                 # LLM configuration, chat model factory, errors
│   ├── db/                  # PostgreSQL + pgvector persistence
│   └── retrieval/           # Semantic retrieval service
├── data/
│   ├── eval/                # Evaluation dataset
│   └── knowledge/           # Synthetic runbooks and incident reports
├── migrations/              # Alembic migrations
├── tests/                   # Unit and API tests
│   └── integration/         # Database integration tests (PostgreSQL + pgvector)
├── scripts/                 # Manual verification scripts
├── docs/
│   ├── ai/                  # Project context and development workflow notes
│   ├── database.md          # Database, migrations and integration testing
│   ├── evaluation.md        # Evaluation harness, metrics and baseline status
│   └── embeddings.md        # Embedding model and local caching
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── compose.yaml             # Local PostgreSQL + pgvector service
├── alembic.ini              # Alembic configuration
├── .env.example             # Safe local configuration template
├── pyproject.toml           # Dependencies, Ruff, mypy and pytest configuration
├── uv.lock                  # Locked dependency versions
└── .python-version          # Python version used by uv
```

## Development status

The project is under active development.

Currently implemented:

- synthetic operational knowledge corpus (runbooks and incident reports)
- document loading and normalization
- deterministic document chunking
- local embedding generation
- PostgreSQL + pgvector persistence with Alembic migrations
- idempotent document ingestion through a local command
- semantic retrieval over the ingested chunks
- `POST /search` API returning ranked source chunks
- `GET /health` endpoint
- LLM integration (`app/llm/`) with configurable model, timeout, retries and output tokens
- context assembly with ranking, deduplication and character budget
- structured answer generation via the Anthropic API
- grounding validation (structural validity and evidence-presence checks)
- `POST /ask` API returning grounded answers with citations
- insufficient-context responses with explicit reason codes
- offline evaluation harness with heuristic metrics and a completed live baseline
- automated unit and database integration tests
- Ruff and mypy quality checks
- GitHub Actions CI

Not implemented and planned for later development:

- LangGraph and agent frameworks
- MCP (Model Context Protocol) integrations
- conversational memory
- LLM tracing and observability integrations
- deployment configuration
