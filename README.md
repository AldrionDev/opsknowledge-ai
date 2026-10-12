# OpsKnowledge AI

OpsKnowledge AI is being developed as an enterprise RAG knowledge assistant
for technical documentation, operational runbooks, incident reports, and
operational knowledge.

The repository currently provides an operational knowledge ingestion and
semantic retrieval foundation: documents are loaded, chunked, embedded locally,
stored in PostgreSQL with pgvector, and searched by meaning through a
`POST /search` API.

```text
implemented:
documents -> chunks -> embeddings -> pgvector -> semantic retrieval

not yet implemented:
retrieved context -> LLM -> grounded generated answer
```

Search returns ranked source chunks with their metadata. It does not generate
answers: LLM integration and RAG answer generation are planned, not yet
implemented. See [Development status](#development-status) for details.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git
- [Docker](https://docs.docker.com/get-docker/) with Docker Compose (the
  `docker compose` command), for the local PostgreSQL + pgvector database
- Internet access on first use, to pull the database image and to download the
  embedding model from the Hugging Face Hub

The project is pinned to Python 3.12 through `.python-version`. uv uses the
required Python version for the project environment.

No cloud account or API key is required. Embeddings are computed locally on
the CPU.

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

| Variable            | Required for                                                      |
| ------------------- | ----------------------------------------------------------------- |
| `DATABASE_URL`      | Migrations, ingestion and `POST /search`                          |
| `TEST_DATABASE_URL` | Database integration tests only ([Running tests](#running-tests)) |

`.env.example` contains development-only values that match the database
credentials in `compose.yaml`; no edit is needed for the local setup. `.env` is
ignored by Git. Never commit real credentials or secrets.

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

Stop the database:

```bash
docker compose down
```

The data is kept in a Docker volume and survives `docker compose down`. See
[docs/database.md](docs/database.md) for the schema, the repository and details
of the local database workflow.

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

The API is served at <http://127.0.0.1:8000>.

`--env-file .env` supplies `DATABASE_URL`, which `POST /search` needs. The plain
development command `uv run uvicorn app.main:app --reload` starts the same
application without any database configuration: `GET /health`, `GET /docs` and
`GET /openapi.json` work, but `POST /search` responds with HTTP 500. Use the
plain command only when search is not needed.

| Endpoint            | Description                                        |
| ------------------- | -------------------------------------------------- |
| `GET /health`       | Health check, returns `{"status": "ok"}`           |
| `POST /search`      | Semantic search over the ingested knowledge corpus |
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
│   ├── api/                 # POST /search endpoint and dependency wiring
│   ├── ingestion/           # Document loading, chunking and ingestion CLI
│   ├── embeddings/          # Local embedding service (FastEmbed)
│   ├── db/                  # PostgreSQL + pgvector persistence
│   └── retrieval/           # Semantic retrieval service
├── data/
│   └── knowledge/           # Synthetic runbooks and incident reports
├── migrations/              # Alembic migrations
├── tests/                   # Unit and API tests
│   └── integration/         # Database integration tests (PostgreSQL + pgvector)
├── scripts/                 # Manual verification scripts
├── docs/
│   ├── ai/                  # Project context and development workflow notes
│   ├── database.md          # Database, migrations and integration testing
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
- automated unit and database integration tests
- Ruff and mypy quality checks
- GitHub Actions CI

Not implemented yet and planned for later development:

- LLM integration and answer generation
- prompt construction
- RAG context assembly
- grounded answer generation
- answer citations
- insufficient-context answer guardrails
- conversational memory
- agents
