# OpsKnowledge AI

OpsKnowledge AI is being developed as an enterprise RAG knowledge assistant
for technical documentation, operational runbooks, incident reports, and
operational knowledge.

The repository currently provides the API and engineering foundation. RAG
functionality is planned, not yet implemented. See
[Development status](#development-status) for details.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git

The project is pinned to Python 3.12 through `.python-version`. uv uses the
required Python version for the project environment.

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

## Running the API

```bash
uv run uvicorn app.main:app --reload
```

The API is served at <http://127.0.0.1:8000>.

| Endpoint        | Description                                |
| --------------- | ------------------------------------------ |
| `/health`       | Health check, returns `{"status": "ok"}`   |
| `/docs`         | Interactive API documentation (Swagger UI) |
| `/openapi.json` | OpenAPI schema                             |

## Running tests

```bash
uv run pytest
```

## Code quality

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

GitHub Actions runs the same checks together with the tests on pull requests
to, and pushes to, `main`.

## Project structure

```text
.
├── app/
│   └── main.py              # FastAPI application and /health endpoint
├── tests/
│   └── test_health.py       # Health endpoint tests
├── docs/
│   └── ai/                  # Project context and development workflow notes
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── pyproject.toml           # Dependencies, Ruff and mypy configuration
├── uv.lock                  # Locked dependency versions
└── .python-version          # Python version used by uv
```

## Development status

The project is under active development.

Currently implemented:

- FastAPI application foundation
- `GET /health` endpoint
- automated tests
- Ruff and mypy quality checks
- GitHub Actions CI

Not implemented yet and planned for later development:

- document ingestion
- embeddings
- vector search
- persistent storage
- RAG
- LLM integration
