# Database

PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
stores document chunks, their source metadata and their embeddings. This
document covers the database foundation: local environment, configuration,
schema and migrations, and the chunk repository that stores, reads, replaces and
removes chunks.

## Components

| Path | Purpose |
|---|---|
| `compose.yaml` | Local PostgreSQL + pgvector service (`pgvector/pgvector:0.8.7-pg17-trixie`) |
| `.env.example` | Safe local-development values for `DATABASE_URL` and `TEST_DATABASE_URL` |
| `app/db/config.py` | Reads and validates `DATABASE_URL` |
| `app/db/session.py` | `create_db_engine`, `create_session_factory`, `session_scope` |
| `app/db/models.py` | SQLAlchemy base and `DocumentChunkRecord` |
| `app/db/verification.py` | `verify_embedding_dimension` |
| `app/ingestion/chunk_repository.py` | `ChunkRepository` contract and `StoredChunk` |
| `app/db/chunk_repository.py` | `SqlAlchemyChunkRepository`, the PostgreSQL implementation |
| `alembic.ini`, `migrations/` | Alembic configuration and migrations |

Engines and sessions are created explicitly; nothing connects at import time.
The API (`/health` included) does not need a database.

## Local workflow

```bash
cp .env.example .env                                  # local, ignored by Git
docker compose up -d --wait                           # starts PostgreSQL, waits for the health check
uv run --env-file .env alembic upgrade head           # applies the migrations
docker compose down                                   # stops the database; data is kept
```

- The database files live in the named Docker volume `opsknowledge_pgdata`, so
  data survives `docker compose down` and restarts.
- The credentials in `compose.yaml` and `.env.example` are development-only
  values and must match each other. Never put real secrets in either file.
- The database listens on `127.0.0.1:5432`. If that port is taken, change the
  host port in `compose.yaml` and in the URLs of your `.env`.
- `.env` is not loaded automatically: pass `--env-file .env` to `uv run`, or
  export `DATABASE_URL` yourself. A missing or invalid `DATABASE_URL` fails with
  a `DatabaseConfigError`; the URL must use the `postgresql+psycopg` driver.
- `export UV_ENV_FILE=.env` makes every `uv run` load `.env`, so a bare
  `uv run alembic upgrade head` works. It also sets `TEST_DATABASE_URL` for
  `uv run pytest`; see [Tests](#tests) for the effect.
- Inspect the schema: `docker compose exec db psql -U opsknowledge -d opsknowledge -c '\d document_chunks'`.

## Schema

Table `document_chunks`, created by migration `0001`. It mirrors
`app.ingestion.models.DocumentChunk` plus the embedding.

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | `text` | Primary key |
| `document_id` | `text` | Not null |
| `chunk_index` | `integer` | Not null, `CHECK (chunk_index >= 0)` |
| `source_path` | `text` | Not null |
| `document_type` | `text` | Not null (`runbook`, `incident`, ...) |
| `content` | `text` | Not null, no length limit |
| `embedding` | `vector(384)` | Not null |

`UNIQUE (document_id, chunk_index)` also serves lookups by `document_id`, so
there is no separate index. There is no vector index yet; similarity search and
its index tuning belong to the retrieval work.

pgvector stores `float32`; vectors read back can differ slightly from the
Python `float` values that were written.

## Embedding dimension

The relationship is explicit and checked:

- the embedding model `BAAI/bge-small-en-v1.5` produces 384 dimensions
  (`MODEL_DIMENSION` in `app/embeddings/fastembed_service.py`);
- the model definition fixes `EMBEDDING_DIMENSION = 384` in `app/db/models.py`;
- migration `0001` declares `vector(384)` as a literal.

The migration does not import application code or the embedding configuration; a
test enforces that. A test asserts `EMBEDDING_DIMENSION == MODEL_DIMENSION`.
`verify_embedding_dimension(engine, expected)` compares a dimension against the
real database column and raises `EmbeddingDimensionMismatchError`; call it with
`EmbeddingService.dimension` before relying on persistence. The `vector` type
also rejects vectors of the wrong length on insert.

To change the dimension (for example with another embedding model):

1. add a **new** migration that alters the column; never edit an existing one;
2. update `EMBEDDING_DIMENSION`;
3. re-embed and re-store all chunks, because vectors of different models or
   dimensions are not comparable.

## Chunk repository

Application code depends on the `ChunkRepository` protocol
(`app/ingestion/chunk_repository.py`), which exposes no SQLAlchemy types.
`SqlAlchemyChunkRepository(create_session_factory(engine))` implements it.

| Operation | Behavior |
|---|---|
| `add_chunks(chunks, embeddings)` | Stores new chunks; `embeddings[i]` belongs to `chunks[i]`. Existing `chunk_id` or `(document_id, chunk_index)` is an error. |
| `get_chunk(chunk_id)` | Returns a `StoredChunk` (`DocumentChunk` plus `embedding: list[float]`), or `None` if unknown. |
| `replace_document_chunks(document_id, chunks, embeddings)` | Deletes all chunks of the document and stores the new set in one transaction. |
| `delete_document_chunks(document_id)` | Removes all chunks of the document; an unknown document is a no-op. |

- Every operation is one transaction. A batch is written with one `INSERT`
  execution instead of one statement per chunk; SQLAlchemy may split a very
  large batch into several statements inside the same transaction. A failed
  replacement rolls back completely: the previous chunks stay, no new chunk is
  left behind, and other documents are never touched.
- Replacing a document repeatedly with the same chunks is idempotent.
  `replace_document_chunks(document_id, [], [])` is equivalent to
  `delete_document_chunks(document_id)`.
- A chunk/embedding count mismatch, an embedding that does not have
  `EMBEDDING_DIMENSION` elements, or (on replace) a chunk of another document
  raises `ValueError` before the database is accessed. These checks are the
  contract's own `require_embeddings` and `require_chunks_of_document`, so every
  implementation reports them identically; the expected dimension is passed in
  by the implementation.
- Database errors (for example `IntegrityError`) are not wrapped; they
  propagate to the caller. Concurrent replacements of the same document are not
  serialized: one of them may fail with an `IntegrityError`.
- Similarity search is not part of the repository.

## Migrations

```bash
uv run alembic upgrade head       # apply
uv run alembic downgrade base     # revert; the vector extension is kept
```

Migrations are hand-written and self-contained. The schema is managed only by
Alembic, never by `create_all()`.

## Tests

Which tests run depends on the environment, not on the command:

| Environment | `uv run pytest` |
|---|---|
| `TEST_DATABASE_URL` not set (default, CI) | Unit tests run; integration tests are **skipped** |
| `TEST_DATABASE_URL` set (for example via `UV_ENV_FILE=.env`) | Unit and integration tests run; the database must be up, otherwise the integration tests error |

Use markers to be explicit, regardless of the environment:

```bash
uv run pytest -m "not integration"                          # unit tests only
docker compose up -d --wait
uv run --env-file .env pytest -m integration -rs -v         # integration tests only
```

Integration tests (marker `integration`, `tests/integration/`) use a real
PostgreSQL + pgvector server. When `TEST_DATABASE_URL` is unset they are skipped
and the skip reason says so.

Each test creates its own empty database named `opsknowledge_test_<32 hex
characters>` on the `TEST_DATABASE_URL` server, migrates it from scratch, and
drops it afterwards.

Database safety:

- **The guarantee is the name check.** The create and drop helpers
  (`tests/integration/temp_database.py`) accept only names matching
  `opsknowledge_test_<32 hex characters>`, and a database can only be dropped if
  the same process created it through `create_database`. A developer database,
  including one that merely starts with `opsknowledge_test_`, is refused before
  any connection is made. The migrations run against the temporary database only,
  because `DATABASE_URL` is overridden for the duration of the test.
- `TEST_DATABASE_URL` is used only as an administrative connection to create and
  drop those databases. As an additional warning, the tests fail when
  `TEST_DATABASE_URL` and `DATABASE_URL` name the same database (compared by
  database name only, and only if `DATABASE_URL` is set). This check does not
  replace the name check.

A run only counts as having verified the database when it reports zero skipped
tests; with `TEST_DATABASE_URL` unset the integration tests are skipped, not
passed. GitHub Actions does not run them yet.
