# Evaluation

The evaluation harness (`app/evaluation/`) runs the questions of
[`data/eval/questions.jsonl`](../data/eval/README.md) through the grounded
`AnswerService`, records retrieval for every question, classifies each answer
outcome, computes retrieval and heuristic answer metrics, and writes a JSON and a
Markdown report.

CI verifies the **harness implementation** only, with fake components and no
network access. It does not measure RAG quality. A quality baseline comes only
from a manual, bounded `--live` run that a person starts deliberately.

## Baseline status

**No live baseline has been performed yet.** No real model output, metric value
or manual verdict exists for this project at the moment. This section will
record the baseline once a separately authorized live run has been made: the
report files, the printed cost bound, the database preparation that was used and
the human manual-review verdicts.

## Modes

```bash
uv run python -m app.evaluation             # describe only
uv run python -m app.evaluation --dry-run   # synthetic and offline
uv run python -m app.evaluation --live --max-requests 24 --max-retries 1 --max-output-tokens 1024
```

| Mode | What it does | Needs |
| ---- | ------------ | ----- |
| none | Loads the dataset, validates the bounds, prints the dataset size and the worst-case output-token bound, exits with status 2. Nothing is run. | Repository files only |
| `--dry-run` | Synthetic run: SHA-256 hash-ranked retrieval over the repository corpus files and a scripted fake LLM. The report is marked **SYNTHETIC / NOT A BASELINE**. | Repository files only: no database, `DATABASE_URL`, embedding model, credentials or network |
| `--live` | Real run: PostgreSQL + pgvector retrieval, FastEmbed query embeddings and the Anthropic chat model, after a preflight. | See [Live run](#live-run-separately-authorized) |

Only the `--live` branch builds real database, embedding or provider
components. No environment variable can enable it.

The synthetic dry-run checks the harness end to end: every outcome class appears,
and the report pipeline is exercised. Its retrieval ranking carries no semantic
signal and its LLM outcomes follow a fixed script, so **none of its figures
measure RAG quality**.

Options:

| Option | Default | Range |
| ------ | ------- | ----- |
| `--max-requests` | dataset size (24) | 1 to 30 |
| `--max-retries` | 1 | 0 to 2 |
| `--max-output-tokens` | 1024 | 1 to 2048 |
| `--limit` | 5 (as `POST /ask`) | 1 to 50; the `k` of hit@k |
| `--output-dir` | `evaluation-output/<UTC time>-<mode>/` (gitignored) | must not exist |

Exit status:

| Status | Meaning |
| ------ | ------- |
| 0 | Run completed |
| 1 | Run stopped early (partial report written), or the report could not be written (the unsaved report is then printed to stdout as `{"persisted": false, "report": ...}`) |
| 2 | Nothing was run |
| 130 | Interrupted (partial report written) |

The harness reads configuration from the process environment only. It never
reads or modifies `.env`.

## Outcome classes

Each evaluated question has exactly one outcome:

| Outcome | Meaning |
| ------- | ------- |
| `answered` | Validated answer with citations |
| `insufficient_context` | Genuine lack of knowledge. Reason: `no_retrieval`, `empty_context` or `model_declined` |
| `invalid_output` | The model output was rejected. Code: `malformed_output`, `unknown_citation`, `missing_evidence` or `inconsistent_response`. Always a failure, never an abstention |
| `provider_error` | The provider call failed (fixed message), classified `transient` or `nonrecoverable` |

A question that was not asked because the run stopped first is marked
`not_evaluated`. This is not an outcome, and it is excluded from every metric.

Each question costs at most one LLM invocation, with no application-level retry.
Questions without retrieved chunks or without assembled context never invoke the
model.

## Metrics

Every rate is reported as `count / n`. Its denominator is the number of
**evaluated** questions of the category, and the rate is `null` when `n` is 0.

### Retrieval (answerable questions, every outcome)

Retrieval metrics are computed from the ranked chunk list that the retrieval
service returned for the question. The harness records it with a wrapper and
makes no extra retrieval, embedding or model calls.

- A chunk is **relevant** when its `source_path` is one of the question's
  `expected_sources`. Matching is at document level.
- The **rank** is the position of the first relevant chunk in the list **as
  returned**, without deduplication. For the ranked chunks A1, A2, A3, B1 with B
  expected, the rank is 4, not 2.
- `hit@k` is 1 when the rank is at most `k`, and `RR@k` is `1 / rank`, else 0.
  The report gives the mean of each over the answerable questions (hit rate and
  MRR@k). `k` is the run's `--limit`.

### Answer heuristics

Keyword checks are case-insensitive substring matches against the answer text,
with the trailing ` [n]` citation marker of each line removed. Citation checks
compare the cited source paths with `expected_sources`. These checks are
**heuristic proxies, not measures of factual correctness**.

- **Answerable:** `heuristic_correct` = answered, an expected source cited, and
  every required keyword present. Every other outcome is a failure, including
  `invalid_output` and `provider_error`.
- **Unanswerable:** abstention = genuine `insufficient_context`; false answer =
  `answered`. `invalid_output` and `provider_error` are separate fail-closed
  buckets, not abstentions.
- **Misleading:** false answer = answered with a forbidden keyword, or answered
  without citing an expected source. When a record has no expected sources, every
  answer is false. A decline and a source-supported correction are accepted. The
  misleading `required_keywords` are not scored.
- **Global:**
  - `invalid_output` rate by code and the `provider_error` rate;
  - the distribution of citation counts over answered questions;
  - per category, the similarity scores of the top-ranked chunk and of all
    retrieved chunks. These scores are **descriptive only. No similarity
    threshold is derived, applied or shipped.**

## Cost bounds

The absolute ceilings are code constants (`app/evaluation/guards.py`):
30 requests, 2 transport retries and 2048 output tokens per attempt. The printed
bound is

```text
max_requests x (1 + max_retries) x max_output_tokens
```

It is 49,152 output tokens at the defaults and 184,320 at the ceilings. It
bounds **output tokens only and is not a total cost bound**: input tokens
(instructions, schema, the assembled context of up to 8000 characters and the
question, re-sent on every retry) and provider pricing are not included.

The report keeps these quantities apart:

| Quantity | Source |
| -------- | ------ |
| Logical LLM invocations | Counted by the request limiter; at most `max_requests` |
| Possible transport attempts | `invocations x (1 + max_retries)`; actual SDK retries are not observable |
| Reported token usage | `usage_metadata` of completed responses, collected by a callback |
| Unknown usage | Invocations without a completed response (provider errors, timeouts) and retried attempts. Reported as unknown, never estimated, and possibly billed. Not applicable to the synthetic dry-run, which makes no provider request |

The run stops early, keeping every completed result, and writes a partial report
when:

- the request cap refuses an invocation (`max_requests`);
- a provider failure is **nonrecoverable** (`provider_nonrecoverable`):
  authentication, permission, model not found, invalid request, context
  overflow, HTTP 413 or 422. These would repeat for every question;
- three consecutive LLM invocations fail with **transient** errors
  (`provider_consecutive_errors`): rate limit, timeout, connection, server or
  overload errors;
- any other error occurs, for example in the database or the embedding model
  (`unexpected_error`). It is reported by exception type only.

## Report and reproducibility

Each run writes `report.json` and `report.md` into a new directory. An existing
directory is never overwritten. The files are written to a staging directory and
renamed into place in one step, so a failed write leaves no complete-looking
report behind. The Markdown is rendered from the JSON and can be regenerated
from it.

The report records:

- **Run:** mode, synthetic flag, start and finish time, `runs: 1`, completion,
  the abort reason and the question where the run stopped.
- **Code:** commit SHA, the Git tree ids of `app/` and `data/knowledge`, and
  whether the worktree was clean. The project squash-merges, so a branch commit
  may not survive on `main`. The tree ids remain verifiable there with
  `git rev-parse <commit>:app`.
- **Dataset:** SHA-256 of the raw file bytes, the version derived from it
  (`sha256:<12 hex>`) and the counts.
- **Corpus:** fingerprint, document count and chunk count of the repository
  corpus.
- **Database** (live only): name, migration revision, verified corpus
  fingerprint and counts, the read-only flag, and the preparation note.
- **Retrieval:** implementation, `limit` = `k`, embedding model and FastEmbed
  version.
- **Context and prompt:** context character budget, prompt version and prompt
  hash.
- **LLM:** model id, `json_schema` structured output, "no sampling parameters",
  output-token bound, retries, timeout, and the `langchain-core`,
  `langchain-anthropic` and `anthropic` versions.
- **Usage:** bounds, ceilings, the worst-case bound and the usage figures above.
- **Results:** the metrics, one record per question (retrieved chunks, expected
  rank, outcome, answer, citations, heuristic facts), a manual-review sample and
  the known limitations.

Manual review: the sample contains every answered misleading question, plus up
to three answered answerable and three answered unanswerable questions, chosen
deterministically by question-id hash. The harness assigns no verdicts. Verdicts
are recorded by a person in the baseline section of this document.

## Live run (separately authorized)

A live run sends prompts with retrieved corpus content to the Anthropic API and
incurs provider charges. Run it only when it has been explicitly authorized.

### Database preparation

Use a **separately provisioned, empty evaluation database**, never the
development database. Ingestion does not remove chunks of deleted or renamed
documents, so a reused database can hold stale chunks.

1. Create an empty database on a PostgreSQL + pgvector server with your own
   tooling, for example `CREATE DATABASE opsknowledge_eval_<yyyymmdd>;` from your
   SQL client. Alternatively, use a separate throwaway pgvector container with
   its own volume.
2. In your own shell, point `DATABASE_URL` at it:
   `export DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:<port>/opsknowledge_eval_<yyyymmdd>`.
3. Apply the migrations: `uv run alembic upgrade head`.
4. From the clean commit to be evaluated, ingest the repository corpus:
   `uv run python -m app.ingestion`. This also downloads and caches the FastEmbed
   model.

The harness never creates, migrates, ingests, truncates or drops anything.

The `.env` guarantee covers the harness process only. Other tools you use to
prepare the database are outside it: for example, `docker compose` reads a
`.env` file in the project directory for variable interpolation.

### Preflight and run

```bash
export ANTHROPIC_API_KEY=...        # your own key, in your own shell
uv run python -m app.evaluation                # review the printed bound
uv run python -m app.evaluation --dry-run      # optional offline harness check
uv run python -m app.evaluation --live --max-requests 24 --max-retries 1 --max-output-tokens 1024
```

Before any model request, `--live` refuses to run (status 2) unless:

1. no endpoint-redirection or tracing variable is set (`ANTHROPIC_BASE_URL`,
   `LANGSMITH_TRACING`, …);
2. the Git worktree is clean, untracked files included;
3. `DATABASE_URL` is set and is a valid URL, and the read-only engine can be
   created (a failure is reported by exception type only, never with the URL);
4. the embedding column dimension matches the model;
5. the stored chunks equal the repository corpus chunk by chunk, compared by
   `chunk_id`. Missing, unexpected (stale or unrelated) and changed chunks
   (metadata or content hash) are listed by source path;
6. the LLM configuration (`ANTHROPIC_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`)
   is valid. The CLI bounds replace `LLM_MAX_RETRIES` and `LLM_MAX_OUTPUT_TOKENS`.

The harness opens every database transaction read-only
(`default_transaction_read_only=on`), so PostgreSQL rejects any write from the
harness process. This is an application safeguard against mistakes, **not a
replacement for a database role with `SELECT`-only privileges**.

After the run, record in the pull request the printed bound and the report
files, and add the summary and the manual verdicts to the baseline section
above.

## Known limitations

- A single run over 24 questions is indicative, not statistical.
- Actual model outputs are nondeterministic; repeated live runs can differ.
- Keyword scoring is heuristic. Paraphrases cause false negatives, keywords in
  wrong or negated statements cause false positives, and forbidden lists are not
  exhaustive (see [data/eval/README.md](../data/eval/README.md)).
- Synthetic dry-run results do not measure RAG quality.
- Database verification establishes the identity, metadata and content of the
  stored chunks. It does not establish which model computed the stored embedding
  vectors.
- Provider usage of failed, timed-out or retried attempts is unknown and may be
  billed. Input tokens are bounded only indirectly, by the fixed context budget.
- Wall-clock time is not bounded; the request timeout is recorded.
- `--live` loads the FastEmbed model. If it is not cached yet, FastEmbed
  downloads it from the Hugging Face Hub. Ingestion caches it, so run it first
  (see [docs/embeddings.md](embeddings.md)).
- An interrupt (`Ctrl+C`) during the run writes a partial report. An interrupt
  while the report itself is being written leaves no report.

## Tests

- **Offline** (`uv run pytest -m "not integration"`, part of CI):
  - dataset loading;
  - fakes, including evidence extraction with forged source markers;
  - bounds, the request cap and provider-failure classification;
  - retrieval and usage recording;
  - every outcome class and code, and the stopping rules;
  - rank, hit@k, MRR and the heuristic metrics;
  - report schema and synthetic labels, no-clobber writes;
  - the CLI, including a dry-run with every real factory patched to fail and
    network access blocked, and the live preflight with fake components.
- **PostgreSQL + pgvector** (`-m integration`, see
  [docs/database.md](database.md#tests)): the harness over real retrieval with
  fake embeddings and a fake LLM, corpus verification (missing, unexpected,
  changed), the rejection of writes by the read-only engine, and the live
  composition root with fake models.
