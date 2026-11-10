# Answer Generation

This document describes how `POST /ask` produces grounded answers from the ingested
knowledge corpus: the pipeline, grounding checks, outcome policy, context assembly,
configuration, safety, and known limitations.

## Architecture

The answering pipeline uses a small set of LangChain components and explicit
application code. The boundary is intentional: LangChain provides the prompt
template, the Anthropic structured-output chain (`json_schema` mode), and the
transport-layer error types. Everything else is application code:

- **context assembly** — ranking, deduplication, budget fitting, source numbering
  (`app/answering/context.py`)
- **prompt construction** — filling the system and human templates with the numbered
  context and the question (`app/answering/prompt.py`)
- **grounding validation** — deterministic structural and evidence-presence checks on
  the model's draft before it becomes an answer (`app/answering/validation.py`)
- **outcome mapping** — translating validation and provider errors into typed HTTP
  responses (`app/answering/service.py`, `app/api/ask.py`, `app/main.py`)

## Pipeline

```text
question
 -> RetrievalService.retrieve(question, limit)
 -> no chunks          -> Answer(insufficient_context / no_retrieval)
 -> assemble_context
 -> empty context      -> Answer(insufficient_context / empty_context)
 -> chain.invoke       (exactly once; no application-level retry)
 -> validate_answer_draft
 -> valid              -> Answer(answered, citations)
 -> invalid            -> InvalidModelOutputError  (HTTP 502)
```

The model is called at most once per request. Transport retries within that single
call are managed by the SDK (see [Configuration](#configuration)).

## Grounding levels

Grounding is enforced by `validate_answer_draft` (`app/answering/validation.py`).
Checks run in a fixed order; the first failure rejects the whole draft (fail-closed).

### Level 1 — Structural validity

Each claim's `source_id` must exist in the assembled context. A reference to a
source that was not retrieved raises `UNKNOWN_CITATION`.

### Level 2 — Evidence presence

The claim's `evidence_quote` must be a contiguous substring of the cited source's
content after normalization (see [Evidence normalization](#evidence-normalization)).
The normalized quote must be at least 20 Unicode code points (`MIN_EVIDENCE_QUOTE_LENGTH`).

A quote that passes Level 2 proves only that the model did not invent it. It does
not prove that the claim correctly interprets or follows from the quoted text.

### Level 3 — Semantic support (not enforced at runtime)

Whether a claim follows from its evidence quote is not verified automatically.
The validation module explicitly does not enforce semantic entailment:

> "A quote that is present in the source proves only that the model did not invent
> the quote; it does not prove entailment. The claim may overstate, negate or
> misread the quoted passage."
> — `app/answering/validation.py`

Semantic quality is assessed heuristically by the evaluation harness (keyword
matching and citation-presence checks) and by human spot-review of a sample.
Neither constitutes an automated semantic-correctness guarantee. See
[docs/evaluation.md](evaluation.md).

## Evidence normalization

Normalization is applied identically to the model's `evidence_quote` and to the
source's content before the substring check (`validation.py:131–142`):

1. Unicode NFC normalization (canonical decomposition + canonical composition).
2. All Unicode whitespace runs (tabs, newlines, non-breaking spaces, etc.) collapsed
   to a single ASCII space; leading and trailing whitespace trimmed. Implemented via
   `" ".join(unicodedata.normalize("NFC", text).split())`.

Normalization is applied to both sides, so whitespace differences between the
model's quote and the stored chunk content do not cause failures.

**Failure cases for `MISSING_EVIDENCE`:**

- Normalized quote is shorter than 20 Unicode code points (excludes single words,
  short identifiers, and error codes that appear in almost any source).
- Normalized quote is not a contiguous substring of the normalized source content.
- Case mismatch — matching is case-sensitive; no case folding is applied.
- Quote references text from a source that was not in the assembled context.

The `citation.evidence_quote` stored in the returned `Answer` is the **normalized**
quote, which may differ from the raw model output in whitespace, but is guaranteed
to be a substring of the normalized source content.

## Outcome policy

Both `answered` and `insufficient_context` are normal results returned as HTTP 200.
Missing knowledge in the corpus is expected behavior, not an error.

| Outcome | HTTP | `code` | Trigger |
| --- | --- | --- | --- |
| `answered` | 200 | — | Validated answer with citations |
| `insufficient_context` | 200 | — | No knowledge: `no_retrieval`, `empty_context`, or `model_declined` |
| invalid model output | 502 | `malformed_output` | Draft could not be parsed or does not fit the schema |
| invalid model output | 502 | `unknown_citation` | Model cited a `source_id` not in the context |
| invalid model output | 502 | `missing_evidence` | Quote absent from source, or shorter than 20 code points |
| invalid model output | 502 | `inconsistent_response` | `insufficient` flag contradicts claims, or claim text has line breaks or citation markers |
| provider failure | 502 | `llm_unavailable` | Provider or transport error |
| missing configuration | 503 | `llm_not_configured` | `ANTHROPIC_API_KEY` absent or LLM configuration invalid |

Error responses have the shape `{"detail": "<fixed message>", "code": "<code>"}`.
Messages are fixed per code; model-generated text is never echoed in error responses.

## Context assembly

Context assembly (`app/answering/context.py`) runs before the model is called:

1. **Sort** by `similarity_score` descending, then `chunk_id` ascending.
2. **Skip** blank-content chunks (they consume no budget and are never deduplicated).
3. **Deduplicate** by `chunk_id`, keeping the highest-ranked non-blank occurrence.
4. **Budget fit** — whole chunks only, in ranked order; a chunk that does not fit
   stops assembly. No chunk is truncated. A first chunk larger than the budget yields
   an empty context.
5. **Number** sources 1-based and contiguous.
6. **Render** as a header + content block per source, joined with a separator.

The budget is `DEFAULT_CONTEXT_CHARACTER_BUDGET = 8000` characters (Unicode code
points, the same unit as the chunker). This constant is not environment-configurable.
`len(context.text) <= 8000` always holds for the text passed to the prompt.

## Configuration

| Variable | Required | Default | Constraint |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes (for `/ask`) | — | Missing → HTTP 503 |
| `LLM_MODEL` | No | `claude-haiku-5-5` | — |
| `LLM_TIMEOUT_SECONDS` | No | `60` | Must be a positive finite float |
| `LLM_MAX_OUTPUT_TOKENS` | No | `1024` | Must be a positive integer |
| `LLM_MAX_RETRIES` | No | `2` | Must be a non-negative integer |

`LLM_MAX_RETRIES` is passed as `max_retries` to the LangChain `ChatAnthropic` model
(`app/llm/factory.py`). It controls SDK-managed transport retries within a single
application-level invocation. The service itself calls the chain exactly once per
request; there is no application-level retry on top of the SDK's.

No sampling parameters (`temperature`, `top_p`, `top_k`) are set or configurable.
Thinking tokens are not configured; they can consume part of the model's output-token
allowance, and output truncation may cause structured-output parsing failures.

A missing or invalid LLM configuration raises `LLMConfigurationError`, which the
application maps to HTTP 503 (`llm_not_configured`). The configuration error
message does not include the value of `ANTHROPIC_API_KEY`.

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to a real key. `.env` is
ignored by Git. Never commit real credentials. The `.env` file is not loaded
automatically; pass it explicitly with `uv run --env-file .env`.

## Safety

**Read-only boundary.** The answering pipeline performs no writes: no database
mutations, no file writes, no infrastructure commands, no tool execution.

**Untrusted context.** Retrieved chunk content comes from the ingested corpus and
is not sanitized before being passed to the model. It is treated as opaque text
in the prompt; the application never evaluates or executes it.

**Data sent to the Anthropic API.** The application passes the rendered prompt
(question + assembled context, up to 8000 characters of retrieved chunk text) and
the structured-output JSON schema to the Anthropic API via the LangChain SDK. The
application does not configure LangSmith or other observability integrations;
whether the installed SDK versions transmit additional metadata to external services
is not established by this application's code.

**No endpoint redirection.** The evaluation harness (`app/llm/preflight.py`) checks
for `ANTHROPIC_BASE_URL`, `LANGSMITH_TRACING`, and related variables before a live
evaluation run, but the API server (`POST /ask`) does not enforce their absence.

## Known limitations

- **Semantic entailment not verified** — a grounded answer may overstate, negate, or
  misread its evidence quote. Level 2 confirms presence, not correctness.
- **Context budget is 8000 characters** — chunks that do not fit are excluded in
  ranked order. A first chunk larger than the budget yields an empty context and an
  `insufficient_context / empty_context` response.
- **Case-sensitive evidence matching** — the model must quote the source text with
  the same casing (after whitespace normalization).
- **Single LLM invocation** — if the model returns an invalid output, the request
  returns HTTP 502 immediately. There is no application-level retry.
- **Transport retry count is not observable** — `LLM_MAX_RETRIES` retries are
  SDK-managed. Actual retry counts and their token usage are not reflected in the
  response.
- **Thinking tokens** — not explicitly configured, but may be used by the model,
  consuming part of `LLM_MAX_OUTPUT_TOKENS` and potentially causing truncation.
- **LangGraph, agents, MCP, conversational memory, and deployment are not
  implemented.**
