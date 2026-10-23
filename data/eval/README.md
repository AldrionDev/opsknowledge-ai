# RAG Evaluation Dataset

`questions.jsonl` is a small, fixed, inspectable set of questions for measuring answer quality on the synthetic knowledge corpus in [`../knowledge/`](../knowledge/README.md). Every question and label is written from that corpus only; no real or external operational data is used.

The dataset is data only. Running it against the answer service and computing metrics is the job of the evaluation harness (issue #40), which also computes the dataset version or content hash. The file itself carries no timestamps or version field.

Each record was checked against the corpus documents and is validated mechanically by the tests below. The dataset annotations have received human editorial approval for use as a heuristic evaluation reference. This approval does not make keyword-based labels verified factual ground truth.

## Categories

| Category | Count | Meaning | Expected outcome |
| -------- | ----- | ------- | ---------------- |
| `answerable` | 10 | A specific answer exists in the corpus. | An answer that cites an expected source and contains every required keyword. |
| `unanswerable` | 7 | A plausible operational question that no corpus document answers. | No answer (`insufficient_context`). |
| `misleading` | 7 | A question with a false premise: a wrong root cause, a near-miss incident, a wrong service or symptom, an invented remediation, or something that does not exist. | Either a correction of the premise that cites an expected source, or no answer. Never content that confirms the premise. |

Total: 24 questions. The answerable questions together cover all eight corpus documents.

Examples:

- **answerable** `answerable-08`: "Which configuration value caused payments-gateway pods to crash during INC-2025-004?" The incident report names `RETRY_BACKOFF_MS` and the invalid value `250ms`.
- **unanswerable** `unanswerable-04`: "Which Node.js version did the storefront 3.2.0 release upgrade to?" INC-2025-002 mentions the runtime upgrade but no version.
- **misleading, correct the premise** `misleading-02`: "Which keep-alive timeout change fixed the search-api 502 errors in INC-2025-003?" INC-2025-003 states that the keep-alive timeout was unrelated; the cause was `OOMKilled` pods. Answers that borrow the INC-2025-002 fix (`65 seconds`, `headersTimeout`) are forbidden.
- **misleading, decline** `misleading-06`: "What was the root cause of INC-2025-007, the checkout-service DNS outage?" No such incident exists, so there is nothing to correct from.

A misleading record with non-empty `expected_sources` asks for a correction of its premise. A misleading record with empty `expected_sources` must be declined, so any answer to it counts as a false answer.

## Record fields

Each line of `questions.jsonl` is one JSON object with exactly these fields:

| Field | Type | Rules |
| ----- | ---- | ----- |
| `id` | string | `<category>-NN`, for example `misleading-03`; unique and stable. Records are ordered by category and number. |
| `category` | string | `answerable`, `unanswerable`, or `misleading`. |
| `question` | string | Non-blank. |
| `expected_sources` | list of strings | Corpus documents that support the answer or the correction. Non-empty for `answerable`, empty for `unanswerable`, either for `misleading` (see above). |
| `required_keywords` | list of strings | Non-empty for `answerable`, empty for `unanswerable`, optional for `misleading` with expected sources. See [Scoring contract](#scoring-contract) for how each category uses them. |
| `forbidden_keywords` | list of strings | Non-empty for `misleading`, empty otherwise. |
| `notes` | string | Non-blank. The expected behavior (answer, decline, or correct the premise), the supporting section, and known heuristic risks. |

Keyword items are non-blank, have no leading or trailing whitespace, and are unique within a list (ignoring case).

## Source paths

`expected_sources` uses the same path format as `KnowledgeDocument.source_path` and the `source_path` returned by retrieval: relative to the corpus root, with `/` separators and no `data/knowledge/` prefix.

```text
incidents/inc-2025-004-payments-gateway-crashloopbackoff-config-error.md
```

Every path must be a document that the corpus loader actually loads.

## Keyword matching and its limits

Keywords are matched as **case-insensitive substrings** of the answer text. The tests enforce rules that keep the labels satisfiable and fair:

- every required keyword appears in at least one expected source;
- no required or forbidden keyword appears in the question, so repeating or negating the question cannot satisfy or fail a check;
- no forbidden keyword appears in the record's expected sources, so a faithful correction is not penalized for quoting them;
- forbidden keywords of a misleading record without expected sources appear nowhere in the corpus.

### Scoring contract

How the evaluation harness (#40) is specified to use the labels:

- **answerable:** heuristic correctness requires an answer that cites an expected source and contains **all** `required_keywords`.
- **unanswerable:** any answer is a false answer; the keyword lists are empty and unused.
- **misleading:** an answer is a heuristic false answer when it contains any `forbidden_keywords` or does not cite a supporting expected source. A declined answer and a source-supported correction are accepted. The optional `required_keywords` of a misleading record are descriptive hints of the expected correction for manual review; they are not a scoring condition of the specified misleading metric.

None of these checks can guarantee semantic correctness.

Keyword checks are a **heuristic proxy for correctness, not a measure of factual accuracy**. They do not understand meaning:

- **False negatives:** a correct answer can paraphrase a keyword, for example `1 GiB` instead of `1Gi`, or `rolled back the configuration` instead of `revert` and `ConfigMap`.
- **False positives on required keywords:** a keyword can appear in a wrong statement (`the cause was not OOMKilled`), and short or generic keywords such as `65`, `restart`, `command`, or `disk` can match by accident.
- **False positives on forbidden keywords:** a correct answer that explicitly names and rejects a near-miss cause, for example by contrasting INC-2025-003 with the `65 seconds` fix of INC-2025-002, is penalized.
- **Coverage:** a forbidden list names likely fabrications, not every possible one. An invented detail that is not listed passes.

The dataset is small, so any figure computed from it is indicative, not statistical. Questions whose answers can only be judged by reading them should be reviewed manually.

## Running the integrity tests

The tests are offline and need no database, embeddings, LLM credentials, or network access:

```bash
uv run pytest tests/test_eval_dataset.py
```

They also run as part of `uv run pytest -m "not integration"`.
