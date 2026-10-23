"""Evaluation report: one JSON document and a Markdown rendering of it.

`build_report` assembles the JSON-compatible report (schema version 1) from the
run result and the run's provenance. The sections and their keys are the same
for every run, except `database`, which is `{"used": false}` for a synthetic run.
A value that cannot be obtained is the string `"unavailable"` or
`"not applicable: ..."`, never an estimate.

`render_markdown` is a pure function of that JSON document, so the Markdown can
always be regenerated from the committed JSON. Model-generated answer text is
rendered JSON-escaped in indented code blocks, so it cannot inject Markdown,
control characters or terminal escape sequences.

`write_report` never overwrites: the target directory must not exist. Both files
are written to a staging directory next to it, which is then renamed into place
in one step. A failed write removes the staging directory, so no half-written,
apparently complete report is left behind.

Reports of `--dry-run` are marked synthetic in the JSON (`run.synthetic`, a
`synthetic` flag on every metrics block) and in the Markdown (a banner and every
metrics heading): their figures measure nothing about the real system.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.answering.context import DEFAULT_CONTEXT_CHARACTER_BUDGET
from app.answering.prompt import PROMPT_HASH, PROMPT_VERSION
from app.evaluation.dataset import Dataset, EvalCategory
from app.evaluation.guards import (
    MAX_CONSECUTIVE_PROVIDER_ERRORS,
    MAX_OUTPUT_TOKENS_CEILING,
    MAX_REQUESTS_CEILING,
    MAX_RETRIES_CEILING,
    WORST_CASE_NOTE,
    CostBounds,
)
from app.evaluation.metrics import (
    compute_metrics,
    expected_rank,
    question_heuristics,
    reciprocal_rank,
    relevant_chunk_count,
)
from app.evaluation.provenance import (
    CorpusChunk,
    GitIdentity,
    corpus_fingerprint,
    document_count,
)
from app.evaluation.runner import Outcome, QuestionResult, RunResult

SCHEMA_VERSION = 1
JSON_FILE = "report.json"
MARKDOWN_FILE = "report.md"

UNAVAILABLE = "unavailable"
NOT_APPLICABLE_FAKE = "not applicable: scripted fake"

SYNTHETIC_BANNER = (
    "SYNTHETIC RUN — NOT A BASELINE. Retrieval is a SHA-256 hash ranking over the "
    "repository corpus, not the embedding model, and every LLM outcome is scripted. "
    "No figure in this report measures the real system."
)
SYNTHETIC_HEADING = " — SYNTHETIC, not a measurement"

LIMITATIONS = (
    "A single run over a small dataset is indicative, not statistical.",
    "Actual model outputs are nondeterministic; repeated live runs can differ.",
    "Keyword-based figures are heuristic proxies, not measures of factual "
    "correctness: paraphrases cause false negatives, keywords inside wrong or negated "
    "statements cause false positives, and forbidden lists are not exhaustive.",
    "Similarity-score distributions are descriptive only; no threshold is derived "
    "or applied.",
    "Database verification checks chunk identity, metadata and content against the "
    "repository corpus; it does not establish which model computed the stored "
    "embedding vectors.",
    "Provider token usage is reported only for completed responses; usage of failed, "
    "timed-out or retried attempts is unknown and may still be billed.",
    "The worst-case output-token bound is not a total cost bound: input tokens and "
    "provider pricing are not included.",
)
SYNTHETIC_LIMITATION = (
    "This run is synthetic (--dry-run): retrieval and LLM outcomes are fakes, so no "
    "figure measures RAG quality."
)

MANUAL_REVIEW_RULE = (
    "Every answered misleading question, plus up to 3 answered answerable and up to "
    "3 answered unanswerable questions, chosen by ascending SHA-256 of the question "
    "id. The harness assigns no verdicts; human verdicts are recorded in "
    "docs/evaluation.md."
)
MANUAL_REVIEW_PER_CATEGORY = 3


class ReportWriteError(Exception):
    """Raised when the report cannot be written; the message names the path only."""


@dataclass(frozen=True)
class ReportInputs:
    mode: str
    synthetic: bool
    started_at: datetime
    finished_at: datetime
    dataset: Dataset
    bounds: CostBounds
    limit: int
    git: GitIdentity | None
    corpus: Mapping[str, CorpusChunk]
    database: dict[str, Any]
    retrieval: dict[str, Any]
    llm: dict[str, Any]
    run: RunResult


def build_report(inputs: ReportInputs) -> dict[str, Any]:
    run = inputs.run
    evaluated = sum(1 for result in run.results if result.evaluated)
    limitations = list(LIMITATIONS)
    if inputs.synthetic:
        limitations.insert(0, SYNTHETIC_LIMITATION)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "mode": inputs.mode,
            "synthetic": inputs.synthetic,
            "started_at": _timestamp(inputs.started_at),
            "finished_at": _timestamp(inputs.finished_at),
            "runs": 1,
            "complete": run.complete,
            "abort_reason": None
            if run.abort_reason is None
            else run.abort_reason.value,
            "abort_detail": run.abort_detail,
            "aborted_at_question": run.aborted_at_question,
            "evaluated_questions": evaluated,
            "not_evaluated_questions": len(run.results) - evaluated,
        },
        "code": _code(inputs.git),
        "dataset": {
            "path": inputs.dataset.path,
            "sha256": inputs.dataset.sha256,
            "version": inputs.dataset.version,
            "record_count": len(inputs.dataset.questions),
            "category_counts": inputs.dataset.category_counts(),
        },
        "corpus": {
            "fingerprint": corpus_fingerprint(inputs.corpus),
            "document_count": document_count(inputs.corpus),
            "chunk_count": len(inputs.corpus),
        },
        "database": inputs.database,
        "retrieval": {"limit": inputs.limit, "k": inputs.limit} | inputs.retrieval,
        "context": {"character_budget": DEFAULT_CONTEXT_CHARACTER_BUDGET},
        "prompt": {"version": PROMPT_VERSION, "hash": PROMPT_HASH},
        "llm": inputs.llm,
        "bounds": {
            "max_requests": inputs.bounds.max_requests,
            "max_retries": inputs.bounds.max_retries,
            "max_output_tokens": inputs.bounds.max_output_tokens,
            "ceilings": {
                "max_requests": MAX_REQUESTS_CEILING,
                "max_retries": MAX_RETRIES_CEILING,
                "max_output_tokens": MAX_OUTPUT_TOKENS_CEILING,
            },
            "worst_case_output_tokens": inputs.bounds.worst_case_output_tokens,
            "worst_case_note": WORST_CASE_NOTE,
            "max_consecutive_provider_errors": MAX_CONSECUTIVE_PROVIDER_ERRORS,
        },
        "usage": _usage(run, inputs.bounds, inputs.synthetic),
        "metrics": compute_metrics(run.results, inputs.limit, inputs.synthetic),
        "questions": [_question(result, inputs.limit) for result in run.results],
        "manual_review": {
            "selection_rule": MANUAL_REVIEW_RULE,
            "items": [
                _review_item(result) for result in select_manual_review(run.results)
            ],
        },
        "limitations": limitations,
    }


def select_manual_review(results: tuple[QuestionResult, ...]) -> list[QuestionResult]:
    """The deterministic manual-review sample; see `MANUAL_REVIEW_RULE`."""

    def answered(category: EvalCategory) -> list[QuestionResult]:
        selected = [
            result
            for result in results
            if result.question.category is category
            and result.outcome is Outcome.ANSWERED
        ]
        return sorted(
            selected, key=lambda r: hashlib.sha256(r.question.id.encode()).hexdigest()
        )

    return (
        answered(EvalCategory.MISLEADING)
        + answered(EvalCategory.ANSWERABLE)[:MANUAL_REVIEW_PER_CATEGORY]
        + answered(EvalCategory.UNANSWERABLE)[:MANUAL_REVIEW_PER_CATEGORY]
    )


def _timestamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _code(git: GitIdentity | None) -> dict[str, Any]:
    if git is None:
        return dict.fromkeys(
            ("commit_sha", "app_tree", "corpus_tree", "worktree_clean"), UNAVAILABLE
        )
    return {
        "commit_sha": git.commit_sha,
        "app_tree": git.app_tree,
        "corpus_tree": git.corpus_tree,
        "worktree_clean": git.worktree_clean,
    }


def _usage(run: RunResult, bounds: CostBounds, synthetic: bool) -> dict[str, Any]:
    responses = [usage for result in run.results for usage in result.usage]
    # A scripted fake makes no provider request, so nothing can be billed.
    attempts: int | str = NOT_APPLICABLE_FAKE
    unknown: int | str = NOT_APPLICABLE_FAKE
    if not synthetic:
        attempts = run.llm_invocations * (1 + bounds.max_retries)
        unknown = sum(1 for r in run.results if r.llm_invoked and not r.usage)
    return {
        "llm_invocations": run.llm_invocations,
        "max_possible_transport_attempts": attempts,
        "transport_retries": "not observable",
        "responses_with_usage": len(responses),
        "invocations_with_unknown_usage": unknown,
        "input_tokens": sum(u.input_tokens for u in responses)
        if responses
        else UNAVAILABLE,
        "output_tokens": sum(u.output_tokens for u in responses)
        if responses
        else UNAVAILABLE,
        "note": (
            "Token totals cover completed responses that reported usage only. Usage of "
            "failed, timed-out or retried attempts is unknown and is not estimated; "
            "such attempts may still be billed."
        ),
    }


def _question(result: QuestionResult, k: int) -> dict[str, Any]:
    question = result.question
    retrieved = result.retrieved
    rank = None
    if retrieved is not None and question.expected_sources:
        rank = expected_rank(retrieved, question.expected_sources, k)
    answer = result.answer
    return {
        "id": question.id,
        "category": question.category.value,
        "status": result.status.value,
        "outcome": None if result.outcome is None else result.outcome.value,
        "reason": None if result.reason is None else result.reason.value,
        "code": None if result.code is None else result.code.value,
        "provider_error": result.provider_error,
        "provider_failure_kind": (
            None
            if result.provider_failure_kind is None
            else result.provider_failure_kind.value
        ),
        "llm_invoked": result.llm_invoked,
        "retrieval": (
            None
            if retrieved is None
            else [
                {
                    "position": position,
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "document_type": str(chunk.document_type),
                    "similarity_score": round(chunk.similarity_score, 6),
                }
                for position, chunk in enumerate(retrieved, start=1)
            ]
        ),
        "expected_rank": rank,
        "hit_at_k": None
        if retrieved is None or not question.expected_sources
        else rank is not None,
        "reciprocal_rank": (
            None
            if retrieved is None or not question.expected_sources
            else round(reciprocal_rank(rank), 4)
        ),
        "relevant_chunks": (
            None
            if retrieved is None or not question.expected_sources
            else relevant_chunk_count(retrieved, question.expected_sources, k)
        ),
        "answer": None if answer is None else answer.answer,
        "citations": (
            []
            if answer is None
            else [
                {
                    "source_id": citation.source_id,
                    "source_path": citation.source_path,
                    "chunk_id": citation.chunk_id,
                }
                for citation in answer.citations
            ]
        ),
        "heuristic": question_heuristics(result),
        "usage": {
            "responses_with_usage": len(result.usage),
            "input_tokens": (
                sum(u.input_tokens for u in result.usage) if result.usage else None
            ),
            "output_tokens": (
                sum(u.output_tokens for u in result.usage) if result.usage else None
            ),
        },
    }


def _review_item(result: QuestionResult) -> dict[str, Any]:
    question = result.question
    answer = result.answer
    return {
        "id": question.id,
        "category": question.category.value,
        "question": question.question,
        "answer": None if answer is None else answer.answer,
        "cited_sources": []
        if answer is None
        else [c.source_path for c in answer.citations],
        "expected_sources": list(question.expected_sources),
        "required_keywords": list(question.required_keywords),
        "forbidden_keywords": list(question.forbidden_keywords),
        "heuristic": question_heuristics(result),
        "notes": question.notes,
    }


# --- Markdown -----------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report as Markdown; a pure function of the JSON document."""
    run = report["run"]
    suffix = SYNTHETIC_HEADING if run["synthetic"] else ""
    metrics = report["metrics"]
    lines = [f"# Evaluation report — {run['mode']}", ""]
    if run["synthetic"]:
        lines += [f"> **{SYNTHETIC_BANNER}**", ""]
    lines += [
        "Generated by `python -m app.evaluation`. The JSON report is the source of "
        "this page; keyword-based figures are heuristic.",
        "",
        "## Run",
        "",
        *_table(["Field", "Value"], [[key, value] for key, value in run.items()]),
        "",
        "## Configuration",
        "",
        *_table(
            ["Setting", "Value"],
            [
                [f"{section}.{key}", value]
                for section in ("code", "dataset", "corpus", "database", "retrieval")
                for key, value in report[section].items()
            ]
            + [["context.character_budget", report["context"]["character_budget"]]]
            + [[f"prompt.{key}", value] for key, value in report["prompt"].items()]
            + [[f"llm.{key}", value] for key, value in report["llm"].items()],
        ),
        "",
        "## Bounds and usage",
        "",
        *_table(
            ["Bound", "Value"],
            [
                [key, value]
                for key, value in report["bounds"].items()
                if key != "worst_case_note"
            ],
        ),
        "",
        f"Worst-case output tokens: {_worst_case_formula(report['bounds'])}. "
        f"{report['bounds']['worst_case_note']}",
        "",
        *_table(
            ["Usage", "Value"],
            [[key, value] for key, value in report["usage"].items() if key != "note"],
        ),
        "",
        report["usage"]["note"],
        "",
        f"## Outcomes{suffix}",
        "",
        *_outcome_table(metrics["global"]["outcomes_by_category"], report["questions"]),
        "",
        f"## Retrieval (answerable questions){suffix}",
        "",
        *_table(
            ["Metric", "Value"],
            [
                [
                    f"hit@{metrics['retrieval']['k']}",
                    _rate(metrics["retrieval"]["hit_at_k"]),
                ],
                [
                    f"MRR@{metrics['retrieval']['k']}",
                    _mrr(metrics["retrieval"]["mrr_at_k"]),
                ],
            ]
            + [
                [f"expected rank {rank}", count]
                for rank, count in metrics["retrieval"][
                    "expected_rank_histogram"
                ].items()
            ],
        ),
        "",
        "Rank is the position in the retrieved chunk list as returned, without "
        "deduplication; a chunk is relevant when its source path is an expected source.",
        "",
        f"## Answerable (heuristic){suffix}",
        "",
        *_rate_table(metrics["answerable"]),
        "",
        f"## Unanswerable{suffix}",
        "",
        *_rate_table(metrics["unanswerable"]),
        "",
        f"## Misleading (heuristic){suffix}",
        "",
        *_rate_table(metrics["misleading"]),
        "",
        "The two false-answer reasons can apply to the same answer.",
        "",
        f"## Global{suffix}",
        "",
        *_rate_table(
            {
                key: metrics["global"][key]
                for key in ("outcomes", "invalid_output", "provider_error")
            }
        ),
        "",
        *_table(
            ["Citation counts (answered)", "Value"],
            [
                [key, value]
                for key, value in metrics["global"][
                    "citation_count_distribution"
                ].items()
            ],
        ),
        "",
        f"## Similarity scores (descriptive only, no threshold){suffix}",
        "",
        *_similarity_table(metrics["global"]["similarity_scores"]),
        "",
        "## Questions",
        "",
        *_question_table(report["questions"]),
        "",
        "## Manual review sample",
        "",
        report["manual_review"]["selection_rule"],
        "",
        *_review_section(report["manual_review"]["items"]),
        "## Known limitations",
        "",
        *[f"- {limitation}" for limitation in report["limitations"]],
        "",
    ]
    return "\n".join(lines)


def _worst_case_formula(bounds: Mapping[str, Any]) -> str:
    return (
        f"{bounds['max_requests']} requests x (1 + {bounds['max_retries']} retries) x "
        f"{bounds['max_output_tokens']} output tokens = {bounds['worst_case_output_tokens']}"
    )


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, dict | list):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    elif value is None:
        value = "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines


def _is_rate(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"count", "n", "rate"}


def _rate(value: Mapping[str, Any]) -> str:
    if value["rate"] is None:
        return f"{value['count']}/{value['n']} (n/a)"
    return f"{value['count']}/{value['n']} ({value['rate'] * 100:.1f}%)"


def _mrr(value: Mapping[str, Any]) -> str:
    shown = "n/a" if value["value"] is None else f"{value['value']:.4f}"
    return f"{shown} (sum {value['reciprocal_rank_sum']} / n {value['n']})"


def _rate_rows(block: Mapping[str, Any], prefix: str = "") -> Iterator[list[str]]:
    for key, value in block.items():
        label = f"{prefix}{key}"
        if _is_rate(value):
            yield [label, _rate(value)]
        elif isinstance(value, dict):
            yield from _rate_rows(value, f"{label}.")


def _rate_table(block: Mapping[str, Any]) -> list[str]:
    return _table(["Metric", "Count / n (rate)"], list(_rate_rows(block)))


def _outcome_table(
    by_category: Mapping[str, Mapping[str, int]], questions: list[Mapping[str, Any]]
) -> list[str]:
    outcomes = [outcome.value for outcome in Outcome]
    rows = []
    for category, counts in by_category.items():
        not_evaluated = sum(
            1
            for question in questions
            if question["category"] == category
            and question["status"] == "not_evaluated"
        )
        rows.append(
            [category, *(counts[outcome] for outcome in outcomes), not_evaluated]
        )
    return _table(["Category", *outcomes, "not_evaluated"], rows)


def _similarity_table(
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    columns = ["n", "min", "p25", "median", "p75", "max", "mean"]
    rows = [
        [category, selection, *(summary[column] for column in columns)]
        for category, selections in scores.items()
        for selection, summary in selections.items()
    ]
    return _table(["Category", "Scores", *columns], rows)


def _question_table(questions: list[Mapping[str, Any]]) -> list[str]:
    rows = []
    for question in questions:
        detail = question["reason"] or question["code"] or question["provider_error"]
        cited = [citation["source_path"] for citation in question["citations"]]
        rows.append(
            [
                question["id"],
                question["status"],
                question["outcome"],
                detail,
                question["expected_rank"],
                cited,
                question["heuristic"],
            ]
        )
    return _table(
        [
            "Question",
            "Status",
            "Outcome",
            "Detail",
            "Expected rank",
            "Cited",
            "Heuristic",
        ],
        rows,
    )


def _review_section(items: list[Mapping[str, Any]]) -> list[str]:
    if not items:
        return ["No answered question was selected.", ""]
    lines: list[str] = []
    for item in items:
        lines += [
            f"### {item['id']} ({item['category']})",
            "",
            "Question and answer, JSON-escaped:",
            "",
            f"    {json.dumps(item['question'], ensure_ascii=True)}",
            f"    {json.dumps(item['answer'], ensure_ascii=True)}",
            "",
            *_table(
                ["Field", "Value"],
                [
                    [key, item[key]]
                    for key in (
                        "cited_sources",
                        "expected_sources",
                        "required_keywords",
                        "forbidden_keywords",
                        "heuristic",
                        "notes",
                    )
                ],
            ),
            "",
            "Manual verdict: not assigned by the harness (see docs/evaluation.md).",
            "",
        ]
    return lines


# --- Writing ------------------------------------------------------------------


def require_new_output_dir(output_dir: Path) -> None:
    """Raise `ReportWriteError` if `output_dir` already exists."""
    if output_dir.exists():
        raise ReportWriteError(f"output directory already exists: {output_dir}")


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write both report files into the new directory `output_dir`, atomically."""
    require_new_output_dir(output_dir)
    staging: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.partial-", dir=output_dir.parent
            )
        )
        text = to_json(report)
        # Rendered from the serialized JSON, so the committed JSON reproduces it.
        markdown = render_markdown(json.loads(text))
        (staging / JSON_FILE).write_text(text, encoding="utf-8")
        (staging / MARKDOWN_FILE).write_text(markdown, encoding="utf-8")
        os.rename(staging, output_dir)
    except OSError as exc:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise ReportWriteError(
            f"cannot write the report to {output_dir}: {type(exc).__name__}"
        ) from None


def to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
