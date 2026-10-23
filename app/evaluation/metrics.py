"""Evaluation metrics, computed from captured run results only.

Pure functions: no retrieval, embedding or model call, no I/O. Every rate is a
`{"count", "n", "rate"}` mapping, `rate = count / n` rounded to 4 decimals, or
`None` when `n == 0`. Denominators are **evaluated** questions only:
`not_evaluated` questions of a partial run are excluded everywhere.

Retrieval rank
--------------

The rank uses the captured result list exactly as `retrieve` returned it: no
deduplication, no reordering, no filtering. Matching is at document level (a chunk
is relevant iff its `source_path` is one of the question's `expected_sources`),
but the position is the chunk position. For the ranked chunks A1, A2, A3, B1 with
B expected, the expected rank is 4, not 2.

- `expected_rank` = smallest 1-based position `i <= k` whose chunk is relevant,
  or `None`;
- `hit@k` = 1 if `expected_rank` is set, else 0;
- `RR@k` = `1 / expected_rank` if set, else 0;
- `hit@k` and `MRR@k` are means over evaluated answerable questions, whatever their
  answer outcome.

`k` is the retrieval `limit` of the run. Only the first `k` captured chunks are
considered.

Answer heuristics
-----------------

Keyword checks are case-insensitive substring matches (the dataset convention)
against the answer text with the trailing ` [n]` citation marker of each line
removed, so a marker can never satisfy or violate a keyword. Citation checks
compare the trusted `Citation.source_path` values with `expected_sources`. These
are heuristic proxies, not measures of factual correctness.

- answerable: `heuristic_correct` = answered, an expected source cited, every
  required keyword present. Every other outcome, `invalid_output` and
  `provider_error` included, is a failure; neither is an abstention.
- unanswerable: abstention = genuine `insufficient_context` (any reason); false
  answer = `answered`; `invalid_output` and `provider_error` are separate
  fail-closed buckets.
- misleading: false answer = answered with a forbidden keyword, or answered
  without citing an expected source (always, when there is none); accepted =
  genuine `insufficient_context` (declined) or an answer that is not false
  (supported correction). `required_keywords` are not scored.

Similarity scores are summarized descriptively. No threshold is derived or
applied.
"""

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from app.answering.models import Answer, InsufficientReason, InvalidOutputCode
from app.evaluation.dataset import EvalCategory
from app.evaluation.runner import Outcome, QuestionResult
from app.retrieval.models import RetrievedChunk

_TRAILING_MARKER = re.compile(r" \[[0-9]+\]$")


def rate(count: int, n: int) -> dict[str, Any]:
    return {"count": count, "n": n, "rate": round(count / n, 4) if n else None}


def expected_rank(
    retrieved: Sequence[RetrievedChunk], expected_sources: Iterable[str], k: int
) -> int | None:
    expected = set(expected_sources)
    for position, chunk in enumerate(retrieved[:k], start=1):
        if chunk.source_path in expected:
            return position
    return None


def relevant_chunk_count(
    retrieved: Sequence[RetrievedChunk], expected_sources: Iterable[str], k: int
) -> int:
    expected = set(expected_sources)
    return sum(1 for chunk in retrieved[:k] if chunk.source_path in expected)


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def answer_haystack(answer_text: str) -> str:
    """The answer text without the trailing ` [n]` marker of each line."""
    return "\n".join(_TRAILING_MARKER.sub("", line) for line in answer_text.split("\n"))


def contains_keyword(haystack: str, keyword: str) -> bool:
    return keyword.casefold() in haystack.casefold()


def cites_expected_source(answer: Answer, expected_sources: Iterable[str]) -> bool:
    expected = set(expected_sources)
    return any(citation.source_path in expected for citation in answer.citations)


def question_heuristics(result: QuestionResult) -> dict[str, Any]:
    """Per-question heuristic facts; empty unless the question was answered."""
    answer = result.answer
    if result.outcome is not Outcome.ANSWERED or answer is None:
        return {}
    question = result.question
    haystack = answer_haystack(answer.answer or "")
    facts: dict[str, Any] = {
        "cites_expected_source": cites_expected_source(
            answer, question.expected_sources
        ),
        "missing_required_keywords": [
            keyword
            for keyword in question.required_keywords
            if not contains_keyword(haystack, keyword)
        ],
        "forbidden_keywords_found": [
            keyword
            for keyword in question.forbidden_keywords
            if contains_keyword(haystack, keyword)
        ],
    }
    if question.category is EvalCategory.ANSWERABLE:
        facts["heuristic_correct"] = (
            facts["cites_expected_source"] and not facts["missing_required_keywords"]
        )
    elif question.category is EvalCategory.MISLEADING:
        facts["heuristic_false_answer"] = bool(facts["forbidden_keywords_found"]) or (
            not facts["cites_expected_source"]
        )
    return facts


def compute_metrics(
    results: Sequence[QuestionResult], k: int, synthetic: bool
) -> dict[str, Any]:
    evaluated = [result for result in results if result.evaluated]
    by_category = {
        category: [r for r in evaluated if r.question.category is category]
        for category in EvalCategory
    }
    metrics = {
        "retrieval": _retrieval(by_category[EvalCategory.ANSWERABLE], k),
        "answerable": _answerable(by_category[EvalCategory.ANSWERABLE]),
        "unanswerable": _unanswerable(by_category[EvalCategory.UNANSWERABLE]),
        "misleading": _misleading(by_category[EvalCategory.MISLEADING]),
        "global": _global(evaluated, by_category),
    }
    for block in metrics.values():
        block["synthetic"] = synthetic
    return metrics


def _retrieval(answerable: list[QuestionResult], k: int) -> dict[str, Any]:
    ranks = [
        expected_rank(r.retrieved or (), r.question.expected_sources, k)
        for r in answerable
    ]
    n = len(ranks)
    reciprocal_sum = sum(reciprocal_rank(rank) for rank in ranks)
    histogram = {str(position): 0 for position in range(1, k + 1)} | {"miss": 0}
    for rank in ranks:
        histogram[str(rank) if rank else "miss"] += 1
    return {
        "k": k,
        "n": n,
        "hit_at_k": rate(sum(1 for rank in ranks if rank), n),
        "mrr_at_k": {
            "reciprocal_rank_sum": round(reciprocal_sum, 6),
            "n": n,
            "value": round(reciprocal_sum / n, 4) if n else None,
        },
        "expected_rank_histogram": histogram,
    }


def _answerable(results: list[QuestionResult]) -> dict[str, Any]:
    n = len(results)
    correct = missing_citation = missing_keywords = 0
    for result in results:
        if result.outcome is not Outcome.ANSWERED:
            continue
        facts = question_heuristics(result)
        if facts["heuristic_correct"]:
            correct += 1
        elif not facts["cites_expected_source"]:
            missing_citation += 1
        else:
            missing_keywords += 1
    return {
        "n": n,
        "heuristic_correct": rate(correct, n),
        "failures": {
            "answered_missing_expected_citation": rate(missing_citation, n),
            "answered_missing_required_keywords": rate(missing_keywords, n),
            "insufficient_context": _by_reason(results, n),
            "invalid_output": _by_code(results, n),
            "provider_error": rate(_count(results, Outcome.PROVIDER_ERROR), n),
        },
    }


def _unanswerable(results: list[QuestionResult]) -> dict[str, Any]:
    n = len(results)
    return {
        "n": n,
        "abstention": rate(_count(results, Outcome.INSUFFICIENT_CONTEXT), n),
        "abstention_by_reason": _by_reason(results, n),
        "false_answer": rate(_count(results, Outcome.ANSWERED), n),
        "invalid_output": _by_code(results, n),
        "provider_error": rate(_count(results, Outcome.PROVIDER_ERROR), n),
    }


def _misleading(results: list[QuestionResult]) -> dict[str, Any]:
    n = len(results)
    false_answer = forbidden = uncited = supported = 0
    for result in results:
        if result.outcome is not Outcome.ANSWERED:
            continue
        facts = question_heuristics(result)
        if facts["heuristic_false_answer"]:
            false_answer += 1
            forbidden += bool(facts["forbidden_keywords_found"])
            uncited += not facts["cites_expected_source"]
        else:
            supported += 1
    declined = _count(results, Outcome.INSUFFICIENT_CONTEXT)
    return {
        "n": n,
        "heuristic_false_answer": rate(false_answer, n),
        # The two reasons can apply to the same answer, so they may overlap.
        "heuristic_false_answer_reasons": {
            "forbidden_keyword": rate(forbidden, n),
            "no_expected_source_cited": rate(uncited, n),
        },
        "accepted": rate(declined + supported, n),
        "accepted_declined": rate(declined, n),
        "accepted_heuristic_supported_correction": rate(supported, n),
        "declined_by_reason": _by_reason(results, n),
        "invalid_output": _by_code(results, n),
        "provider_error": rate(_count(results, Outcome.PROVIDER_ERROR), n),
    }


def _global(
    evaluated: list[QuestionResult],
    by_category: dict[EvalCategory, list[QuestionResult]],
) -> dict[str, Any]:
    n = len(evaluated)
    answered = [
        r for r in evaluated if r.outcome is Outcome.ANSWERED and r.answer is not None
    ]
    citation_counts = [len(r.answer.citations) for r in answered if r.answer]
    return {
        "n": n,
        "outcomes": {
            outcome.value: rate(_count(evaluated, outcome), n) for outcome in Outcome
        },
        "outcomes_by_category": {
            category.value: {
                outcome.value: _count(results, outcome) for outcome in Outcome
            }
            for category, results in by_category.items()
        },
        "invalid_output": _by_code(evaluated, n),
        "provider_error": rate(_count(evaluated, Outcome.PROVIDER_ERROR), n),
        "llm_invoked_questions": sum(1 for r in evaluated if r.llm_invoked),
        "citation_count_distribution": {
            "n": len(citation_counts),
            "histogram": {
                str(count): number
                for count, number in sorted(Counter(citation_counts).items())
            },
            "min": min(citation_counts, default=None),
            "median": statistics.median(citation_counts) if citation_counts else None,
            "max": max(citation_counts, default=None),
        },
        "similarity_scores": {
            category.value: {
                "top1": describe(
                    [r.retrieved[0].similarity_score for r in results if r.retrieved]
                ),
                "all": describe(
                    [
                        chunk.similarity_score
                        for r in results
                        for chunk in r.retrieved or ()
                    ]
                ),
            }
            for category, results in by_category.items()
        },
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    """Descriptive summary: n, min, quartiles, max, mean; `None` values when empty."""
    if not values:
        return {"n": 0} | dict.fromkeys(("min", "p25", "median", "p75", "max", "mean"))
    if len(values) == 1:
        p25, median, p75 = values[0], values[0], values[0]
    else:
        p25, median, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "p25": round(p25, 4),
        "median": round(median, 4),
        "p75": round(p75, 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
    }


def _count(results: Iterable[QuestionResult], outcome: Outcome) -> int:
    return sum(1 for result in results if result.outcome is outcome)


def _by_reason(results: list[QuestionResult], n: int) -> dict[str, Any]:
    return {
        reason.value: rate(
            sum(
                1
                for r in results
                if r.outcome is Outcome.INSUFFICIENT_CONTEXT and r.reason is reason
            ),
            n,
        )
        for reason in InsufficientReason
    }


def _by_code(results: list[QuestionResult], n: int) -> dict[str, Any]:
    return {
        code.value: rate(
            sum(
                1
                for r in results
                if r.outcome is Outcome.INVALID_OUTPUT and r.code is code
            ),
            n,
        )
        for code in InvalidOutputCode
    }
