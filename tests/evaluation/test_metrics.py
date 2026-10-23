"""Metric semantics on constructed results: rank, heuristics, denominators."""

from collections.abc import Iterator
from typing import Any

import pytest

from app.answering.models import InsufficientReason, InvalidOutputCode
from app.evaluation.dataset import EvalCategory
from app.evaluation.metrics import (
    answer_haystack,
    compute_metrics,
    contains_keyword,
    describe,
    expected_rank,
    question_heuristics,
    reciprocal_rank,
    relevant_chunk_count,
)
from app.evaluation.runner import Outcome
from tests.evaluation.support import (
    answer,
    evaluated,
    not_evaluated,
    question,
    ranked,
)

A, B, C = "runbooks/a.md", "incidents/b.md", "runbooks/c.md"
U = EvalCategory.UNANSWERABLE
M = EvalCategory.MISLEADING


# --- rank, hit@k, MRR ------------------------------------------------------------


def test_repeated_nonmatching_chunks_keep_their_positions() -> None:
    retrieved = ranked(A, A, A, B)

    rank = expected_rank(retrieved, [B], k=4)

    assert rank == 4
    assert reciprocal_rank(rank) == 0.25


def test_repeated_relevant_chunks_rank_at_the_first_one() -> None:
    retrieved = ranked(A, B, B, C)

    assert expected_rank(retrieved, [B], k=4) == 2
    assert relevant_chunk_count(retrieved, [B], k=4) == 2


def test_any_of_several_expected_sources_counts() -> None:
    retrieved = ranked(A, C, B)

    assert expected_rank(retrieved, [B, C], k=3) == 2


def test_no_relevant_chunk_is_a_miss() -> None:
    assert expected_rank(ranked(A, A, C), [B], k=3) is None
    assert reciprocal_rank(None) == 0.0
    assert expected_rank((), [B], k=5) is None


def test_a_relevant_chunk_beyond_k_is_a_miss() -> None:
    assert expected_rank(ranked(A, A, C, B), [B], k=3) is None
    assert relevant_chunk_count(ranked(A, A, C, B), [B], k=3) == 0


def test_hit_at_k_and_mrr_over_answerable_questions() -> None:
    results = [
        evaluated(
            question("answerable-01", expected=[B]),
            Outcome.ANSWERED,
            answer_=answer("x", B),
            retrieved=ranked(A, A, A, B),
        ),
        evaluated(
            question("answerable-02", expected=[A]),
            Outcome.INVALID_OUTPUT,
            code=InvalidOutputCode.MALFORMED_OUTPUT,
            retrieved=ranked(A, B),
        ),
        evaluated(
            question("answerable-03", expected=[C]),
            Outcome.PROVIDER_ERROR,
            retrieved=ranked(A, B),
        ),
        evaluated(
            question("answerable-04", expected=[B]),
            Outcome.INSUFFICIENT_CONTEXT,
            reason=InsufficientReason.NO_RETRIEVAL,
            retrieved=(),
            llm_invoked=False,
        ),
    ]

    retrieval = compute_metrics(results, k=4, synthetic=False)["retrieval"]

    assert retrieval["hit_at_k"] == {"count": 2, "n": 4, "rate": 0.5}
    assert retrieval["mrr_at_k"] == {
        "reciprocal_rank_sum": 1.25,
        "n": 4,
        "value": 0.3125,
    }
    assert retrieval["expected_rank_histogram"] == {
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 1,
        "miss": 2,
    }


def test_retrieval_metrics_use_answerable_questions_only() -> None:
    results = [
        evaluated(
            question("misleading-01", M, expected=[A]),
            Outcome.ANSWERED,
            answer_=answer("x", A),
            retrieved=ranked(A),
        ),
        evaluated(
            question("unanswerable-01", U, expected=[]),
            Outcome.ANSWERED,
            answer_=answer("x", A),
            retrieved=ranked(A),
        ),
    ]

    retrieval = compute_metrics(results, k=5, synthetic=False)["retrieval"]

    assert retrieval["n"] == 0
    assert retrieval["hit_at_k"]["rate"] is None
    assert retrieval["mrr_at_k"]["value"] is None


# --- keyword and citation heuristics ------------------------------------------------


def test_citation_markers_are_not_part_of_the_haystack() -> None:
    text = "Raise the limit to 5 [5]\nRestart the pod [12]"

    haystack = answer_haystack(text)

    assert haystack == "Raise the limit to 5\nRestart the pod"
    assert not contains_keyword(answer_haystack("Restart the pod [12]"), "12")
    assert not contains_keyword(answer_haystack("Restart the pod [12]"), "[12]")


def test_keywords_match_case_insensitively_as_substrings() -> None:
    assert contains_keyword("The pod was OOMKilled twice", "oomkilled")
    assert contains_keyword("terminationGracePeriodSeconds", "GracePeriod")
    assert not contains_keyword("pre stop hook", "preStop")


def test_answerable_heuristic_correctness() -> None:
    q = question(expected=[A], required=["preStop", "grace"])
    correct = evaluated(
        q, Outcome.ANSWERED, answer_=answer("Use a PRESTOP hook and grace [1]", A)
    )
    wrong_source = evaluated(
        q, Outcome.ANSWERED, answer_=answer("preStop grace [1]", B)
    )
    missing_keyword = evaluated(
        q, Outcome.ANSWERED, answer_=answer("preStop only [1]", A)
    )

    assert question_heuristics(correct)["heuristic_correct"] is True
    assert question_heuristics(wrong_source)["heuristic_correct"] is False
    assert question_heuristics(missing_keyword)["missing_required_keywords"] == [
        "grace"
    ]
    assert question_heuristics(evaluated(q, Outcome.PROVIDER_ERROR)) == {}


# --- category metrics ----------------------------------------------------------------


def rates(block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for value in block.values():
        if isinstance(value, dict) and set(value) == {"count", "n", "rate"}:
            yield value
        elif isinstance(value, dict):
            yield from rates(value)


def test_answerable_failures_include_invalid_output_and_provider_errors() -> None:
    q = question(expected=[A], required=["preStop"])
    results = [
        evaluated(q, Outcome.ANSWERED, answer_=answer("preStop [1]", A)),
        evaluated(q, Outcome.ANSWERED, answer_=answer("preStop [1]", B)),
        evaluated(q, Outcome.ANSWERED, answer_=answer("nothing [1]", A)),
        evaluated(q, Outcome.INVALID_OUTPUT, code=InvalidOutputCode.MISSING_EVIDENCE),
        evaluated(q, Outcome.PROVIDER_ERROR),
        evaluated(
            q, Outcome.INSUFFICIENT_CONTEXT, reason=InsufficientReason.MODEL_DECLINED
        ),
    ]

    metrics = compute_metrics(results, k=5, synthetic=False)["answerable"]

    assert metrics["n"] == 6
    assert metrics["heuristic_correct"]["count"] == 1
    failures = metrics["failures"]
    assert failures["answered_missing_expected_citation"]["count"] == 1
    assert failures["answered_missing_required_keywords"]["count"] == 1
    assert failures["invalid_output"]["missing_evidence"]["count"] == 1
    assert failures["provider_error"]["count"] == 1
    assert failures["insufficient_context"]["model_declined"]["count"] == 1
    assert sum(rate["count"] for rate in rates(metrics)) == 6


def test_invalid_output_is_never_an_unanswerable_abstention() -> None:
    q = question("unanswerable-01", U, expected=[])
    results = [
        evaluated(
            q,
            Outcome.INSUFFICIENT_CONTEXT,
            reason=InsufficientReason.NO_RETRIEVAL,
            llm_invoked=False,
        ),
        evaluated(
            q, Outcome.INSUFFICIENT_CONTEXT, reason=InsufficientReason.MODEL_DECLINED
        ),
        evaluated(
            q, Outcome.INVALID_OUTPUT, code=InvalidOutputCode.INCONSISTENT_RESPONSE
        ),
        evaluated(q, Outcome.INVALID_OUTPUT, code=InvalidOutputCode.MALFORMED_OUTPUT),
        evaluated(q, Outcome.PROVIDER_ERROR),
        evaluated(q, Outcome.ANSWERED, answer_=answer("made up [1]", A)),
    ]

    metrics = compute_metrics(results, k=5, synthetic=False)["unanswerable"]

    assert metrics["abstention"] == {"count": 2, "n": 6, "rate": 0.3333}
    assert metrics["false_answer"]["count"] == 1
    assert metrics["invalid_output"]["inconsistent_response"]["count"] == 1
    assert metrics["invalid_output"]["malformed_output"]["count"] == 1
    assert metrics["provider_error"]["count"] == 1
    total = (
        metrics["abstention"]["count"]
        + metrics["false_answer"]["count"]
        + sum(r["count"] for r in metrics["invalid_output"].values())
        + metrics["provider_error"]["count"]
    )
    assert total == 6


def test_misleading_false_answers_and_accepted_outcomes() -> None:
    with_source = question(
        "misleading-01",
        M,
        expected=[A],
        required=["disk"],
        forbidden=["imagePullSecrets"],
    )
    no_source = question("misleading-06", M, expected=[], forbidden=["CoreDNS"])
    results = [
        # supported correction: cites A, no forbidden keyword, required keyword absent
        evaluated(
            with_source, Outcome.ANSWERED, answer_=answer("Nodes were full [1]", A)
        ),
        evaluated(
            with_source, Outcome.ANSWERED, answer_=answer("Add IMAGEPULLSECRETS [1]", A)
        ),
        evaluated(
            with_source, Outcome.ANSWERED, answer_=answer("Disk was full [1]", B)
        ),
        evaluated(
            no_source, Outcome.ANSWERED, answer_=answer("An unrelated fact [1]", A)
        ),
        evaluated(
            no_source,
            Outcome.INSUFFICIENT_CONTEXT,
            reason=InsufficientReason.MODEL_DECLINED,
        ),
        evaluated(
            with_source, Outcome.INVALID_OUTPUT, code=InvalidOutputCode.UNKNOWN_CITATION
        ),
        evaluated(no_source, Outcome.PROVIDER_ERROR),
    ]

    metrics = compute_metrics(results, k=5, synthetic=False)["misleading"]

    assert metrics["n"] == 7
    assert metrics["heuristic_false_answer"]["count"] == 3
    assert metrics["heuristic_false_answer_reasons"]["forbidden_keyword"]["count"] == 1
    assert (
        metrics["heuristic_false_answer_reasons"]["no_expected_source_cited"]["count"]
        == 2
    )
    assert metrics["accepted_heuristic_supported_correction"]["count"] == 1
    assert metrics["accepted_declined"]["count"] == 1
    assert metrics["accepted"]["count"] == 2
    assert metrics["invalid_output"]["unknown_citation"]["count"] == 1
    assert metrics["provider_error"]["count"] == 1


def test_not_evaluated_questions_are_excluded_from_every_denominator() -> None:
    q = question(expected=[A])
    results = [
        evaluated(q, Outcome.ANSWERED, answer_=answer("x [1]", A), retrieved=ranked(A)),
        not_evaluated(q),
        not_evaluated(question("unanswerable-01", U, expected=[])),
    ]

    metrics = compute_metrics(results, k=5, synthetic=False)

    assert metrics["answerable"]["n"] == 1
    assert metrics["retrieval"]["n"] == 1
    assert metrics["unanswerable"]["n"] == 0
    assert metrics["unanswerable"]["abstention"]["rate"] is None
    assert metrics["global"]["n"] == 1


def test_global_rates_citation_distribution_and_scores() -> None:
    q = question(expected=[A])
    results = [
        evaluated(
            q,
            Outcome.ANSWERED,
            answer_=answer("x [1]\ny [2]", A, B),
            retrieved=ranked(A, B),
        ),
        evaluated(q, Outcome.ANSWERED, answer_=answer("x [1]", A), retrieved=ranked(A)),
        evaluated(
            q,
            Outcome.INVALID_OUTPUT,
            code=InvalidOutputCode.MALFORMED_OUTPUT,
            retrieved=ranked(B),
        ),
        evaluated(q, Outcome.PROVIDER_ERROR, retrieved=()),
    ]

    metrics = compute_metrics(results, k=5, synthetic=True)
    global_ = metrics["global"]

    assert global_["invalid_output"]["malformed_output"] == {
        "count": 1,
        "n": 4,
        "rate": 0.25,
    }
    assert global_["provider_error"]["rate"] == 0.25
    assert global_["citation_count_distribution"] == {
        "n": 2,
        "histogram": {"1": 1, "2": 1},
        "min": 1,
        "median": 1.5,
        "max": 2,
    }
    scores = global_["similarity_scores"]["answerable"]
    assert scores["top1"]["n"] == 3
    assert scores["all"]["n"] == 4
    assert all(block["synthetic"] is True for block in metrics.values())


def test_describe_handles_empty_single_and_many_values() -> None:
    assert describe([]) == {
        "n": 0,
        "min": None,
        "p25": None,
        "median": None,
        "p75": None,
        "max": None,
        "mean": None,
    }
    assert describe([0.5])["median"] == 0.5
    assert describe([0.1, 0.2, 0.3, 0.4, 0.5]) == {
        "n": 5,
        "min": 0.1,
        "p25": 0.2,
        "median": 0.3,
        "p75": 0.4,
        "max": 0.5,
        "mean": 0.3,
    }


def keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from keys(nested)


@pytest.mark.parametrize("synthetic", [True, False])
def test_no_threshold_is_derived(synthetic: bool) -> None:
    results = [
        evaluated(
            question(expected=[A]),
            Outcome.ANSWERED,
            answer_=answer("x [1]", A),
            retrieved=ranked(A, B),
        ),
    ]

    metrics = compute_metrics(results, k=5, synthetic=synthetic)

    assert not any("threshold" in key for key in keys(metrics))
