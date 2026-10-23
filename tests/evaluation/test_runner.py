"""Outcome classification and the stopping rules of the evaluation run."""

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from app.answering.models import AnswerDraft, InsufficientReason, InvalidOutputCode
from app.answering.service import AnswerService
from app.evaluation.dataset import EvalQuestion
from app.evaluation.fakes import FakeBehavior
from app.evaluation.guards import (
    ProviderFailureKind,
    RecordingRetrievalService,
    RequestLimiter,
    UsageRecorder,
)
from app.evaluation.runner import AbortReason, Outcome, QuestionStatus, run_evaluation
from app.llm.errors import LLMError
from app.retrieval.models import RetrievedChunk
from tests.evaluation.support import (
    Harness,
    MappedRetrieval,
    build_harness_for,
    chunk,
    question,
)

B = FakeBehavior


def questions(count: int) -> list[EvalQuestion]:
    return [question(f"answerable-{index:02d}") for index in range(1, count + 1)]


def test_every_outcome_class_and_code_is_classified() -> None:
    qs = questions(8)
    harness = build_harness_for(
        qs,
        [
            B.QUOTE_FIRST_SOURCE,
            B.DECLINE,
            B.UNKNOWN_CITATION,
            B.MISSING_EVIDENCE,
            B.INCONSISTENT,
            B.MALFORMED,
            B.TRANSIENT_PROVIDER_ERROR,
            B.QUOTE_FIRST_SOURCE,
        ],
    )

    run = harness.run(qs)

    outcomes = [
        (r.outcome, r.reason or r.code or r.provider_error) for r in run.results
    ]
    assert outcomes == [
        (Outcome.ANSWERED, None),
        (Outcome.INSUFFICIENT_CONTEXT, InsufficientReason.MODEL_DECLINED),
        (Outcome.INVALID_OUTPUT, InvalidOutputCode.UNKNOWN_CITATION),
        (Outcome.INVALID_OUTPUT, InvalidOutputCode.MISSING_EVIDENCE),
        (Outcome.INVALID_OUTPUT, InvalidOutputCode.INCONSISTENT_RESPONSE),
        (Outcome.INVALID_OUTPUT, InvalidOutputCode.MALFORMED_OUTPUT),
        (Outcome.PROVIDER_ERROR, "LLM provider rate limit exceeded"),
        (Outcome.ANSWERED, None),
    ]
    assert run.complete
    assert run.results[6].provider_failure_kind is ProviderFailureKind.TRANSIENT
    assert all(r.llm_invoked for r in run.results)


@pytest.mark.parametrize(
    ("retrieved", "reason"),
    [
        ([], InsufficientReason.NO_RETRIEVAL),
        ([chunk("c1", content="   \n ")], InsufficientReason.EMPTY_CONTEXT),
        ([chunk("c1", content="x" * 9000)], InsufficientReason.EMPTY_CONTEXT),
    ],
)
def test_questions_without_context_never_invoke_the_model(
    retrieved: list[RetrievedChunk], reason: InsufficientReason
) -> None:
    q = question()
    harness = build_harness_for([q], [B.QUOTE_FIRST_SOURCE], {q.question: retrieved})

    [result] = harness.run([q]).results

    assert (result.outcome, result.reason) == (Outcome.INSUFFICIENT_CONTEXT, reason)
    assert result.llm_invoked is False
    assert harness.fake.calls == []
    assert harness.limiter.invocations == 0


def test_each_question_is_asked_exactly_once_without_retries() -> None:
    qs = questions(3)
    harness = build_harness_for(
        qs, [B.TRANSIENT_PROVIDER_ERROR, B.MALFORMED, B.DECLINE]
    )

    harness.run(qs)

    assert [call["question"] for call in harness.fake.calls] == [q.question for q in qs]
    assert harness.recorder.take() is None


def test_retrieval_is_captured_for_every_outcome() -> None:
    qs = questions(4)
    harness = build_harness_for(
        qs, [B.QUOTE_FIRST_SOURCE, B.DECLINE, B.MALFORMED, B.TRANSIENT_PROVIDER_ERROR]
    )

    run = harness.run(qs)

    assert all(r.retrieved and r.retrieved[0].chunk_id == "c1" for r in run.results)


def test_the_request_cap_stops_the_run_with_a_partial_result() -> None:
    qs = questions(4)
    harness = build_harness_for(qs, [B.DECLINE] * 4, max_requests=2)

    run = harness.run(qs)

    assert run.abort_reason is AbortReason.MAX_REQUESTS
    assert run.aborted_at_question == "answerable-03"
    assert not run.complete
    assert run.llm_invocations == 2 == len(harness.fake.calls)
    statuses = [r.status for r in run.results]
    assert (
        statuses == [QuestionStatus.EVALUATED] * 2 + [QuestionStatus.NOT_EVALUATED] * 2
    )
    refused, later = run.results[2], run.results[3]
    assert refused.outcome is None and refused.llm_invoked is False
    assert refused.retrieved is not None  # retrieval happened before the refusal
    assert later.retrieved is None


def test_questions_without_context_do_not_use_up_the_cap() -> None:
    qs = questions(3)
    harness = build_harness_for(
        qs, [B.DECLINE] * 3, {qs[0].question: []}, max_requests=2
    )

    run = harness.run(qs)

    assert run.complete
    assert run.llm_invocations == 2


def test_a_nonrecoverable_provider_error_stops_the_run_after_that_question() -> None:
    qs = questions(4)
    harness = build_harness_for(
        qs, [B.DECLINE, B.NONRECOVERABLE_PROVIDER_ERROR, B.DECLINE, B.DECLINE]
    )

    run = harness.run(qs)

    assert run.abort_reason is AbortReason.PROVIDER_NONRECOVERABLE
    assert run.aborted_at_question == "answerable-02"
    failed = run.results[1]
    assert failed.status is QuestionStatus.EVALUATED
    assert failed.outcome is Outcome.PROVIDER_ERROR
    assert failed.provider_failure_kind is ProviderFailureKind.NONRECOVERABLE
    assert failed.provider_error == "LLM provider authentication failed"
    assert [r.status for r in run.results[2:]] == [QuestionStatus.NOT_EVALUATED] * 2
    assert run.llm_invocations == 2


def test_three_consecutive_transient_errors_stop_the_run() -> None:
    qs = questions(5)
    harness = build_harness_for(qs, [B.TRANSIENT_PROVIDER_ERROR] * 5)

    run = harness.run(qs)

    assert run.abort_reason is AbortReason.PROVIDER_CONSECUTIVE_ERRORS
    assert run.aborted_at_question == "answerable-03"
    assert [r.outcome for r in run.results[:3]] == [Outcome.PROVIDER_ERROR] * 3
    assert run.llm_invocations == 3


def test_a_successful_invocation_resets_the_transient_streak() -> None:
    qs = questions(5)
    harness = build_harness_for(
        qs,
        [
            B.TRANSIENT_PROVIDER_ERROR,
            B.TRANSIENT_PROVIDER_ERROR,
            B.MALFORMED,
            B.TRANSIENT_PROVIDER_ERROR,
            B.TRANSIENT_PROVIDER_ERROR,
        ],
    )

    assert harness.run(qs).complete


def test_questions_without_an_invocation_do_not_reset_the_streak() -> None:
    qs = questions(4)
    harness = build_harness_for(
        qs, [B.TRANSIENT_PROVIDER_ERROR] * 4, {qs[1].question: []}
    )

    run = harness.run(qs)

    assert run.abort_reason is AbortReason.PROVIDER_CONSECUTIVE_ERRORS
    assert run.aborted_at_question == "answerable-04"


class FailingRetrieval:
    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        raise RuntimeError("postgresql://user:SENTINEL-PASSWORD@db/opsknowledge")


def test_an_infrastructure_error_stops_the_run_and_keeps_the_type_only() -> None:
    qs = questions(2)
    harness = Harness(FailingRetrieval(), {})

    run = harness.run(qs)

    assert run.abort_reason is AbortReason.UNEXPECTED_ERROR
    assert run.abort_detail == "RuntimeError"
    assert [r.status for r in run.results] == [QuestionStatus.NOT_EVALUATED] * 2
    assert all(r.outcome is None for r in run.results)
    assert "SENTINEL-PASSWORD" not in repr(run)


def test_an_interrupt_returns_the_partial_result() -> None:
    qs = questions(3)

    def chain(inputs: dict[str, Any]) -> Any:
        if inputs["question"] == qs[1].question:
            raise KeyboardInterrupt
        return AnswerDraft(insufficient=True, claims=[])

    recorder = RecordingRetrievalService(
        MappedRetrieval({q.question: [chunk("c1")] for q in qs})
    )
    usage = UsageRecorder()
    limiter = RequestLimiter(RunnableLambda(chain), max_requests=30)
    service = AnswerService(recorder, limiter.as_runnable())

    run = run_evaluation(qs, service, recorder, limiter, usage, 5)

    assert run.abort_reason is AbortReason.INTERRUPTED
    assert run.aborted_at_question == qs[1].id
    assert run.results[0].status is QuestionStatus.EVALUATED
    assert run.results[1].status is QuestionStatus.NOT_EVALUATED
    assert run.results[1].llm_invoked is True
    assert run.results[2].retrieved is None


class InterruptingRecorder(RecordingRetrievalService):
    """Raises from `take()`, i.e. outside the answer call."""

    def __init__(self, inner: MappedRetrieval, failure: BaseException) -> None:
        super().__init__(inner)
        self.failure = failure
        self.takes = 0

    def take(self) -> list[RetrievedChunk] | None:
        self.takes += 1
        if self.takes == 2:
            raise self.failure
        return super().take()


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (KeyboardInterrupt(), AbortReason.INTERRUPTED),
        (RuntimeError("defect"), AbortReason.UNEXPECTED_ERROR),
    ],
)
def test_a_stop_outside_the_answer_call_keeps_completed_results(
    failure: BaseException, reason: AbortReason
) -> None:
    qs = questions(3)
    recorder = InterruptingRecorder(
        MappedRetrieval({q.question: [chunk("c1")] for q in qs}), failure
    )
    limiter = RequestLimiter(
        RunnableLambda(lambda inputs: AnswerDraft(insufficient=True, claims=[])), 30
    )
    service = AnswerService(recorder, limiter.as_runnable())

    run = run_evaluation(qs, service, recorder, limiter, UsageRecorder(), 5)

    assert run.abort_reason is reason
    assert run.aborted_at_question == qs[1].id
    assert run.results[0].status is QuestionStatus.EVALUATED
    assert [r.status for r in run.results[1:]] == [QuestionStatus.NOT_EVALUATED] * 2
    assert len(run.results) == 3


def test_an_unclassified_provider_failure_fails_closed() -> None:
    qs = questions(3)

    def chain(inputs: dict[str, Any]) -> Any:
        raise LLMError("LLM provider request failed")

    recorder = RecordingRetrievalService(
        MappedRetrieval({q.question: [chunk("c1")] for q in qs})
    )
    limiter = RequestLimiter(RunnableLambda(chain), 30)
    service = AnswerService(recorder, limiter.as_runnable())

    run = run_evaluation(qs, service, recorder, limiter, UsageRecorder(), 5)

    assert run.results[0].provider_failure_kind is ProviderFailureKind.NONRECOVERABLE
    assert run.abort_reason is AbortReason.PROVIDER_NONRECOVERABLE
    assert run.llm_invocations == 1
