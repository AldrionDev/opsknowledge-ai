"""Run the dataset through `AnswerService` and classify every question's outcome.

Outcome classes (exactly one per evaluated question):

- `answered`: a validated answer with citations;
- `insufficient_context`: genuine lack of knowledge, with its reason
  (`no_retrieval`, `empty_context`, `model_declined`);
- `invalid_output`: the model output was rejected, with its `InvalidOutputCode`.
  It is a failure, never an abstention;
- `provider_error`: the provider call failed (`LLMError`), with the fixed
  message and the `ProviderFailureKind` the limiter observed.

A question that was not evaluated because the run stopped first has status
`not_evaluated` and no outcome; it is not a fifth outcome class and is excluded
from every metric.

The run stops early, keeping every completed result, when:

- the request cap refuses an invocation (`max_requests`): the refused question
  and all later ones are `not_evaluated`;
- a provider failure is nonrecoverable (`provider_nonrecoverable`): that question
  is evaluated as `provider_error`, all later ones are `not_evaluated`;
- `MAX_CONSECUTIVE_PROVIDER_ERRORS` consecutive LLM invocations fail
  (`provider_consecutive_errors`). Questions that do not invoke the LLM neither
  count nor reset the streak; any other outcome of an invocation resets it;
- any other exception occurs, such as a database or embedding failure or a
  defect (`unexpected_error`). Only the exception type is kept: its message may
  carry connection details. Such failures are never counted as `provider_error`,
  which would distort that metric;
- `KeyboardInterrupt` (`interrupted`): the partial result is returned so the
  caller can still write the report.

An interrupt or an error outside the answer call itself (for example while the
result is recorded) stops the run in the same way, so completed results are
never lost. A provider failure the limiter could not classify is treated as
nonrecoverable: the run fails closed instead of repeating it.

There are no application-level retries: each question is asked exactly once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.answering.errors import InvalidModelOutputError
from app.answering.models import (
    Answer,
    AnswerStatus,
    InsufficientReason,
    InvalidOutputCode,
)
from app.answering.service import AnswerService
from app.evaluation.dataset import EvalQuestion
from app.evaluation.guards import (
    MAX_CONSECUTIVE_PROVIDER_ERRORS,
    ProviderFailureKind,
    RecordingRetrievalService,
    RequestCapReachedError,
    RequestLimiter,
    TokenUsage,
    UsageRecorder,
)
from app.llm.errors import LLMError
from app.retrieval.models import RetrievedChunk


class Outcome(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    INVALID_OUTPUT = "invalid_output"
    PROVIDER_ERROR = "provider_error"


class QuestionStatus(StrEnum):
    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"


class AbortReason(StrEnum):
    MAX_REQUESTS = "max_requests"
    PROVIDER_NONRECOVERABLE = "provider_nonrecoverable"
    PROVIDER_CONSECUTIVE_ERRORS = "provider_consecutive_errors"
    UNEXPECTED_ERROR = "unexpected_error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class QuestionResult:
    question: EvalQuestion
    status: QuestionStatus
    outcome: Outcome | None = None
    answer: Answer | None = None
    reason: InsufficientReason | None = None
    code: InvalidOutputCode | None = None
    provider_error: str | None = None
    provider_failure_kind: ProviderFailureKind | None = None
    llm_invoked: bool = False
    retrieved: tuple[RetrievedChunk, ...] | None = None
    usage: tuple[TokenUsage, ...] = ()

    @property
    def evaluated(self) -> bool:
        return self.status is QuestionStatus.EVALUATED


@dataclass(frozen=True)
class RunResult:
    """Every question of the dataset, in order, evaluated or not."""

    results: tuple[QuestionResult, ...]
    llm_invocations: int
    abort_reason: AbortReason | None = None
    abort_detail: str | None = None
    aborted_at_question: str | None = None

    @property
    def complete(self) -> bool:
        return self.abort_reason is None


@dataclass(frozen=True)
class _Abort:
    reason: AbortReason
    detail: str | None = None


def run_evaluation(
    questions: Sequence[EvalQuestion],
    answer_service: AnswerService,
    retrieval: RecordingRetrievalService,
    limiter: RequestLimiter,
    usage: UsageRecorder,
    limit: int,
) -> RunResult:
    """Ask every question once, in order; see the module docstring for stopping rules.

    `answer_service` must be built on `retrieval` and on `limiter.as_runnable()`,
    and `usage` attached to the chain inside the limiter.
    """
    results: list[QuestionResult] = []
    consecutive_provider_errors = 0

    try:
        for question in questions:
            invocations_before = limiter.invocations
            responses_before = len(usage.responses)
            abort: _Abort | None = None
            answer: Answer | None = None
            outcome: Outcome | None = None
            code: InvalidOutputCode | None = None
            provider_error: str | None = None
            kind: ProviderFailureKind | None = None
            try:
                answer = answer_service.answer(question.question, limit)
                outcome = (
                    Outcome.ANSWERED
                    if answer.status is AnswerStatus.ANSWERED
                    else Outcome.INSUFFICIENT_CONTEXT
                )
            except InvalidModelOutputError as exc:
                outcome, code = Outcome.INVALID_OUTPUT, exc.code
            except LLMError as exc:
                outcome, provider_error = Outcome.PROVIDER_ERROR, str(exc)
                # Fail closed: a provider failure the limiter did not classify
                # stops the run rather than being retried on later questions.
                kind = limiter.last_failure_kind or ProviderFailureKind.NONRECOVERABLE
            except RequestCapReachedError:
                abort = _Abort(AbortReason.MAX_REQUESTS)
            except KeyboardInterrupt:
                abort = _Abort(AbortReason.INTERRUPTED)
            except Exception as exc:
                abort = _Abort(AbortReason.UNEXPECTED_ERROR, type(exc).__name__)

            retrieved = retrieval.take()
            llm_invoked = limiter.invocations > invocations_before
            result = QuestionResult(
                question=question,
                status=QuestionStatus.NOT_EVALUATED
                if abort
                else QuestionStatus.EVALUATED,
                outcome=outcome,
                answer=answer if outcome is Outcome.ANSWERED else None,
                reason=answer.reason if answer is not None else None,
                code=code,
                provider_error=provider_error,
                provider_failure_kind=kind,
                llm_invoked=llm_invoked,
                retrieved=None if retrieved is None else tuple(retrieved),
                usage=tuple(usage.responses[responses_before:]),
            )
            results.append(result)

            if abort is None and result.outcome is Outcome.PROVIDER_ERROR:
                consecutive_provider_errors += 1
                if result.provider_failure_kind is ProviderFailureKind.NONRECOVERABLE:
                    abort = _Abort(AbortReason.PROVIDER_NONRECOVERABLE)
                elif consecutive_provider_errors >= MAX_CONSECUTIVE_PROVIDER_ERRORS:
                    abort = _Abort(AbortReason.PROVIDER_CONSECUTIVE_ERRORS)
            elif llm_invoked:
                consecutive_provider_errors = 0

            if abort is not None:
                return _stopped(results, questions, limiter, abort, question.id)

    except KeyboardInterrupt:
        # Outside `answer()`, e.g. while recording; the completed results stay.
        return _stopped(results, questions, limiter, _Abort(AbortReason.INTERRUPTED))
    except Exception as exc:
        return _stopped(
            results,
            questions,
            limiter,
            _Abort(AbortReason.UNEXPECTED_ERROR, type(exc).__name__),
        )

    return RunResult(results=tuple(results), llm_invocations=limiter.invocations)


def _stopped(
    results: list[QuestionResult],
    questions: Sequence[EvalQuestion],
    limiter: RequestLimiter,
    abort: _Abort,
    aborted_at: str | None = None,
) -> RunResult:
    """The partial result: questions without a result are `not_evaluated`."""
    remaining = questions[len(results) :]
    if aborted_at is None and remaining:
        aborted_at = remaining[0].id
    return RunResult(
        results=tuple(results)
        + tuple(
            QuestionResult(question=later, status=QuestionStatus.NOT_EVALUATED)
            for later in remaining
        ),
        llm_invocations=limiter.invocations,
        abort_reason=abort.reason,
        abort_detail=abort.detail,
        aborted_at_question=aborted_at,
    )
