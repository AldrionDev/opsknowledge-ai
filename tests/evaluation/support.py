"""Builders shared by the evaluation tests."""

from collections.abc import Mapping, Sequence
from typing import Any

from app.answering.models import Answer, AnswerStatus, Citation, InvalidOutputCode
from app.answering.service import AnswerService
from app.evaluation.dataset import EvalCategory, EvalQuestion
from app.evaluation.fakes import FakeBehavior, ScriptedFakeChain
from app.evaluation.guards import (
    ProviderFailureKind,
    RecordingRetrievalService,
    RequestLimiter,
    UsageRecorder,
)
from app.evaluation.runner import (
    Outcome,
    QuestionResult,
    QuestionStatus,
    RunResult,
    run_evaluation,
)
from app.ingestion.models import DocumentType
from app.retrieval.models import RetrievedChunk

QUOTE_TEXT = (
    "The payments pod restarts because the container exceeds its memory limit "
    "and is killed by the kernel."
)


def question(
    id_: str = "answerable-01",
    category: EvalCategory = EvalCategory.ANSWERABLE,
    text: str | None = None,
    expected: Sequence[str] = ("runbooks/a.md",),
    required: Sequence[str] = (),
    forbidden: Sequence[str] = (),
) -> EvalQuestion:
    return EvalQuestion(
        id=id_,
        category=category,
        question=text or f"Question {id_}?",
        expected_sources=tuple(expected),
        required_keywords=tuple(required),
        forbidden_keywords=tuple(forbidden),
        notes="test",
    )


def chunk(
    chunk_id: str,
    source_path: str = "runbooks/a.md",
    content: str = QUOTE_TEXT,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{source_path}",
        chunk_index=0,
        source_path=source_path,
        document_type=DocumentType.RUNBOOK,
        content=content,
        similarity_score=score,
    )


def ranked(*source_paths: str) -> tuple[RetrievedChunk, ...]:
    """Chunks in the given order, scores descending."""
    return tuple(
        chunk(f"c{position}", path, score=1.0 - position / 100)
        for position, path in enumerate(source_paths, start=1)
    )


def answer(text: str, *cited_paths: str) -> Answer:
    return Answer(
        status=AnswerStatus.ANSWERED,
        answer=text,
        citations=tuple(
            Citation(
                source_id=index,
                chunk_id=f"c{index}",
                document_id=f"doc-{path}",
                source_path=path,
                document_type=DocumentType.RUNBOOK,
                chunk_index=0,
                evidence_quote="quoted evidence of twenty chars",
            )
            for index, path in enumerate(cited_paths, start=1)
        ),
        reason=None,
    )


def evaluated(
    q: EvalQuestion,
    outcome: Outcome,
    *,
    answer_: Answer | None = None,
    reason: Any = None,
    code: InvalidOutputCode | None = None,
    retrieved: Sequence[RetrievedChunk] = (),
    llm_invoked: bool = True,
) -> QuestionResult:
    return QuestionResult(
        question=q,
        status=QuestionStatus.EVALUATED,
        outcome=outcome,
        answer=answer_,
        reason=reason,
        code=code,
        provider_error="LLM provider rate limit exceeded"
        if outcome is Outcome.PROVIDER_ERROR
        else None,
        provider_failure_kind=ProviderFailureKind.TRANSIENT
        if outcome is Outcome.PROVIDER_ERROR
        else None,
        llm_invoked=llm_invoked,
        retrieved=tuple(retrieved),
    )


def not_evaluated(q: EvalQuestion) -> QuestionResult:
    return QuestionResult(question=q, status=QuestionStatus.NOT_EVALUATED)


class MappedRetrieval:
    """Returns fixed chunks per query text and counts the calls."""

    def __init__(self, results: Mapping[str, Sequence[RetrievedChunk]]) -> None:
        self._results = results
        self.calls: list[str] = []

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self._results[query])[:limit]


class Harness:
    """A real `AnswerService` wired the way the CLI wires it, over test fakes."""

    def __init__(
        self,
        retrieval: Any,
        script: Mapping[str, FakeBehavior],
        max_requests: int = 30,
        default: FakeBehavior = FakeBehavior.DECLINE,
    ) -> None:
        self.fake = ScriptedFakeChain(script, default)
        self.recorder = RecordingRetrievalService(retrieval)
        self.usage = UsageRecorder()
        self.limiter = RequestLimiter(
            self.fake.as_runnable().with_config(callbacks=[self.usage]), max_requests
        )
        self.service = AnswerService(self.recorder, self.limiter.as_runnable())

    def run(self, questions: Sequence[EvalQuestion], limit: int = 5) -> RunResult:
        return run_evaluation(
            questions, self.service, self.recorder, self.limiter, self.usage, limit
        )


def build_harness_for(
    questions: Sequence[EvalQuestion],
    behaviors: Sequence[FakeBehavior],
    retrieved: Mapping[str, Sequence[RetrievedChunk]] | None = None,
    max_requests: int = 30,
) -> Harness:
    """One behavior per question; every question retrieves `QUOTE_TEXT` by default."""
    results = retrieved or {}
    retrieval = MappedRetrieval(
        {q.question: results.get(q.question, (chunk("c1"),)) for q in questions}
    )
    script = {
        q.question: behavior for q, behavior in zip(questions, behaviors, strict=True)
    }
    return Harness(retrieval, script, max_requests=max_requests)
