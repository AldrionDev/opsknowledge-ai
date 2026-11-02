"""Report schema, labels, usage accounting, rendering and safe writing."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.answering.models import Answer, AnswerStatus
from app.evaluation import report as report_module
from app.evaluation.dataset import CORPUS_ROOT, EvalCategory, load_dataset
from app.evaluation.fakes import FakeBehavior, build_dry_run_script
from app.evaluation.guards import CostBounds, TokenUsage
from app.evaluation.provenance import GitIdentity, corpus_manifest
from app.evaluation.report import (
    SYNTHETIC_BANNER,
    ReportInputs,
    ReportWriteError,
    build_report,
    render_markdown,
    select_manual_review,
    write_report,
)
from app.evaluation.runner import Outcome, QuestionStatus, RunResult
from tests.evaluation.support import (
    Harness,
    MappedRetrieval,
    answer,
    chunk,
    evaluated,
    question,
)

STARTED = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
FINISHED = datetime(2026, 10, 23, 12, 5, tzinfo=UTC)
GIT = GitIdentity("a" * 40, "b" * 40, "c" * 40, worktree_clean=True)
LLM = {
    "implementation": "anthropic",
    "model": "claude-haiku-5-5",
    "structured_output_method": "json_schema",
    "sampling_parameters": "none",
    "max_output_tokens": 1024,
    "max_retries": 1,
    "timeout_seconds": 60.0,
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "run",
    "code",
    "dataset",
    "corpus",
    "database",
    "retrieval",
    "context",
    "prompt",
    "llm",
    "bounds",
    "usage",
    "metrics",
    "questions",
    "manual_review",
    "limitations",
}


def dataset_run(
    behaviors: dict[str, FakeBehavior] | None = None, max_requests: int = 30
) -> tuple[RunResult, MappedRetrieval]:
    dataset = load_dataset()
    retrieval = MappedRetrieval({q.question: [chunk("c1")] for q in dataset.questions})
    harness = Harness(
        retrieval,
        behaviors or build_dry_run_script(dataset.questions),
        max_requests=max_requests,
    )
    return harness.run(dataset.questions), retrieval


def inputs(run: RunResult, synthetic: bool, **changes: Any) -> ReportInputs:
    values: dict[str, Any] = {
        "mode": "dry_run" if synthetic else "live",
        "synthetic": synthetic,
        "started_at": STARTED,
        "finished_at": FINISHED,
        "dataset": load_dataset(),
        "bounds": CostBounds(max_requests=24),
        "limit": 5,
        "git": GIT,
        "corpus": corpus_manifest(CORPUS_ROOT),
        "database": {"used": False},
        "retrieval": {"implementation": "synthetic_hash"},
        "llm": LLM,
        "run": run,
    }
    return ReportInputs(**(values | changes))


@pytest.fixture(scope="module")
def synthetic_report() -> dict[str, Any]:
    run, _ = dataset_run()
    return build_report(inputs(run, synthetic=True))


def test_every_mandatory_field_is_present(synthetic_report: dict[str, Any]) -> None:
    report = synthetic_report

    assert set(report) == TOP_LEVEL_KEYS
    assert report["schema_version"] == 1
    assert report["run"]["runs"] == 1
    assert set(report["run"]) >= {
        "mode",
        "synthetic",
        "started_at",
        "finished_at",
        "complete",
        "abort_reason",
        "aborted_at_question",
        "evaluated_questions",
        "not_evaluated_questions",
    }
    assert report["code"] == {
        "commit_sha": "a" * 40,
        "app_tree": "b" * 40,
        "corpus_tree": "c" * 40,
        "worktree_clean": True,
    }
    assert report["dataset"]["sha256"] == load_dataset().sha256
    assert report["dataset"]["record_count"] == 24
    assert report["corpus"]["chunk_count"] == len(corpus_manifest(CORPUS_ROOT))
    assert report["retrieval"]["limit"] == report["retrieval"]["k"] == 5
    assert report["context"] == {"character_budget": 8000}
    assert set(report["prompt"]) == {"version", "hash"}
    assert report["llm"]["sampling_parameters"] == "none"
    assert report["bounds"]["worst_case_output_tokens"] == 49152
    assert report["bounds"]["ceilings"] == {
        "max_requests": 30,
        "max_retries": 2,
        "max_output_tokens": 2048,
    }
    assert "not a total API cost bound" in report["bounds"]["worst_case_note"]
    assert len(report["questions"]) == 24


def test_keyword_figures_are_labelled_heuristic(
    synthetic_report: dict[str, Any],
) -> None:
    metrics = synthetic_report["metrics"]
    markdown = render_markdown(synthetic_report)

    assert "heuristic_correct" in metrics["answerable"]
    assert "heuristic_false_answer" in metrics["misleading"]
    assert "## Answerable (heuristic)" in markdown
    assert "## Misleading (heuristic)" in markdown


def test_limitations_are_documented(synthetic_report: dict[str, Any]) -> None:
    text = " ".join(synthetic_report["limitations"]).lower()

    for phrase in (
        "single run",
        "nondeterministic",
        "heuristic",
        "no threshold",
        "embedding vectors",
        "unknown",
        "not a total cost bound",
        "synthetic",
    ):
        assert phrase in text


def test_a_synthetic_report_is_marked_everywhere(
    synthetic_report: dict[str, Any],
) -> None:
    markdown = render_markdown(synthetic_report)
    metric_headings = [
        line
        for line in markdown.splitlines()
        if line.startswith("## ")
        and line.split(" (")[0].split(" —")[0]
        in {
            "## Outcomes",
            "## Retrieval",
            "## Answerable",
            "## Unanswerable",
            "## Misleading",
            "## Global",
            "## Similarity scores",
        }
    ]

    assert synthetic_report["run"]["synthetic"] is True
    assert all(block["synthetic"] for block in synthetic_report["metrics"].values())
    assert SYNTHETIC_BANNER in markdown
    assert "NOT A BASELINE" in markdown
    assert len(metric_headings) == 7
    assert all("SYNTHETIC, not a measurement" in line for line in metric_headings)


def test_a_live_report_carries_no_synthetic_marks() -> None:
    run, _ = dataset_run({})
    report = build_report(inputs(run, synthetic=False))
    markdown = render_markdown(report)

    assert report["run"]["synthetic"] is False
    assert "SYNTHETIC" not in markdown
    assert not any(block["synthetic"] for block in report["metrics"].values())


def test_metrics_are_computed_from_the_captured_retrieval_only() -> None:
    run, retrieval = dataset_run()

    build_report(inputs(run, synthetic=True))

    assert len(retrieval.calls) == 24


# --- usage -------------------------------------------------------------------------


def test_usage_is_unavailable_when_no_response_reported_it(
    synthetic_report: dict[str, Any],
) -> None:
    usage = synthetic_report["usage"]

    assert usage["input_tokens"] == usage["output_tokens"] == "unavailable"
    assert usage["responses_with_usage"] == 0
    assert usage["llm_invocations"] == 24
    # A scripted fake makes no provider request: nothing unknown can be billed.
    assert usage["invocations_with_unknown_usage"] == "not applicable: scripted fake"
    assert usage["max_possible_transport_attempts"] == "not applicable: scripted fake"
    assert usage["transport_retries"] == "not observable"


def test_reported_usage_is_summed_and_unknown_usage_is_counted() -> None:
    q = question()
    with_usage = evaluated(
        q, Outcome.ANSWERED, answer_=answer("x [1]", "runbooks/a.md")
    )
    with_usage = replace(with_usage, usage=(TokenUsage(1000, 80),))
    failed = evaluated(q, Outcome.PROVIDER_ERROR)
    run = RunResult(results=(with_usage, failed), llm_invocations=2)

    usage = build_report(inputs(run, synthetic=False))["usage"]

    assert (usage["input_tokens"], usage["output_tokens"]) == (1000, 80)
    assert usage["responses_with_usage"] == 1
    assert usage["invocations_with_unknown_usage"] == 1
    assert usage["max_possible_transport_attempts"] == 2 * (1 + 1)


# --- partial runs --------------------------------------------------------------------


def test_a_partial_run_keeps_completed_results_and_marks_the_rest() -> None:
    run, _ = dataset_run({}, max_requests=3)

    report = build_report(inputs(run, synthetic=True))

    assert report["run"]["complete"] is False
    assert report["run"]["abort_reason"] == "max_requests"
    assert report["run"]["aborted_at_question"] == "answerable-04"
    assert report["run"]["evaluated_questions"] == 3
    assert report["run"]["not_evaluated_questions"] == 21
    statuses = [q["status"] for q in report["questions"]]
    assert statuses == ["evaluated"] * 3 + ["not_evaluated"] * 21
    assert report["metrics"]["answerable"]["n"] == 3
    assert "not_evaluated" in render_markdown(report)


# --- manual review -------------------------------------------------------------------


def test_the_manual_review_sample_is_deterministic() -> None:
    def answered(id_: str, category: EvalCategory) -> Any:
        return evaluated(
            question(id_, category), Outcome.ANSWERED, answer_=answer("x [1]", "a")
        )

    C = EvalCategory
    results = tuple(
        [answered(f"answerable-{i:02d}", C.ANSWERABLE) for i in range(1, 6)]
        + [
            answered("misleading-01", C.MISLEADING),
            answered("misleading-02", C.MISLEADING),
        ]
        + [answered("unanswerable-01", C.UNANSWERABLE)]
    )

    selected = [r.question.id for r in select_manual_review(results)]

    assert selected == [r.question.id for r in select_manual_review(results[::-1])]
    assert selected[:2] == sorted(
        ["misleading-01", "misleading-02"],
        key=lambda i: hashlib.sha256(i.encode()).hexdigest(),
    )
    assert sum(1 for i in selected if i.startswith("answerable")) == 3
    assert selected[-1] == "unanswerable-01"


def test_the_harness_assigns_no_manual_verdicts(
    synthetic_report: dict[str, Any],
) -> None:
    items = synthetic_report["manual_review"]["items"]

    assert items
    assert all("verdict" not in key for item in items for key in item)
    assert "human verdicts" in synthetic_report["manual_review"]["selection_rule"]


# --- rendering and writing -------------------------------------------------------------


def test_model_text_is_escaped_in_markdown() -> None:
    hostile = "Restart \x1b[31mALERT\x07 | ```\n# heading"
    q = question()
    result = evaluated(
        q,
        Outcome.ANSWERED,
        answer_=Answer(
            status=AnswerStatus.ANSWERED,
            answer=hostile,
            citations=answer("x", "runbooks/a.md").citations,
            reason=None,
        ),
    )
    misleading = replace(
        result, question=question("misleading-01", EvalCategory.MISLEADING)
    )
    run = RunResult(results=(result, misleading), llm_invocations=2)

    markdown = render_markdown(build_report(inputs(run, synthetic=False)))

    assert "\x1b" not in markdown and "\x07" not in markdown
    assert "\\u001b[31mALERT" in markdown
    assert "\n# heading" not in markdown


def test_written_files_are_reproducible_from_the_json(
    synthetic_report: dict[str, Any], tmp_path: Path
) -> None:
    target = tmp_path / "out" / "run-1"

    write_report(synthetic_report, target)

    assert sorted(path.name for path in target.iterdir()) == [
        "report.json",
        "report.md",
    ]
    loaded = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert render_markdown(loaded) == (target / "report.md").read_text(encoding="utf-8")
    assert sorted(path.name for path in target.parent.iterdir()) == ["run-1"]


def test_an_existing_output_directory_is_never_overwritten(
    synthetic_report: dict[str, Any], tmp_path: Path
) -> None:
    target = tmp_path / "run"
    target.mkdir()
    (target / "report.json").write_text("previous", encoding="utf-8")

    with pytest.raises(ReportWriteError, match="already exists"):
        write_report(synthetic_report, target)

    assert (target / "report.json").read_text(encoding="utf-8") == "previous"


def test_a_failed_write_leaves_no_report_behind(
    synthetic_report: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_rename(source: Any, target: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(report_module.os, "rename", failing_rename)
    target = tmp_path / "run"

    with pytest.raises(ReportWriteError) as caught:
        write_report(synthetic_report, target)

    assert "PermissionError" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_not_evaluated_questions_have_no_outcome() -> None:
    run, _ = dataset_run({}, max_requests=1)

    questions = build_report(inputs(run, synthetic=True))["questions"]

    later = [q for q in questions if q["status"] == QuestionStatus.NOT_EVALUATED.value]
    assert later and all(q["outcome"] is None for q in later)
