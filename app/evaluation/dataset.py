"""Strict loader of the RAG evaluation dataset (`data/eval/questions.jsonl`).

The dataset is one JSON object per line with exactly the fields documented in
`data/eval/README.md`; blank lines are ignored. Loading checks structure only:
the field set, the field types, the category, non-blank strings, unique ids and
unique question texts. The labelling rules (keywords against the corpus, category
counts, coverage) are the dataset's own integrity tests in
`tests/test_eval_dataset.py`, not the harness's concern.

The dataset carries no version field, so its identity is the SHA-256 digest of
the raw file bytes: any change to the file, whitespace included, changes it.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "eval" / "questions.jsonl"
CORPUS_ROOT = REPO_ROOT / "data" / "knowledge"

_FIELDS = frozenset(
    {
        "id",
        "category",
        "question",
        "expected_sources",
        "required_keywords",
        "forbidden_keywords",
        "notes",
    }
)
_STRING_FIELDS = ("id", "category", "question", "notes")
_LIST_FIELDS = ("expected_sources", "required_keywords", "forbidden_keywords")


class DatasetError(Exception):
    """Raised when the dataset file cannot be read or is structurally invalid."""


class EvalCategory(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    MISLEADING = "misleading"


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: EvalCategory
    question: str
    expected_sources: tuple[str, ...]
    required_keywords: tuple[str, ...]
    forbidden_keywords: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class Dataset:
    """The loaded questions in file order, with the identity of the file."""

    path: str
    sha256: str
    questions: tuple[EvalQuestion, ...]

    @property
    def version(self) -> str:
        return f"sha256:{self.sha256[:12]}"

    def category_counts(self) -> dict[str, int]:
        counts = Counter(question.category for question in self.questions)
        return {category.value: counts[category] for category in EvalCategory}


def load_dataset(path: Path = DATASET_PATH) -> Dataset:
    """Load and validate the dataset at `path`.

    Raises `DatasetError` naming the line and the violated rule.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DatasetError(
            f"cannot read dataset {_display_path(path)}: {type(exc).__name__}"
        ) from None
    return parse_dataset(data, _display_path(path))


def parse_dataset(data: bytes, path: str) -> Dataset:
    """Parse raw dataset bytes; `path` is only recorded, never opened."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DatasetError("dataset is not valid UTF-8") from None

    questions: list[EvalQuestion] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except ValueError as exc:
            raise DatasetError(f"line {line_number}: invalid JSON ({exc})") from None
        questions.append(_to_question(value, line_number))

    if not questions:
        raise DatasetError("dataset contains no records")
    _require_unique(questions)
    return Dataset(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        questions=tuple(questions),
    )


def _to_question(value: Any, line_number: int) -> EvalQuestion:
    def fail(message: str) -> DatasetError:
        return DatasetError(f"line {line_number}: {message}")

    if not isinstance(value, dict):
        raise fail("not a JSON object")
    if set(value) != _FIELDS:
        missing = sorted(_FIELDS - set(value))
        unexpected = sorted(set(value) - _FIELDS)
        raise fail(f"fields: missing {missing}, unexpected {unexpected}")
    for field in _STRING_FIELDS:
        if not isinstance(value[field], str) or not value[field].strip():
            raise fail(f"{field} must be a non-blank string")
    for field in _LIST_FIELDS:
        items = value[field]
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise fail(f"{field} must be a list of non-blank strings")
    try:
        category = EvalCategory(value["category"])
    except ValueError:
        raise fail(f"unknown category {value['category']!r}") from None

    return EvalQuestion(
        id=value["id"],
        category=category,
        question=value["question"],
        expected_sources=tuple(value["expected_sources"]),
        required_keywords=tuple(value["required_keywords"]),
        forbidden_keywords=tuple(value["forbidden_keywords"]),
        notes=value["notes"],
    )


def _require_unique(questions: list[EvalQuestion]) -> None:
    for attribute in ("id", "question"):
        counts = Counter(getattr(question, attribute) for question in questions)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        if duplicates:
            raise DatasetError(f"duplicate {attribute} values: {len(duplicates)}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.name
