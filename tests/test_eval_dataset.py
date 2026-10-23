"""Offline schema and integrity checks of the RAG evaluation dataset.

The dataset in `data/eval/questions.jsonl` is plain data; these tests are its
validator. They read only repository files and never need a database,
embeddings, a language model, or network access. Keyword checks use the
dataset's matching convention: case-insensitive substring matching.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from app.ingestion.loader import load_knowledge_documents

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "eval" / "questions.jsonl"
CORPUS_ROOT = REPO_ROOT / "data" / "knowledge"

FIELDS = (
    "id",
    "category",
    "question",
    "expected_sources",
    "required_keywords",
    "forbidden_keywords",
    "notes",
)
STRING_FIELDS = ("id", "category", "question", "notes")
LIST_FIELDS = ("expected_sources", "required_keywords", "forbidden_keywords")
CATEGORY_COUNTS = {"answerable": 10, "unanswerable": 7, "misleading": 7}

# Whether a list field must be non-empty (True), empty (False), or may be either
# (None). A misleading record with expected sources asks for a correction of
# its premise; one without expected sources must be declined.
LIST_RULES: dict[str, dict[str, bool | None]] = {
    "answerable": {
        "expected_sources": True,
        "required_keywords": True,
        "forbidden_keywords": False,
    },
    "unanswerable": {
        "expected_sources": False,
        "required_keywords": False,
        "forbidden_keywords": False,
    },
    "misleading": {
        "expected_sources": None,
        "required_keywords": None,
        "forbidden_keywords": True,
    },
}

ID_PATTERN = re.compile(r"(answerable|unanswerable|misleading)-\d{2}")
INCIDENT_ID_PATTERN = re.compile(r"INC-\d{4}-\d{3}")

RUNBOOK_502 = "runbooks/http-502-behind-load-balancer.md"
INCIDENT_001 = (
    "incidents/inc-2025-001-report-service-imagepullbackoff-node-disk-full.md"
)


# --- validator ---


def parse_dataset(text: str) -> list[dict[str, Any]]:
    """Parse JSONL text: one JSON object per line, blank lines are ignored."""
    records = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: not a JSON object")
        records.append(value)
    return records


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def contains(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()


def record_errors(record: dict[str, Any], corpus: dict[str, str]) -> list[str]:
    """Return every problem of one record; `corpus` maps source path to content."""
    missing = sorted(set(FIELDS) - set(record))
    unexpected = sorted(set(record) - set(FIELDS))
    if missing or unexpected:
        return [f"fields: missing {missing}, unexpected {unexpected}"]

    errors = [
        f"{field}: must be a non-blank string"
        for field in STRING_FIELDS
        if not isinstance(record[field], str) or not record[field].strip()
    ]
    errors += [
        f"{field}: must be a list of strings"
        for field in LIST_FIELDS
        if not isinstance(record[field], list)
        or not all(isinstance(item, str) for item in record[field])
    ]
    if errors:
        return errors  # the remaining rules rely on the field types

    category: str = record["category"]
    question: str = record["question"]
    sources: list[str] = record["expected_sources"]
    required: list[str] = record["required_keywords"]
    forbidden: list[str] = record["forbidden_keywords"]

    if category not in LIST_RULES:
        return [f"category: unknown category {category!r}"]
    if not ID_PATTERN.fullmatch(record["id"]):
        errors.append(f"id: invalid format {record['id']!r}")
    elif record["id"].rsplit("-", 1)[0] != category:
        errors.append(f"id: prefix does not match category {category!r}")

    for field in LIST_FIELDS:
        items: list[str] = record[field]
        if any(not item.strip() or item != item.strip() for item in items):
            errors.append(f"{field}: items must be non-blank and stripped")
        if len({item.casefold() for item in items}) != len(items):
            errors.append(f"{field}: duplicate items")
        rule = LIST_RULES[category][field]
        if rule is True and not items:
            errors.append(f"{field}: must not be empty for {category}")
        if rule is False and items:
            errors.append(f"{field}: must be empty for {category}")

    errors += [
        f"expected_sources: unknown source {source!r}"
        for source in sources
        if source not in corpus
    ]
    source_texts = [corpus[source] for source in sources if source in corpus]

    if required and not sources:
        errors.append("required_keywords: require expected_sources")
    for keyword in required:
        if not any(contains(text, keyword) for text in source_texts):
            errors.append(
                f"required_keywords: {keyword!r} not found in any expected source"
            )
    for field, keywords in (
        ("required_keywords", required),
        ("forbidden_keywords", forbidden),
    ):
        errors += [
            f"{field}: {keyword!r} appears in the question"
            for keyword in keywords
            if contains(question, keyword)
        ]
    for keyword in forbidden:
        if any(contains(text, keyword) for text in source_texts):
            errors.append(
                f"forbidden_keywords: {keyword!r} occurs in an expected source"
            )
        if not sources and any(contains(text, keyword) for text in corpus.values()):
            errors.append(f"forbidden_keywords: {keyword!r} occurs in the corpus")

    if sources:
        errors += [
            f"question: incident ID {incident_id} not in any expected source"
            for incident_id in INCIDENT_ID_PATTERN.findall(question)
            if not any(incident_id in text for text in source_texts)
        ]
    return errors


def dataset_errors(records: list[dict[str, Any]], corpus: dict[str, str]) -> list[str]:
    """Return the problems of the dataset as a whole; records must be valid."""
    errors = []
    id_counts = Counter(record["id"] for record in records)
    errors += [f"duplicate id {id_!r}" for id_, count in id_counts.items() if count > 1]

    category_counts = Counter(record["category"] for record in records)
    if category_counts != CATEGORY_COUNTS:
        errors.append(f"category counts {dict(category_counts)} != {CATEGORY_COUNTS}")

    covered = {
        source
        for record in records
        if record["category"] == "answerable"
        for source in record["expected_sources"]
    }
    errors += [
        f"document not covered by an answerable record: {source}"
        for source in sorted(set(corpus) - covered)
    ]
    return errors


def dataset_file_errors(path: Path, corpus: dict[str, str]) -> list[str]:
    """Validate a dataset file; whole-dataset rules run only on valid records."""
    records = parse_dataset(path.read_text(encoding="utf-8"))
    errors = [
        f"{record.get('id', f'record {index}')}: {error}"
        for index, record in enumerate(records, start=1)
        for error in record_errors(record, corpus)
    ]
    return errors or dataset_errors(records, corpus)


# --- fixtures ---


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    return {
        document.source_path: document.content
        for document in load_knowledge_documents(CORPUS_ROOT)
    }


@pytest.fixture(scope="module")
def records() -> list[dict[str, Any]]:
    return parse_dataset(DATASET_PATH.read_text(encoding="utf-8"))


def find(records: list[dict[str, Any]], record_id: str) -> dict[str, Any]:
    (record,) = (record for record in records if record["id"] == record_id)
    return record


# --- the committed dataset ---


def test_dataset_is_valid(corpus: dict[str, str]) -> None:
    assert dataset_file_errors(DATASET_PATH, corpus) == []


def test_records_are_ordered_by_category_and_number(
    records: list[dict[str, Any]],
) -> None:
    expected_ids = [
        f"{category}-{number:02d}"
        for category, count in CATEGORY_COUNTS.items()
        for number in range(1, count + 1)
    ]

    assert [record["id"] for record in records] == expected_ids


def test_dataset_file_uses_one_line_per_record_with_lf_endings() -> None:
    raw = DATASET_PATH.read_bytes()

    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert all(line.strip() for line in raw.decode("utf-8").split("\n")[:-1])


# --- parsing ---


def test_parse_ignores_blank_lines() -> None:
    text = '{"a": 1}\n\n   \n{"b": 2}\n'

    assert parse_dataset(text) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param('{"a": 1}\n{"a": \n', "line 2", id="invalid-json"),
        pytest.param('["a", "b"]\n', "line 1: not a JSON object", id="array"),
        pytest.param('"text"\n', "line 1: not a JSON object", id="string"),
        pytest.param('{"a": 1, "a": 2}\n', "duplicate keys", id="duplicate-key"),
    ],
)
def test_parse_rejects_malformed_lines(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_dataset(text)


# --- record validation (in-memory mutations of valid records) ---


@pytest.mark.parametrize("field", FIELDS)
def test_rejects_record_with_missing_field(
    records: list[dict[str, Any]], corpus: dict[str, str], field: str
) -> None:
    record = dict(find(records, "answerable-01"))
    del record[field]

    assert record_errors(record, corpus) == [
        f"fields: missing [{field!r}], unexpected []"
    ]


def test_rejects_record_with_unexpected_field(
    records: list[dict[str, Any]], corpus: dict[str, str]
) -> None:
    record = {**find(records, "answerable-01"), "created_at": "2025-01-01"}

    assert record_errors(record, corpus) == [
        "fields: missing [], unexpected ['created_at']"
    ]


INVALID_CHANGES = [
    # field types and blank values
    ("answerable-01", {"question": 42}, "question: must be a non-blank string"),
    ("answerable-01", {"question": "  "}, "question: must be a non-blank string"),
    ("answerable-01", {"notes": None}, "notes: must be a non-blank string"),
    ("answerable-01", {"id": True}, "id: must be a non-blank string"),
    (
        "answerable-01",
        {"expected_sources": RUNBOOK_502},
        "expected_sources: must be a list of strings",
    ),
    (
        "answerable-01",
        {"required_keywords": ["preStop", 1]},
        "required_keywords: must be a list of strings",
    ),
    # category and id
    ("answerable-01", {"category": "factual"}, "category: unknown category"),
    ("answerable-01", {"category": "Answerable"}, "category: unknown category"),
    ("answerable-01", {"id": "answerable-1"}, "id: invalid format"),
    ("answerable-01", {"id": "answerable-01-x"}, "id: invalid format"),
    ("answerable-01", {"id": "misleading-01"}, "id: prefix does not match"),
    # expected source paths must be loadable corpus-relative source paths
    *(
        ("answerable-01", {"expected_sources": [path]}, "unknown source")
        for path in [
            "runbooks/does-not-exist.md",
            f"data/knowledge/{RUNBOOK_502}",
            f"/{RUNBOOK_502}",
            f"runbooks/../{RUNBOOK_502}",
            RUNBOOK_502.replace("/", "\\"),
            RUNBOOK_502.removesuffix(".md"),
            "README.md",
        ]
    ),
    (
        "answerable-01",
        {"expected_sources": [RUNBOOK_502, RUNBOOK_502]},
        "expected_sources: duplicate items",
    ),
    # keyword list items
    (
        "answerable-01",
        {"required_keywords": [" preStop"]},
        "required_keywords: items must be non-blank and stripped",
    ),
    (
        "answerable-01",
        {"required_keywords": ["preStop", ""]},
        "required_keywords: items must be non-blank and stripped",
    ),
    (
        "answerable-01",
        {"required_keywords": ["preStop", "PRESTOP"]},
        "required_keywords: duplicate items",
    ),
    # answerable rules
    (
        "answerable-01",
        {"expected_sources": []},
        "expected_sources: must not be empty for answerable",
    ),
    (
        "answerable-01",
        {"required_keywords": []},
        "required_keywords: must not be empty for answerable",
    ),
    (
        "answerable-01",
        {"forbidden_keywords": ["sidecar"]},
        "forbidden_keywords: must be empty for answerable",
    ),
    (
        "answerable-01",
        {"required_keywords": ["preStop", "drain hook"]},
        "'drain hook' not found in any expected source",
    ),
    (
        "answerable-01",
        {"required_keywords": ["preStop", "load balancer"]},
        "required_keywords: 'load balancer' appears in the question",
    ),
    (
        "answerable-05",
        {"question": "What was the root cause of INC-2025-002?"},
        "incident ID INC-2025-002 not in any expected source",
    ),
    # unanswerable rules
    (
        "unanswerable-01",
        {"expected_sources": [INCIDENT_001]},
        "expected_sources: must be empty for unanswerable",
    ),
    (
        "unanswerable-01",
        {"forbidden_keywords": ["Alice"]},
        "forbidden_keywords: must be empty for unanswerable",
    ),
    # misleading rules
    (
        "misleading-01",
        {"forbidden_keywords": []},
        "forbidden_keywords: must not be empty for misleading",
    ),
    (
        "misleading-01",
        {"forbidden_keywords": ["renewed"]},
        "forbidden_keywords: 'renewed' appears in the question",
    ),
    (
        "misleading-01",
        {"forbidden_keywords": ["no space left on device"]},
        "'no space left on device' occurs in an expected source",
    ),
    (
        "misleading-06",
        {"forbidden_keywords": ["OOMKilled"]},
        "forbidden_keywords: 'OOMKilled' occurs in the corpus",
    ),
    (
        "misleading-06",
        {"required_keywords": ["CoreDNS"]},
        "required_keywords: require expected_sources",
    ),
]


@pytest.mark.parametrize(("record_id", "changes", "message"), INVALID_CHANGES)
def test_rejects_invalid_record(
    records: list[dict[str, Any]],
    corpus: dict[str, str],
    record_id: str,
    changes: dict[str, Any],
    message: str,
) -> None:
    record = {**find(records, record_id), **changes}

    errors = record_errors(record, corpus)

    assert any(message in error for error in errors), errors


# --- dataset validation ---


def test_rejects_duplicate_ids(
    records: list[dict[str, Any]], corpus: dict[str, str]
) -> None:
    duplicate = {**find(records, "unanswerable-02"), "id": "unanswerable-01"}
    mutated = [duplicate if r["id"] == "unanswerable-02" else r for r in records]

    assert dataset_errors(mutated, corpus) == ["duplicate id 'unanswerable-01'"]


def test_rejects_category_counts_other_than_documented(
    records: list[dict[str, Any]], corpus: dict[str, str]
) -> None:
    mutated = [record for record in records if record["id"] != "misleading-07"]

    errors = dataset_errors(mutated, corpus)

    assert len(errors) == 1
    assert errors[0].startswith("category counts")


def test_rejects_document_without_answerable_coverage(
    records: list[dict[str, Any]], corpus: dict[str, str]
) -> None:
    mutated = [
        {**record, "expected_sources": [RUNBOOK_502]}
        if record["id"] == "answerable-04"
        else record
        for record in records
    ]

    assert dataset_errors(mutated, corpus) == [
        "document not covered by an answerable record: "
        "runbooks/linux-disk-space-exhaustion.md"
    ]


# --- whole-file checks on broken copies ---


def broken_copy(tmp_path: Path, old: str, new: str) -> Path:
    text = DATASET_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "questions.jsonl"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_file_with_broken_source_path_is_rejected(
    tmp_path: Path, corpus: dict[str, str]
) -> None:
    path = broken_copy(tmp_path, INCIDENT_001, "incidents/inc-2025-999-missing.md")

    errors = dataset_file_errors(path, corpus)

    assert "answerable-05: expected_sources: unknown source" in " ".join(errors)


def test_file_with_missing_field_is_rejected(
    tmp_path: Path, corpus: dict[str, str]
) -> None:
    path = broken_copy(tmp_path, '"notes": ', '"note": ')

    errors = dataset_file_errors(path, corpus)

    assert errors == ["answerable-01: fields: missing ['notes'], unexpected ['note']"]


def test_file_with_malformed_line_is_rejected(tmp_path: Path) -> None:
    path = broken_copy(tmp_path, '{"id": "answerable-03"', '{"id" "answerable-03"')

    with pytest.raises(ValueError, match="line 3"):
        parse_dataset(path.read_text(encoding="utf-8"))
