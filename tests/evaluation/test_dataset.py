"""Strict dataset loading. Mutations are made on in-memory bytes, never on the file."""

import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    DATASET_PATH,
    DatasetError,
    EvalCategory,
    load_dataset,
    parse_dataset,
)

RAW = DATASET_PATH.read_bytes()
LINES = [line for line in RAW.decode("utf-8").split("\n") if line.strip()]


def with_first_record(**changes: object) -> bytes:
    record = json.loads(LINES[0])
    for key, value in changes.items():
        if value is None:
            del record[key]
        else:
            record[key] = value
    return "\n".join([json.dumps(record), *LINES[1:]]).encode("utf-8")


def test_the_repository_dataset_loads_in_file_order() -> None:
    dataset = load_dataset()

    assert len(dataset.questions) == 24
    assert dataset.category_counts() == {
        "answerable": 10,
        "unanswerable": 7,
        "misleading": 7,
    }
    assert dataset.questions[0].id == "answerable-01"
    assert dataset.questions[-1].id == "misleading-07"
    assert dataset.path == "data/eval/questions.jsonl"


def test_the_identity_is_the_sha256_of_the_raw_bytes() -> None:
    dataset = load_dataset()

    assert dataset.sha256 == hashlib.sha256(RAW).hexdigest()
    assert dataset.version == f"sha256:{dataset.sha256[:12]}"


def test_any_byte_change_changes_the_identity() -> None:
    assert parse_dataset(RAW + b"\n", "x").sha256 != load_dataset().sha256


def test_records_keep_their_labels() -> None:
    misleading_06 = next(q for q in load_dataset().questions if q.id == "misleading-06")

    assert misleading_06.category is EvalCategory.MISLEADING
    assert misleading_06.expected_sources == ()
    assert misleading_06.forbidden_keywords == ("CoreDNS", "ndots")


def test_blank_lines_are_ignored() -> None:
    padded = b"\n\n" + RAW.replace(b"\n", b"\n  \n")

    assert len(parse_dataset(padded, "x").questions) == 24


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (with_first_record(notes=None), "missing ['notes']"),
        (with_first_record(extra="x"), "unexpected ['extra']"),
        (with_first_record(question="  "), "question must be a non-blank string"),
        (with_first_record(id=7), "id must be a non-blank string"),
        (with_first_record(expected_sources="runbooks/a.md"), "expected_sources"),
        (with_first_record(required_keywords=["ok", " "]), "required_keywords"),
        (with_first_record(forbidden_keywords=[1]), "forbidden_keywords"),
        (with_first_record(category="trivia"), "unknown category"),
        (b"{not json}\n", "invalid JSON"),
        (b"[1, 2]\n", "not a JSON object"),
        (b'{"id": "a", "id": "b"}\n', "duplicate keys"),
        (b"\n  \n", "no records"),
        (b"\xff\xfe", "not valid UTF-8"),
        ((LINES[0] + "\n" + LINES[0]).encode(), "duplicate id"),
    ],
)
def test_invalid_datasets_are_rejected(data: bytes, message: str) -> None:
    with pytest.raises(DatasetError) as caught:
        parse_dataset(data, "x")

    assert message in str(caught.value)


def test_duplicate_question_text_is_rejected() -> None:
    first = json.loads(LINES[0])
    second = json.loads(LINES[1]) | {"question": first["question"]}
    data = "\n".join([json.dumps(first), json.dumps(second)]).encode()

    with pytest.raises(DatasetError, match="duplicate question"):
        parse_dataset(data, "x")


def test_a_missing_file_is_reported_without_os_details(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as caught:
        load_dataset(tmp_path / "missing.jsonl")

    assert str(caught.value) == "cannot read dataset missing.jsonl: FileNotFoundError"
