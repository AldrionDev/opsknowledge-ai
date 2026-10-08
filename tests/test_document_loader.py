from pathlib import Path

import pytest

from app.ingestion.loader import (
    DocumentLoadError,
    load_knowledge_documents,
    normalize_content,
)
from app.ingestion.models import DocumentType, build_document_id

REAL_CORPUS_ROOT = Path(__file__).resolve().parent.parent / "data" / "knowledge"


def write(root: Path, relative_path: str, content: str | bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8", newline="")
    else:
        path.write_bytes(content)
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "runbooks").mkdir()
    (tmp_path / "incidents").mkdir()
    return tmp_path


# --- normalize_content ---


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_normalize_content_converts_line_endings_to_lf(newline: str) -> None:
    text = newline.join(["# Title", "", "Body", "More"])

    assert normalize_content(text) == "# Title\n\nBody\nMore"


def test_normalize_content_removes_leading_and_trailing_blank_lines() -> None:
    text = "\n  \n\t\n# Title\n\nBody\n\n   \n\n"

    assert normalize_content(text) == "# Title\n\nBody"


def test_normalize_content_preserves_internal_and_indentation_whitespace() -> None:
    text = "    indented first line  \n\n\n\tTabbed\nHard break  \nlast line"

    assert normalize_content(text) == text


def test_normalize_content_preserves_trailing_spaces_of_last_nonblank_line() -> None:
    assert normalize_content("# Title\nHard break  \n\n") == "# Title\nHard break  "


def test_normalize_content_is_idempotent() -> None:
    once = normalize_content("\r\n\r\n# Title\r\n\r\nBody  \r\n\r\n")

    assert normalize_content(once) == once


def test_normalize_content_returns_empty_string_for_blank_text() -> None:
    assert normalize_content(" \n\t\r\n\n") == ""


# --- load_knowledge_documents ---


def test_loads_multiple_documents(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", "# Runbook A\n")
    write(corpus, "runbooks/nested/b.md", "# Runbook B\n")
    write(corpus, "incidents/c.md", "# Incident C\n")

    documents = load_knowledge_documents(corpus)

    assert [(d.source_path, d.content) for d in documents] == [
        ("incidents/c.md", "# Incident C"),
        ("runbooks/a.md", "# Runbook A"),
        ("runbooks/nested/b.md", "# Runbook B"),
    ]


def test_detects_runbook_document_type(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", "# A\n")
    write(corpus, "runbooks/nested/b.md", "# B\n")

    documents = load_knowledge_documents(corpus)

    assert {d.document_type for d in documents} == {DocumentType.RUNBOOK}


def test_detects_incident_document_type(corpus: Path) -> None:
    write(corpus, "incidents/a.md", "# A\n")
    write(corpus, "incidents/nested/b.md", "# B\n")

    documents = load_knowledge_documents(corpus)

    assert {d.document_type for d in documents} == {DocumentType.INCIDENT}


def test_returns_documents_in_deterministic_order(corpus: Path) -> None:
    paths = ["runbooks/z.md", "incidents/b.md", "runbooks/a.md", "incidents/a.md"]
    for relative_path in paths:
        write(corpus, relative_path, f"# {relative_path}\n")

    first = [d.source_path for d in load_knowledge_documents(corpus)]
    second = [d.source_path for d in load_knowledge_documents(corpus)]

    assert first == [
        "incidents/a.md",
        "incidents/b.md",
        "runbooks/a.md",
        "runbooks/z.md",
    ]
    assert second == first


def test_document_ids_are_independent_of_corpus_location(tmp_path: Path) -> None:
    roots = [tmp_path / "checkout-one", tmp_path / "elsewhere" / "checkout-two"]
    for root in roots:
        write(root, "runbooks/a.md", "# A\n")
        write(root, "incidents/b.md", "# B\n")

    ids_per_root = [
        {d.source_path: d.document_id for d in load_knowledge_documents(root)}
        for root in roots
    ]

    assert ids_per_root[0] == ids_per_root[1]
    assert ids_per_root[0]["runbooks/a.md"] == build_document_id("runbooks/a.md")
    assert len(set(ids_per_root[0].values())) == 2


def test_source_paths_are_relative_posix_paths(corpus: Path) -> None:
    write(corpus, "runbooks/nested/deeper/a.md", "# A\n")

    (document,) = load_knowledge_documents(corpus)

    assert document.source_path == "runbooks/nested/deeper/a.md"
    assert str(corpus) not in document.source_path
    assert "\\" not in document.source_path
    assert not Path(document.source_path).is_absolute()


def test_normalizes_content_of_loaded_documents(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", "\r\n# A\r\n\r\nBody  \r\nEnd\r\n\r\n")

    (document,) = load_knowledge_documents(corpus)

    assert document.content == "# A\n\nBody  \nEnd"


def test_reads_utf8_content_and_strips_byte_order_mark(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", b"\xef\xbb\xbf# Caf\xc3\xa9 \xe2\x80\x94 runbook\n")

    (document,) = load_knowledge_documents(corpus)

    assert document.content == "# Café — runbook"


@pytest.mark.parametrize("content", ["", "   \n\t\r\n\n"])
def test_rejects_empty_or_whitespace_only_documents(corpus: Path, content: str) -> None:
    write(corpus, "runbooks/ok.md", "# OK\n")
    write(corpus, "incidents/blank.md", content)

    with pytest.raises(DocumentLoadError, match=r"incidents/blank\.md"):
        load_knowledge_documents(corpus)


def test_rejects_invalid_utf8_documents(corpus: Path) -> None:
    write(corpus, "runbooks/bad.md", b"# Title\n\xff\xfe invalid\n")

    with pytest.raises(DocumentLoadError, match=r"runbooks/bad\.md") as exc_info:
        load_knowledge_documents(corpus)

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_reports_unreadable_documents(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(corpus, "runbooks/locked.md", "# Locked\n")
    original_read_bytes = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self.name == "locked.md":
            raise PermissionError("denied")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(DocumentLoadError, match=r"runbooks/locked\.md") as exc_info:
        load_knowledge_documents(corpus)

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_error_messages_do_not_contain_absolute_document_paths(corpus: Path) -> None:
    write(corpus, "runbooks/empty.md", "")

    with pytest.raises(DocumentLoadError) as exc_info:
        load_knowledge_documents(corpus)

    assert str(corpus) not in str(exc_info.value)


def test_ignores_non_markdown_files(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", "# A\n")
    write(corpus, "runbooks/notes.txt", "not markdown")
    write(corpus, "runbooks/a.md.bak", "# backup")
    write(corpus, "runbooks/upper.MD", "# upper-case suffix")
    write(corpus, "runbooks/image.png", b"\x89PNG\xff\xfe")
    write(corpus, "incidents/data.json", "{}")

    documents = load_knowledge_documents(corpus)

    assert [d.source_path for d in documents] == ["runbooks/a.md"]


def test_ignores_files_outside_document_directories(corpus: Path) -> None:
    write(corpus, "runbooks/a.md", "# A\n")
    write(corpus, "drafts/b.md", "# Draft\n")
    write(corpus, "top-level.md", "# Top level\n")

    documents = load_knowledge_documents(corpus)

    assert [d.source_path for d in documents] == ["runbooks/a.md"]


def test_excludes_corpus_level_readme(corpus: Path) -> None:
    write(corpus, "README.md", "# Corpus documentation\n")
    write(corpus, "runbooks/a.md", "# A\n")

    documents = load_knowledge_documents(corpus)

    assert [d.source_path for d in documents] == ["runbooks/a.md"]


def test_does_not_follow_symlinks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    outside = tmp_path / "outside"
    write(corpus, "runbooks/a.md", "# A\n")
    (corpus / "incidents").mkdir()
    write(outside, "secret.md", "# Outside the corpus\n")
    write(outside, "linked-dir/inner.md", "# Inside a linked directory\n")
    try:
        (corpus / "runbooks" / "linked-file.md").symlink_to(outside / "secret.md")
        (corpus / "runbooks" / "linked-dir").symlink_to(
            outside / "linked-dir", target_is_directory=True
        )
    except OSError:
        pytest.skip("Symbolic links are not supported in this environment")

    documents = load_knowledge_documents(corpus)

    assert [d.source_path for d in documents] == ["runbooks/a.md"]


@pytest.mark.parametrize("category", ["runbooks", "incidents"])
def test_rejects_symlinked_document_directory(tmp_path: Path, category: str) -> None:
    corpus = tmp_path / "corpus"
    outside = tmp_path / "outside"
    other = "incidents" if category == "runbooks" else "runbooks"
    (corpus / other).mkdir(parents=True)
    write(outside, "secret.md", "# Outside the corpus\n")
    try:
        (corpus / category).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links are not supported in this environment")

    with pytest.raises(DocumentLoadError, match=category):
        load_knowledge_documents(corpus)


def test_returns_empty_list_for_empty_document_directories(corpus: Path) -> None:
    assert load_knowledge_documents(corpus) == []


def test_rejects_missing_corpus_root(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_knowledge_documents(tmp_path / "missing")


def test_rejects_corpus_root_that_is_a_file(tmp_path: Path) -> None:
    file_path = write(tmp_path, "corpus.md", "# Not a directory\n")

    with pytest.raises(DocumentLoadError):
        load_knowledge_documents(file_path)


@pytest.mark.parametrize("missing", ["runbooks", "incidents"])
def test_rejects_missing_document_directory(corpus: Path, missing: str) -> None:
    (corpus / missing).rmdir()

    with pytest.raises(DocumentLoadError, match=missing):
        load_knowledge_documents(corpus)


# --- real corpus smoke test ---


def test_real_corpus_loads_deterministically() -> None:
    documents = load_knowledge_documents(REAL_CORPUS_ROOT)

    assert documents
    assert load_knowledge_documents(REAL_CORPUS_ROOT) == documents
    assert [d.source_path for d in documents] == sorted(
        d.source_path for d in documents
    )
    assert len({d.document_id for d in documents}) == len(documents)
    directory_by_type = {
        DocumentType.RUNBOOK: "runbooks/",
        DocumentType.INCIDENT: "incidents/",
    }
    for document in documents:
        assert document.source_path.startswith(
            directory_by_type[document.document_type]
        )
        assert not Path(document.source_path).is_absolute()
        assert Path(document.source_path).name != "README.md"
        assert document.content
