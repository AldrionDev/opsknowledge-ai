from pathlib import Path

from app.ingestion.models import DocumentType, KnowledgeDocument, build_document_id

# Corpus subdirectories that contain knowledge documents. Anything else under
# the corpus root (for example the corpus-level README.md) is not loaded.
DOCUMENT_DIRECTORIES: dict[str, DocumentType] = {
    "runbooks": DocumentType.RUNBOOK,
    "incidents": DocumentType.INCIDENT,
}

MARKDOWN_SUFFIX = ".md"


class DocumentLoadError(Exception):
    """Raised when the corpus or one of its documents cannot be loaded."""


def normalize_content(text: str) -> str:
    """Apply the minimal, conservative normalization to document text.

    - line endings (`\\r\\n` and `\\r`) become `\\n`
    - leading and trailing blank lines are removed

    Whitespace inside lines, including the trailing spaces of the last
    non-blank line (a Markdown hard line break), is preserved.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def load_knowledge_documents(corpus_root: Path) -> list[KnowledgeDocument]:
    """Load the knowledge documents under `corpus_root`.

    Markdown files are discovered recursively in `runbooks/` and `incidents/`;
    both directories are required and must not be symbolic links. Symbolic
    links to files or directories inside them are not followed, so content
    outside the corpus root is never loaded. Files are read as UTF-8.

    Documents are returned ordered by `source_path`. Loading stops at the first
    problem and raises `DocumentLoadError` that identifies the offending path
    relative to the corpus root.
    """
    if not corpus_root.is_dir():
        raise DocumentLoadError(f"Corpus root is not a directory: {corpus_root}")

    files: list[tuple[str, Path, DocumentType]] = []
    for directory_name, document_type in DOCUMENT_DIRECTORIES.items():
        directory = corpus_root / directory_name
        if directory.is_symlink():
            raise DocumentLoadError(
                f"Document directory must not be a symbolic link: {directory_name}/"
            )
        if not directory.is_dir():
            raise DocumentLoadError(f"Missing document directory: {directory_name}/")
        for path in directory.rglob("*"):
            if (
                path.suffix == MARKDOWN_SUFFIX
                and path.is_file()
                and not path.is_symlink()
            ):
                source_path = path.relative_to(corpus_root).as_posix()
                files.append((source_path, path, document_type))

    return [
        _load_document(source_path, path, document_type)
        for source_path, path, document_type in sorted(files, key=lambda f: f[0])
    ]


def _load_document(
    source_path: str, path: Path, document_type: DocumentType
) -> KnowledgeDocument:
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"Document is not valid UTF-8: {source_path}") from exc
    except OSError as exc:
        raise DocumentLoadError(f"Document cannot be read: {source_path}") from exc

    content = normalize_content(text)
    if not content:
        raise DocumentLoadError(f"Document is empty: {source_path}")

    return KnowledgeDocument(
        document_id=build_document_id(source_path),
        source_path=source_path,
        document_type=document_type,
        content=content,
    )
