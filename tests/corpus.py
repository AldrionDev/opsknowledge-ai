from pathlib import Path

PARAGRAPH_SIZE = 600  # two paragraphs never fit into one default-sized chunk


def multi_chunk_text(tag: str, chunks: int) -> str:
    """Markdown text that the default chunker splits into `chunks` chunks."""
    return "\n\n".join(
        f"{tag} paragraph {index}. ".ljust(PARAGRAPH_SIZE, "x")
        for index in range(chunks)
    )


def write_corpus(root: Path, documents: dict[str, str]) -> Path:
    """Create a corpus below `root`; keys are paths relative to the corpus root.

    Both document directories always exist, as the loader requires.
    """
    (root / "runbooks").mkdir(parents=True, exist_ok=True)
    (root / "incidents").mkdir(parents=True, exist_ok=True)
    for source_path, content in documents.items():
        path = root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root
