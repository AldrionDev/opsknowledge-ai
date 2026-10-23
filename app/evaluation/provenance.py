"""Provenance of an evaluation run: code, packages, corpus and database.

Git identity
------------

`git_identity` records `HEAD`, the tree ids of `app/` and `data/knowledge`, and
whether the worktree is clean (`git status --porcelain`, so untracked files count
and gitignored ones do not). Tree ids are content-addressed: the project squash
merges, so a branch commit can disappear from `main`, but `git rev-parse
<commit>:app` on the merged commit still equals the recorded `app_tree` as long as
the code did not change. Git runs with a fixed argument list, without a shell.

Corpus manifest
---------------

The manifest is one `CorpusChunk` per chunk of the repository corpus, built with
the production loader and chunker: the chunk identity (`chunk_id`), its metadata
(`document_id`, `chunk_index`, `source_path`, `document_type`) and the SHA-256 of
its content. Chunk ids identify positions, not content, which is why the content
hash is part of the comparison. The fingerprint is the SHA-256 of the canonical
JSON of all rows sorted by `chunk_id`.

Database verification
---------------------

Before a live run the stored chunks are read (`SELECT`, never the embedding) and
compared with the manifest **by `chunk_id`**:

- missing: in the manifest, not in the database;
- unexpected: in the database, not in the manifest, for example stale chunks of a
  deleted or renamed document, which ingestion does not remove;
- changed: same `chunk_id`, but different metadata or content hash.

Any difference refuses the run. The check establishes the identity, metadata and
content of the stored chunks. It does **not** establish which model computed the
stored embedding vectors; that remains an assumption of the run.

Read-only access
----------------

`create_read_only_engine` sets `default_transaction_read_only=on` for every
connection, so PostgreSQL rejects any write from the harness. That is an
application safeguard against mistakes in this process. It is not a replacement
for a database role that only has `SELECT` privileges.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from sqlalchemy import Engine, create_engine, select, text

from app.db.models import DocumentChunkRecord
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents

PACKAGES = ("fastembed", "langchain-core", "langchain-anthropic", "anthropic")
READ_ONLY_OPTIONS = "-c default_transaction_read_only=on"
_GIT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class GitIdentity:
    commit_sha: str
    app_tree: str
    corpus_tree: str
    worktree_clean: bool


def git_identity(repo_root: Path) -> GitIdentity | None:
    """The Git identity of `repo_root`, or `None` if Git cannot provide it."""
    commit = _git(repo_root, "rev-parse", "HEAD")
    app_tree = _git(repo_root, "rev-parse", "HEAD:app")
    corpus_tree = _git(repo_root, "rev-parse", "HEAD:data/knowledge")
    status = _git(repo_root, "status", "--porcelain")
    if commit is None or app_tree is None or corpus_tree is None or status is None:
        return None
    return GitIdentity(
        commit_sha=commit,
        app_tree=app_tree,
        corpus_tree=corpus_tree,
        worktree_clean=status == "",
    )


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    source_path: str
    document_type: str
    content_sha256: str


@dataclass(frozen=True)
class CorpusDiff:
    missing: tuple[CorpusChunk, ...]
    unexpected: tuple[CorpusChunk, ...]
    changed: tuple[tuple[CorpusChunk, CorpusChunk], ...]

    @property
    def matches(self) -> bool:
        return not (self.missing or self.unexpected or self.changed)


def corpus_manifest(corpus_root: Path) -> dict[str, CorpusChunk]:
    """The manifest of the corpus at `corpus_root`, keyed by `chunk_id`."""
    return {
        chunk.chunk_id: _corpus_chunk(
            chunk.chunk_id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.source_path,
            str(chunk.document_type),
            chunk.content,
        )
        for document in load_knowledge_documents(corpus_root)
        for chunk in chunk_document(document)
    }


def corpus_fingerprint(chunks: Mapping[str, CorpusChunk]) -> str:
    rows = [asdict(chunks[chunk_id]) for chunk_id in sorted(chunks)]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def document_count(chunks: Mapping[str, CorpusChunk]) -> int:
    return len({chunk.source_path for chunk in chunks.values()})


def compare_corpus(
    expected: Mapping[str, CorpusChunk], actual: Mapping[str, CorpusChunk]
) -> CorpusDiff:
    """Compare two manifests by chunk identity; see the module docstring."""
    return CorpusDiff(
        missing=tuple(expected[i] for i in sorted(expected.keys() - actual.keys())),
        unexpected=tuple(actual[i] for i in sorted(actual.keys() - expected.keys())),
        changed=tuple(
            (expected[i], actual[i])
            for i in sorted(expected.keys() & actual.keys())
            if expected[i] != actual[i]
        ),
    )


def create_read_only_engine(database_url: str) -> Engine:
    """An engine whose every transaction is read-only; does not connect."""
    return create_engine(database_url, connect_args={"options": READ_ONLY_OPTIONS})


def read_database_corpus(engine: Engine) -> dict[str, CorpusChunk]:
    """The manifest of the stored chunks, keyed by `chunk_id`; embeddings are not read."""
    record = DocumentChunkRecord
    statement = select(
        record.chunk_id,
        record.document_id,
        record.chunk_index,
        record.source_path,
        record.document_type,
        record.content,
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).all()
    return {row.chunk_id: _corpus_chunk(*row) for row in rows}


def read_migration_revision(engine: Engine) -> str | None:
    """The applied Alembic revision, or `None` if the version table is missing or empty."""
    with engine.connect() as connection:
        exists = connection.execute(
            text("SELECT to_regclass('alembic_version') IS NOT NULL")
        ).scalar_one()
        if not exists:
            return None
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one_or_none()
    return None if revision is None else str(revision)


def _corpus_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    source_path: str,
    document_type: str,
    content: str,
) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=source_path,
        document_type=document_type,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
