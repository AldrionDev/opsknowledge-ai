"""Git identity, package versions, corpus manifest and its comparison (no database)."""

import subprocess
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from app.evaluation import provenance
from app.evaluation.dataset import CORPUS_ROOT, REPO_ROOT
from app.evaluation.provenance import (
    READ_ONLY_OPTIONS,
    CorpusChunk,
    compare_corpus,
    corpus_fingerprint,
    corpus_manifest,
    create_read_only_engine,
    document_count,
    git_identity,
    package_versions,
)
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_knowledge_documents
from tests.corpus import write_corpus


def test_the_manifest_matches_the_production_chunker() -> None:
    manifest = corpus_manifest(CORPUS_ROOT)
    chunks = [
        c for d in load_knowledge_documents(CORPUS_ROOT) for c in chunk_document(d)
    ]

    assert set(manifest) == {c.chunk_id for c in chunks}
    assert document_count(manifest) == 8
    sample = chunks[0]
    assert manifest[sample.chunk_id].source_path == sample.source_path
    assert manifest[sample.chunk_id].document_type == str(sample.document_type)


def test_the_fingerprint_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "c", {"runbooks/a.md": "# A\n\nFirst version text."})
    before = corpus_fingerprint(corpus_manifest(root))

    assert corpus_fingerprint(corpus_manifest(root)) == before
    (root / "runbooks" / "a.md").write_text(
        "# A\n\nSecond version text.", encoding="utf-8"
    )
    assert corpus_fingerprint(corpus_manifest(root)) != before


def test_the_fingerprint_does_not_depend_on_mapping_order() -> None:
    manifest = corpus_manifest(CORPUS_ROOT)
    reversed_manifest = dict(reversed(list(manifest.items())))

    assert corpus_fingerprint(reversed_manifest) == corpus_fingerprint(manifest)


def entry(chunk_id: str, **changes: Any) -> CorpusChunk:
    base = CorpusChunk(chunk_id, "doc-a", 0, "runbooks/a.md", "runbook", "1" * 64)
    return replace(base, **changes)


def test_identical_manifests_match() -> None:
    manifest = {"c1": entry("c1"), "c2": entry("c2", chunk_index=1)}

    assert compare_corpus(manifest, dict(manifest)).matches


def test_differences_are_reported_by_chunk_identity() -> None:
    expected = {
        "kept": entry("kept"),
        "gone": entry("gone", chunk_index=1),
        "edited": entry("edited", chunk_index=2),
        "moved": entry("moved", chunk_index=3),
        "retyped": entry("retyped", chunk_index=4),
    }
    actual = {
        "kept": entry("kept"),
        "edited": entry("edited", chunk_index=2, content_sha256="2" * 64),
        "moved": entry("moved", chunk_index=3, source_path="runbooks/b.md"),
        "retyped": entry("retyped", chunk_index=4, document_type="incident"),
        "stale": entry("stale", source_path="runbooks/deleted.md"),
    }

    diff = compare_corpus(expected, actual)

    assert not diff.matches
    assert [c.chunk_id for c in diff.missing] == ["gone"]
    assert [c.chunk_id for c in diff.unexpected] == ["stale"]
    assert [(e.chunk_id, a.chunk_id) for e, a in diff.changed] == [
        ("edited", "edited"),
        ("moved", "moved"),
        ("retyped", "retyped"),
    ]


def test_the_read_only_engine_sets_the_transaction_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        provenance, "create_engine", lambda url, **kwargs: calls.append((url, kwargs))
    )

    create_read_only_engine("postgresql+psycopg://u:p@localhost/db")

    assert calls == [
        (
            "postgresql+psycopg://u:p@localhost/db",
            {"connect_args": {"options": READ_ONLY_OPTIONS}},
        )
    ]
    assert READ_ONLY_OPTIONS == "-c default_transaction_read_only=on"


def test_package_versions_are_recorded() -> None:
    versions = package_versions()

    assert set(versions) == {
        "fastembed",
        "langchain-core",
        "langchain-anthropic",
        "anthropic",
    }
    assert versions["langchain-core"] == metadata.version("langchain-core")


# --- git ---------------------------------------------------------------------


def fake_git(outputs: dict[tuple[str, ...], tuple[int, str]]) -> Any:
    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args[0] == "git" and kwargs["cwd"] == REPO_ROOT
        assert "shell" not in kwargs
        code, stdout = outputs[tuple(args[1:])]
        return subprocess.CompletedProcess(args, code, stdout=stdout, stderr="")

    return run


GIT_OK = {
    ("rev-parse", "HEAD"): (0, "a" * 40 + "\n"),
    ("rev-parse", "HEAD:app"): (0, "b" * 40 + "\n"),
    ("rev-parse", "HEAD:data/knowledge"): (0, "c" * 40 + "\n"),
    ("status", "--porcelain"): (0, ""),
}


def test_a_clean_git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.subprocess, "run", fake_git(GIT_OK))

    identity = git_identity(REPO_ROOT)

    assert identity is not None
    assert identity.commit_sha == "a" * 40
    assert (identity.app_tree, identity.corpus_tree) == ("b" * 40, "c" * 40)
    assert identity.worktree_clean is True


def test_untracked_or_modified_files_make_the_worktree_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = GIT_OK | {("status", "--porcelain"): (0, "?? notes.txt\n")}
    monkeypatch.setattr(provenance.subprocess, "run", fake_git(outputs))

    identity = git_identity(REPO_ROOT)

    assert identity is not None and identity.worktree_clean is False


def test_a_failing_git_command_gives_no_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = GIT_OK | {("rev-parse", "HEAD:app"): (128, "")}
    monkeypatch.setattr(provenance.subprocess, "run", fake_git(outputs))

    assert git_identity(REPO_ROOT) is None


def test_missing_git_gives_no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", missing)

    assert git_identity(REPO_ROOT) is None


def test_the_real_repository_has_a_git_identity() -> None:
    identity = git_identity(REPO_ROOT)

    assert identity is not None
    assert len(identity.commit_sha) == 40
