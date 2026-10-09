# Embeddings

Application code depends on the `EmbeddingService` protocol in
`app/embeddings/service.py`. The only implementation is `FastEmbedService`
(`app/embeddings/fastembed_service.py`), which runs a model locally on CPU. It
needs no API key and no cloud account.

## Model

| | |
|---|---|
| Library | `fastembed` (ONNX Runtime), version pinned in `uv.lock` |
| Model | `BAAI/bge-small-en-v1.5` |
| Vector dimension | 384 (`MODEL_DIMENSION`) |
| Input limit | 512 tokens (longer text is truncated by the model) |
| Output | plain Python `list[float]`; the model returns L2-normalized vectors |
| Language | English |

The model name and dimension are constants in `fastembed_service.py` and are
covered by tests. The database column dimension must match; see
[database.md](database.md#embedding-dimension). No query instruction prefix is added; for this model version it
is optional, and FastEmbed does not add one.

## Model artifact, revision and licensing

FastEmbed does not load the original BAAI weights. It downloads a quantized ONNX
port from the Hugging Face Hub repository `Qdrant/bge-small-en-v1.5-onnx-Q`
(`model_optimized.onnx`, about 67 MB).

- Observed revision of that repository: `aa8f8b060edb00e03bfdd08813a2949946c8ba55`
  (last modified 2026-09-24, observed 2026-10-08).
- The revision is **informational, not enforced**. FastEmbed resolves the current
  head of the repository at download time; nothing in this project pins it.
- Because the ONNX port is quantized, vectors are not bit-identical to those of
  the original PyTorch model.
- **If the model artifact or its revision changes, previously stored document
  embeddings must be regenerated before queries are run against them.** Vectors
  from different revisions are not guaranteed to be comparable.

Licensing, by source:

- Original model `BAAI/bge-small-en-v1.5`: MIT, as declared on its Hugging Face
  model card.
- Converted artifact `Qdrant/bge-small-en-v1.5-onnx-Q`: MIT, as declared on that
  repository's model card.
- `fastembed` library: Apache-2.0 (PyPI metadata).

## First run and offline use

The first use downloads the model and needs internet access. By default FastEmbed
caches it in `FASTEMBED_CACHE_PATH` or, if unset, in a directory in the system
temp dir, which may be cleared on reboot. For a persistent cache:

```bash
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
```

`FastEmbedService(cache_dir=...)` overrides the location in code. After the model
is cached, set `HF_HUB_OFFLINE=1` to run without network access.

The model is loaded once in the `FastEmbedService` constructor and reused by all
calls on that instance. Creating another instance loads it again.

## Input and output contract

- `embed_documents([])` returns `[]` without running the model.
- Empty or whitespace-only texts raise `ValueError` naming the index
  (`texts[2]`); the whole batch is rejected before the model runs. Valid texts
  are passed to the model unchanged.
- `embed_query` raises `ValueError` for an empty or whitespace-only query.
- The result has one vector per text, in input order; every vector, including
  query vectors, has `dimension` finite elements.
- Normalization is a property of the model: `FastEmbedService` does not verify it,
  the manual smoke script does.
- Model load or inference failures, a wrong vector count (a query must yield
  exactly one vector), a wrong dimension, or non-finite values raise
  `EmbeddingError`.

## Tests and smoke verification

Unit tests (`uv run pytest`) use a deterministic fake service and a stubbed
FastEmbed model. They never download a model or use the network.

The real model is verified manually, outside pytest and CI:

```bash
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
uv run python -m scripts.embedding_smoke                    # first run: downloads the model
HF_HUB_OFFLINE=1 uv run python -m scripts.embedding_smoke   # cached, offline
```

The script checks vector counts, dimension, finiteness, normalization, input
order and model reuse, and exits non-zero on failure.
