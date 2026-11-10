# Project Context

## Product

OpsKnowledge AI is an enterprise GenAI knowledge assistant for technical documentation, operational runbooks, incident reports, and operational knowledge.

The application uses Retrieval-Augmented Generation (RAG) to provide grounded answers with source citations.

## Capabilities

Implemented:

- technical knowledge Q&A via `POST /ask`
- source-backed answers with grounding validation
- insufficient-context handling
- semantic retrieval (`POST /search`)
- retrieval and answer evaluation (offline harness and live baseline)

Planned for later milestones:

- incident summarization
- structured information extraction
- LLM tracing and observability

Capabilities are implemented incrementally through GitHub issues and milestones.

## Current development principle

Do not build the final architecture upfront.

Each milestone should introduce only the concepts required at that stage.

The initial milestones intentionally avoid AI frameworks so the underlying components remain understandable.

## Safety boundary

The initial product is read-only.

It may retrieve, analyze, summarize, and answer questions about knowledge.

It must not execute infrastructure changes, modify cloud resources, restart services, or perform autonomous operational actions.