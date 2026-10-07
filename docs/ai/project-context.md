# Project Context

## Product

OpsKnowledge AI is an enterprise GenAI knowledge assistant for technical documentation, operational runbooks, incident reports, and operational knowledge.

The application will eventually use Retrieval-Augmented Generation (RAG) to provide grounded answers with source citations.

## Intended capabilities

Planned capabilities include:

- technical knowledge Q&A
- incident search
- incident summarization
- structured information extraction
- source-backed answers
- insufficient-context handling
- retrieval and answer evaluation
- LLM tracing and observability

These capabilities must be implemented incrementally through GitHub issues and milestones.

## Current development principle

Do not build the final architecture upfront.

Each milestone should introduce only the concepts required at that stage.

The initial milestones intentionally avoid AI frameworks so the underlying components remain understandable.

## Safety boundary

The initial product is read-only.

It may retrieve, analyze, summarize, and answer questions about knowledge.

It must not execute infrastructure changes, modify cloud resources, restart services, or perform autonomous operational actions.