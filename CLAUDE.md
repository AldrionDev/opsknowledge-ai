# OpsKnowledge AI

Enterprise RAG knowledge assistant for technical documentation, runbooks, incident reports, and operational knowledge.

## Core rules

- Use Python 3.12.
- Use `uv` for Python and dependency management; do not use `pip` directly.
- Keep `pyproject.toml` and `uv.lock` in sync.
- GitHub issues are the source of truth for implementation scope.
- Work only within the current issue.
- Do not implement future milestone functionality unless explicitly requested.
- Prefer simple, readable solutions over speculative abstractions.
- Do not add dependencies unless required by the current issue.
- Before implementation, inspect the issue and repository and propose a concise plan.
- Run all checks required by the issue before claiming completion.
- Do not commit directly to `main`.
- Never expose or commit secrets.
- Never add AI/Claude attribution, co-author entries, signatures, or generated-by notices to commits or pull requests.

## References

Read only when relevant:

- `docs/ai/project-context.md` — product purpose and scope
- `docs/ai/development-workflow.md` — development and Git workflow
