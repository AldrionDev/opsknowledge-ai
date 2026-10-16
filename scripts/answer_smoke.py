"""Manual smoke verification of the real structured-output chain.

Not part of pytest or CI: with `--live` it calls the Anthropic API. It checks that
the real chat model accepts `with_structured_output(AnswerDraft,
method="json_schema")` and returns a draft the grounding validation understands,
for one answerable and one unanswerable question over fixed synthetic sources.
Run it from the repository root, with the key exported in your own shell:

    export ANTHROPIC_API_KEY=...   # your own key; never read from .env
    uv run --frozen python -m scripts.answer_smoke --live

Without `--live` it prints what it would do, reads no key, builds no model and
makes no network call. No environment variable replaces the flag.

Environment preflight (after `--live`, before the key is read, any model is built
or any request is made): the run is refused, and nothing is done, if one of these
shell variables is set. Only their names are reported, never their values.

- endpoint redirection: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_URL`,
  `LANGSMITH_GATEWAY`, `LANGSMITH_GATEWAY_API_KEY`: the installed
  `langchain-anthropic` would send the key and the prompts to that endpoint;
- tracing: `LANGSMITH_TRACING`, `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`,
  `LANGCHAIN_TRACING`, unless unset, blank or an explicit false value (`false`,
  `0`, `no`, `off`): it would ship prompts and outputs to an external service.

Fixed bounds, not configurable and not influenced by `LLM_MODEL`,
`LLM_MAX_OUTPUT_TOKENS`, `LLM_MAX_RETRIES` or `LLM_TIMEOUT_SECONDS` (the script
reads `ANTHROPIC_API_KEY` only):

- model `claude-haiku-5-5` (`DEFAULT_MODEL`);
- at most 1024 output tokens per request;
- no transport retries (`max_retries=0`);
- at most two model invocations, enforced in code: a third is refused before the
  chain is called.

Output-token upper bound: 2 requests x 1024 = 2048 tokens. This is not a complete
monetary cost ceiling: input tokens (instructions, schema and the short
synthetic sources) and provider pricing are not included.

Fail fast: if the first invocation raises any error, the script reports a safe
diagnostic (the error type and its fixed message or code) and stops; it never
makes the second request, retries, or switches the structured-output method or
the model. A decision is then needed, as the issue demands: do not work around an
unsupported `json_schema`.

Failure diagnostics are fixed texts in four classes, so a local defect is never
mistaken for a provider verdict: provider or transport failure (fixed `LLMError`
message), invalid model output (code only, covers parsing and grounding failures),
local invocation limit, and unexpected local error (exception type name only, a
probable defect in this script, not evidence about the provider or the schema).
Raw exception messages, tracebacks, response bodies and source content are never
printed. The model's answer text is printed JSON-escaped, so control and ANSI
characters stay inert; `AnswerService` output itself is unchanged.

Exit status: 0 if both invocations completed; 1 if one failed; 2 if the run was
not started (`--live` missing, unsafe environment, key missing). A well-formed
answer that is not the expected outcome is reported as an observation, not as a
failure.

Assumptions and limitations:

- It verifies only that the real model accepts `json_schema` structured output for
  `AnswerDraft` and that its output passes parsing and grounding validation, on two
  fixed synthetic questions. It says nothing about answer quality or retrieval.
- The preflight knows the variable names read by the locked `langchain-anthropic`
  1.7.6 / `anthropic` stack; it is not a general network policy. HTTP(S) proxy
  variables are not blocked, and tracing switched on in code is not detected.
- Output tokens are bounded (2 x 1024); input tokens and price are not, so this is
  not a monetary cost ceiling. The first request with a new schema can be slower
  because the provider compiles a grammar (the request timeout is 60 seconds).
- A single run is indicative, and model output is not deterministic.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from app.answering.errors import InvalidModelOutputError
from app.answering.models import Answer, AnswerStatus
from app.answering.prompt import PROMPT_HASH, PROMPT_VERSION, build_answer_chain
from app.answering.service import AnswerService
from app.ingestion.models import DocumentType
from app.llm.config import API_KEY_ENV, DEFAULT_MODEL, LLMConfig
from app.llm.errors import LLMError
from app.llm.factory import create_chat_model
from app.retrieval.models import RetrievedChunk

SMOKE_MODEL = DEFAULT_MODEL
SMOKE_MAX_OUTPUT_TOKENS = 1024
SMOKE_MAX_RETRIES = 0
MAX_MODEL_INVOCATIONS = 2

PACKAGES = ("langchain-anthropic", "langchain-core", "anthropic")

# Variables that would send the request to another endpoint or ship prompts to an
# external tracing service. The names are those read by the locked stack.
ENDPOINT_OVERRIDE_VARS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_URL",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
)
TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)
_TRACING_OFF_VALUES = frozenset({"", "false", "0", "no", "off"})

SYNTHETIC_CHUNKS = (
    RetrievedChunk(
        chunk_id="smoke-1",
        document_id="smoke-runbook-crashloop",
        chunk_index=0,
        source_path="runbooks/smoke-crashloop.md",
        document_type=DocumentType.RUNBOOK,
        content=(
            "The payments-gateway pod enters CrashLoopBackOff when the config map "
            "is missing the DATABASE_HOST key. Add the key to the config map and "
            "restart the deployment to recover the pod."
        ),
        similarity_score=0.9,
    ),
    RetrievedChunk(
        chunk_id="smoke-2",
        document_id="smoke-runbook-disk",
        chunk_index=0,
        source_path="runbooks/smoke-disk.md",
        document_type=DocumentType.RUNBOOK,
        content=(
            "Node disk pressure evicts pods from the node. Free space by removing "
            "unused container images with the container runtime prune command."
        ),
        similarity_score=0.7,
    ),
)


@dataclass(frozen=True)
class SmokeQuestion:
    label: str
    text: str
    expected: AnswerStatus


QUESTIONS = (
    SmokeQuestion(
        "answerable",
        "Why does the payments-gateway pod enter CrashLoopBackOff, and how is it recovered?",
        AnswerStatus.ANSWERED,
    ),
    SmokeQuestion(
        "unanswerable",
        "What is the quarterly on-call rotation schedule of the database team?",
        AnswerStatus.INSUFFICIENT_CONTEXT,
    ),
)


class InvocationLimitError(Exception):
    """Raised instead of a model call once the invocation limit is reached."""


class FixedRetrieval:
    """Returns the fixed synthetic chunks for any question."""

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        return list(SYNTHETIC_CHUNKS[:limit])


class InvocationLimiter:
    """Wraps a chain and refuses a call beyond `limit`, before the chain is invoked."""

    def __init__(self, chain: Runnable[dict[str, Any], Any], limit: int) -> None:
        self._chain = chain
        self._limit = limit
        self.attempts = 0

    def __call__(self, inputs: dict[str, Any]) -> Any:
        if self.attempts >= self._limit:
            raise InvocationLimitError(
                f"refused: more than {self._limit} model invocations"
            )
        self.attempts += 1
        return self._chain.invoke(inputs)


def build_smoke_config(api_key: str) -> LLMConfig:
    """The pinned configuration; the process environment cannot change it.

    `LLMConfig(api_key=...)` takes the dataclass defaults and never reads
    `LLM_*`, so `load_llm_config()` is deliberately not used.
    """
    return replace(
        LLMConfig(api_key=api_key),
        model=SMOKE_MODEL,
        max_output_tokens=SMOKE_MAX_OUTPUT_TOKENS,
        max_retries=SMOKE_MAX_RETRIES,
    )


def find_unsafe_environment(environ: Mapping[str, str]) -> list[str]:
    """Names of the variables that make a live run unsafe; never their values.

    Fail closed: an endpoint variable counts if it is non-blank, a tracing variable
    unless it is blank or an explicit false value.
    """
    names = [name for name in ENDPOINT_OVERRIDE_VARS if environ.get(name, "").strip()]
    names += [
        name
        for name in TRACING_VARS
        if environ.get(name, "").strip().lower() not in _TRACING_OFF_VALUES
    ]
    return names


def read_api_key() -> str:
    return os.environ.get(API_KEY_ENV, "").strip()


def run_smoke(
    chain: Runnable[dict[str, Any], Any],
    emit: Callable[[str], None] = print,
) -> int:
    """Ask the fixed questions in order; stop at the first failure.

    Returns the exit status. Only safe diagnostics are emitted: never the raw
    exception text, the key or the model output of a failed invocation.
    """
    limiter = InvocationLimiter(chain, MAX_MODEL_INVOCATIONS)
    service = AnswerService(FixedRetrieval(), RunnableLambda(limiter))

    for question in QUESTIONS:
        started = time.perf_counter()
        try:
            answer = service.answer(question.text, limit=len(SYNTHETIC_CHUNKS))
        except InvocationLimitError:
            _emit_failure(
                emit, question, "invocation limit reached (local guard)", LIMIT_NOTE
            )
            return 1
        except InvalidModelOutputError as exc:
            _emit_failure(
                emit, question, f"invalid model output, code={exc.code}", DECISION_NOTE
            )
            return 1
        except LLMError as exc:
            # `LLMError` messages are fixed texts chosen by `app.llm.errors`.
            _emit_failure(emit, question, f"provider failure: {exc}", DECISION_NOTE)
            return 1
        except Exception as exc:
            # Fail fast on anything else without echoing its message, which could
            # carry detail. Only the type name is safe to show.
            _emit_failure(
                emit,
                question,
                f"unexpected local error ({type(exc).__name__})",
                DEFECT_NOTE,
            )
            return 1
        _emit_result(emit, question, answer, time.perf_counter() - started)

    emit(f"model invocations: {limiter.attempts} (limit {MAX_MODEL_INVOCATIONS})")
    return 0


DECISION_NOTE = (
    "stopped: no further model request was made; a decision is needed before "
    "anything is changed"
)
LIMIT_NOTE = (
    "stopped: the local invocation limit refused the request before any model call"
)
DEFECT_NOTE = (
    "stopped: no further model request was made; this points to a defect in this "
    "script or its inputs and is not evidence about the provider or the schema"
)


def _emit_failure(
    emit: Callable[[str], None],
    question: SmokeQuestion,
    diagnostic: str,
    note: str,
) -> None:
    emit(f"[FAIL] {question.label}: {diagnostic}")
    emit(note)


def _emit_result(
    emit: Callable[[str], None],
    question: SmokeQuestion,
    answer: Answer,
    seconds: float,
) -> None:
    verdict = "as expected" if answer.status is question.expected else "OBSERVATION"
    emit(
        f"[{question.label}] status={answer.status} reason={answer.reason} "
        f"citations={len(answer.citations)} ({verdict}, expected {question.expected}) "
        f"latency={seconds:.1f}s"
    )
    for citation in answer.citations:
        emit(f"  cites [{citation.source_id}] {citation.source_path}")
    if answer.answer is not None:
        # JSON string encoding keeps control and ANSI characters inert.
        emit(f"  answer: {json.dumps(answer.answer)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.answer_smoke",
        description="Manual live smoke test of the structured-output answer chain.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=f"call the Anthropic API ({MAX_MODEL_INVOCATIONS} requests at most)",
    )
    args = parser.parse_args(argv)

    if not args.live:
        print(
            f"Dry description: with --live this makes at most {MAX_MODEL_INVOCATIONS} "
            f"requests to {SMOKE_MODEL}, {SMOKE_MAX_OUTPUT_TOKENS} output tokens each, "
            f"no retries. Nothing was done; pass --live to run.",
            file=sys.stderr,
        )
        return 2

    unsafe = find_unsafe_environment(os.environ)
    if unsafe:
        print(
            "Refusing to run: these environment variables are set: "
            f"{', '.join(unsafe)} (values not shown). Unset them in your shell; "
            "nothing was done.",
            file=sys.stderr,
        )
        return 2

    api_key = read_api_key()
    if not api_key:
        print(
            f"{API_KEY_ENV} is not set or is blank; nothing was done.", file=sys.stderr
        )
        return 2

    config = build_smoke_config(api_key)
    print(
        f"model={config.model} max_output_tokens={config.max_output_tokens} "
        f"max_retries={config.max_retries} method=json_schema"
    )
    for package in PACKAGES:
        print(f"{package}=={metadata.version(package)}")
    print(f"prompt version={PROMPT_VERSION} hash={PROMPT_HASH}")

    chain = build_answer_chain(create_chat_model(config))
    return run_smoke(chain)


if __name__ == "__main__":
    raise SystemExit(main())
