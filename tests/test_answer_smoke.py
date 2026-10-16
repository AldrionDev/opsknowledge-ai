import ast
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import ModelRateLimitError, OutputParserException
from langchain_core.runnables import RunnableLambda

from app.answering.models import AnswerDraft, Claim
from app.llm.config import DEFAULT_MODEL, LLMConfig
from app.llm.factory import create_chat_model
from scripts import answer_smoke

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "answer_smoke.py"

GUARDED_VARS = (*answer_smoke.ENDPOINT_OVERRIDE_VARS, *answer_smoke.TRACING_VARS)

KEY = "sk-ant-test-key-SENTINEL-9e41"
SECRET_VALUE = "https://SENTINEL-redirect-host-4d2f.example"
SENTINEL_PROVIDER = "SENTINEL-PROVIDER-TEXT-b7c0"

ANSWERED = AnswerDraft(
    insufficient=False,
    claims=[
        Claim(
            text="The pod fails without the DATABASE_HOST key",
            source_id=1,
            evidence_quote="enters CrashLoopBackOff when the config map is missing",
        )
    ],
)
DECLINED = AnswerDraft(insufficient=True, claims=[])


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no `LLM_*` settings and no way to reach the provider or the network."""
    for name in list(os.environ):
        if (
            name == "ANTHROPIC_API_KEY"
            or name.startswith("LLM_")
            or name in GUARDED_VARS
        ):
            monkeypatch.delenv(name)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the smoke script reached for a model or the network")

    monkeypatch.setattr(answer_smoke, "create_chat_model", forbidden)
    monkeypatch.setattr(answer_smoke, "build_answer_chain", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


class SequenceChain:
    """Returns or raises the scripted results in order; counts every call."""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls = 0
        self.runnable = RunnableLambda(self._invoke)

    def _invoke(self, inputs: dict[str, Any]) -> Any:
        self.calls += 1
        result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def run(chain: SequenceChain) -> tuple[int, list[str]]:
    lines: list[str] = []
    return answer_smoke.run_smoke(chain.runnable, emit=lines.append), lines


# --- execution guard -------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"ANTHROPIC_API_KEY": KEY},
        {"SMOKE_LIVE": "1", "LIVE": "1", "ANTHROPIC_API_KEY": KEY},
    ],
)
def test_without_the_live_flag_nothing_is_done(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: dict[str, str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def key_read() -> str:
        raise AssertionError("the key must not be read without --live")

    monkeypatch.setattr(answer_smoke, "read_api_key", key_read)

    assert answer_smoke.main([]) == 2

    captured = capsys.readouterr()
    assert "--live" in captured.err
    assert KEY not in captured.out + captured.err


@pytest.mark.parametrize("argv", [["--other"], ["--liv"], ["--live=1"], ["live"]])
def test_only_the_exact_live_flag_is_accepted(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        answer_smoke.main(argv)

    assert info.value.code == 2


@pytest.mark.parametrize("key", [None, "", "   "])
def test_live_without_a_key_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str | None,
) -> None:
    if key is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)

    assert answer_smoke.main(["--live"]) == 2

    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_live_builds_the_pinned_model_and_runs_both_questions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "99999")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    built: list[LLMConfig] = []
    chain = SequenceChain(ANSWERED, DECLINED)

    def fake_create(config: LLMConfig) -> object:
        built.append(config)
        return object()

    monkeypatch.setattr(answer_smoke, "create_chat_model", fake_create)
    monkeypatch.setattr(
        answer_smoke, "build_answer_chain", lambda model: chain.runnable
    )

    assert answer_smoke.main(["--live"]) == 0

    (config,) = built
    assert (config.model, config.max_output_tokens, config.max_retries) == (
        DEFAULT_MODEL,
        1024,
        0,
    )
    assert chain.calls == 2
    output = capsys.readouterr()
    assert KEY not in output.out + output.err
    assert "status=answered" in output.out
    assert "status=insufficient_context" in output.out


# --- pinned configuration --------------------------------------------------


def test_pinned_constants() -> None:
    assert answer_smoke.SMOKE_MODEL == DEFAULT_MODEL == "claude-haiku-5-5"
    assert answer_smoke.SMOKE_MAX_OUTPUT_TOKENS == 1024
    assert answer_smoke.SMOKE_MAX_RETRIES == 0
    assert answer_smoke.MAX_MODEL_INVOCATIONS == 2


def test_environment_cannot_change_model_tokens_or_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "99999")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "1")

    config = answer_smoke.build_smoke_config(KEY)

    assert config.model == DEFAULT_MODEL
    assert config.max_output_tokens == 1024
    assert config.max_retries == 0
    assert config.timeout_seconds == LLMConfig(api_key="x").timeout_seconds
    assert KEY not in repr(config)


def test_the_real_chat_model_is_built_with_the_pinned_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "99999")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")

    model = create_chat_model(answer_smoke.build_smoke_config("dummy-key"))

    assert model.model == DEFAULT_MODEL
    assert model.max_tokens == 1024
    assert model.max_retries == 0


# --- invocation ceiling ----------------------------------------------------


def test_the_third_invocation_is_refused_before_the_chain_is_called() -> None:
    chain = SequenceChain(DECLINED)
    limiter = answer_smoke.InvocationLimiter(chain.runnable, 2)

    limiter({})
    limiter({})
    with pytest.raises(answer_smoke.InvocationLimitError):
        limiter({})

    assert chain.calls == 2
    assert limiter.attempts == 2


def test_the_smoke_run_makes_exactly_two_invocations() -> None:
    chain = SequenceChain(ANSWERED, DECLINED)

    status, lines = run(chain)

    assert status == 0
    assert chain.calls == 2
    assert len(answer_smoke.QUESTIONS) == 2
    assert any("status=answered" in line and "as expected" in line for line in lines)
    assert any(
        "status=insufficient_context" in line and "as expected" in line
        for line in lines
    )


def test_an_unexpected_but_well_formed_outcome_is_an_observation_not_a_failure() -> (
    None
):
    chain = SequenceChain(DECLINED, ANSWERED)

    status, lines = run(chain)

    assert status == 0
    assert sum("OBSERVATION" in line for line in lines) == 2


def test_fixed_inputs_fit_the_context_budget() -> None:
    from app.answering.context import assemble_context

    context = assemble_context(list(answer_smoke.SYNTHETIC_CHUNKS))

    assert len(context.sources) == len(answer_smoke.SYNTHETIC_CHUNKS) == 2


# --- fail fast -------------------------------------------------------------

FAILURES: list[tuple[str, Callable[[], object]]] = [
    ("provider", lambda: ModelRateLimitError(SENTINEL_PROVIDER)),
    (
        "malformed",
        lambda: OutputParserException(
            f"bad {SENTINEL_PROVIDER}", llm_output=SENTINEL_PROVIDER
        ),
    ),
    (
        "unknown-citation",
        lambda: AnswerDraft(
            insufficient=False,
            claims=[Claim(text="x", source_id=9, evidence_quote=SENTINEL_PROVIDER * 2)],
        ),
    ),
    ("inconsistent", lambda: AnswerDraft(insufficient=True, claims=ANSWERED.claims)),
    ("unexpected", lambda: KeyError(SENTINEL_PROVIDER)),
]


@pytest.mark.parametrize(
    "make_failure", [f for _, f in FAILURES], ids=[n for n, _ in FAILURES]
)
def test_a_failure_of_the_first_invocation_stops_before_the_second(
    make_failure: Callable[[], object],
) -> None:
    chain = SequenceChain(make_failure(), DECLINED)

    status, lines = run(chain)

    assert status == 1
    assert chain.calls == 1
    text = "\n".join(lines)
    assert "[FAIL] answerable" in text
    assert "no further model request" in text
    assert SENTINEL_PROVIDER not in text
    assert "[unanswerable]" not in text


def test_a_failure_of_the_second_invocation_is_reported_and_ends_the_run() -> None:
    chain = SequenceChain(ANSWERED, ModelRateLimitError(SENTINEL_PROVIDER), ANSWERED)

    status, lines = run(chain)

    assert status == 1
    assert chain.calls == 2
    assert "[FAIL] unanswerable" in "\n".join(lines)


def test_failure_diagnostics_use_the_fixed_message_and_code_only() -> None:
    _, provider_lines = run(SequenceChain(ModelRateLimitError(SENTINEL_PROVIDER)))
    _, malformed_lines = run(
        SequenceChain(OutputParserException("raw", llm_output="raw"))
    )

    assert "LLM provider rate limit exceeded" in "\n".join(provider_lines)
    assert "code=malformed_output" in "\n".join(malformed_lines)


# --- module hygiene --------------------------------------------------------


def test_the_script_has_no_database_embedding_or_web_imports() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert not roots & {
        "fastapi",
        "sqlalchemy",
        "psycopg",
        "pgvector",
        "fastembed",
        "dotenv",
    }
    assert not any(
        name.startswith(("app.db", "app.embeddings", "app.api")) for name in imported
    )


def test_importing_the_script_does_not_read_the_key() -> None:
    code = (
        "import os, app.answering.service, app.llm.factory\n"
        "class Env(dict):\n"
        "    def get(self, key, default=None):\n"
        "        assert key != 'ANTHROPIC_API_KEY', 'key read at import'\n"
        "        return super().get(key, default)\n"
        "    def __getitem__(self, key):\n"
        "        assert key != 'ANTHROPIC_API_KEY', 'key read at import'\n"
        "        return super().__getitem__(key)\n"
        "os.environ = Env(os.environ)\n"
        "import scripts.answer_smoke\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


# --- invocation limiter wiring in the real run path -------------------------


def test_run_smoke_enforces_the_invocation_limit_in_its_real_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(answer_smoke, "MAX_MODEL_INVOCATIONS", 1)
    chain = SequenceChain(ANSWERED, DECLINED)

    status, lines = run(chain)

    assert status == 1  # the run fails instead of reporting success
    assert chain.calls == 1  # the second invocation never reached the model
    text = "\n".join(lines)
    assert "[answerable] status=answered" in text
    assert "[FAIL] unanswerable: invocation limit reached" in text
    assert "[unanswerable] status" not in text
    assert "defect" not in text and "decision is needed" not in text


# --- failure classification and safe diagnostics -----------------------------


class SecretCarryingError(Exception):
    def __str__(self) -> str:
        return SENTINEL_PROVIDER

    def __repr__(self) -> str:
        return f"SecretCarryingError({SENTINEL_PROVIDER!r})"


@pytest.mark.parametrize(
    "failure",
    [
        KeyError(SENTINEL_PROVIDER),
        RuntimeError(f"boom {SENTINEL_PROVIDER}"),
        TypeError(SENTINEL_PROVIDER),
        SecretCarryingError(),
        ValueError(f"sk-ant-{SENTINEL_PROVIDER}"),
    ],
    ids=["key-error", "runtime-error", "type-error", "custom-str-and-repr", "key-like"],
)
def test_unexpected_errors_are_local_defects_and_leak_nothing(
    failure: Exception,
) -> None:
    chain = SequenceChain(failure, DECLINED)

    status, lines = run(chain)

    text = "\n".join(lines)
    assert status == 1
    assert chain.calls == 1
    assert f"unexpected local error ({type(failure).__name__})" in text
    assert "not evidence about the provider or the schema" in text
    assert SENTINEL_PROVIDER not in text
    assert "decision is needed" not in text
    assert "Traceback" not in text


def test_each_failure_class_gets_its_own_wording() -> None:
    _, provider = run(SequenceChain(ModelRateLimitError(SENTINEL_PROVIDER)))
    _, parsing = run(SequenceChain(OutputParserException("x", llm_output="x")))
    _, grounding = run(
        SequenceChain(AnswerDraft(insufficient=True, claims=ANSWERED.claims))
    )
    _, defect = run(SequenceChain(TypeError("x")))

    assert "provider failure" in "\n".join(provider)
    assert "code=malformed_output" in "\n".join(parsing)
    assert "code=inconsistent_response" in "\n".join(grounding)
    assert "provider failure" not in "\n".join(defect)
    assert "invalid model output" not in "\n".join(defect)
    for lines in (provider, parsing, grounding):
        assert "defect" not in "\n".join(lines)


# --- answer text is printed inert ---------------------------------------------


def test_model_answer_text_is_printed_json_escaped() -> None:
    draft = AnswerDraft(
        insufficient=False,
        claims=[
            Claim(
                text="red \x1b[31mALERT\x1b[0m bell \x07 done",
                source_id=1,
                evidence_quote="enters CrashLoopBackOff when the config map is missing",
            )
        ],
    )

    status, lines = run(SequenceChain(draft, DECLINED))

    text = "\n".join(lines)
    assert status == 0
    assert "\x1b" not in text and "\x07" not in text
    assert "\\u001b[31mALERT" in text


# --- environment preflight ----------------------------------------------------


@pytest.mark.parametrize("name", answer_smoke.ENDPOINT_OVERRIDE_VARS)
def test_endpoint_override_blocks_the_run_before_any_key_or_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv(name, SECRET_VALUE)

    def key_read() -> str:
        raise AssertionError("the key must not be read in an unsafe environment")

    monkeypatch.setattr(answer_smoke, "read_api_key", key_read)

    assert (
        answer_smoke.main(["--live"]) == 2
    )  # create/build are forbidden by the fixture

    captured = capsys.readouterr()
    assert name in captured.err
    assert SECRET_VALUE not in captured.out + captured.err
    assert KEY not in captured.out + captured.err


@pytest.mark.parametrize("name", answer_smoke.TRACING_VARS)
@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "enabled"])
def test_active_tracing_blocks_the_run_before_any_key_or_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(answer_smoke, "read_api_key", lambda: pytest.fail("key read"))

    assert answer_smoke.main(["--live"]) == 2

    captured = capsys.readouterr()
    assert name in captured.err
    assert KEY not in captured.out + captured.err


def test_all_blocked_names_are_reported_together() -> None:
    environ = dict.fromkeys(GUARDED_VARS, "true")

    assert sorted(answer_smoke.find_unsafe_environment(environ)) == sorted(GUARDED_VARS)


@pytest.mark.parametrize("value", ["", "  ", "false", "False", "0", "no", "off"])
def test_tracing_explicitly_off_or_blank_is_allowed(value: str) -> None:
    environ = dict.fromkeys(answer_smoke.TRACING_VARS, value)

    assert answer_smoke.find_unsafe_environment(environ) == []


@pytest.mark.parametrize("name", answer_smoke.ENDPOINT_OVERRIDE_VARS)
def test_blank_endpoint_variable_is_allowed(name: str) -> None:
    assert answer_smoke.find_unsafe_environment({name: "  "}) == []


def test_clean_environment_stays_usable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("UNRELATED_VARIABLE", "anything")
    chain = SequenceChain(ANSWERED, DECLINED)
    monkeypatch.setattr(answer_smoke, "create_chat_model", lambda config: object())
    monkeypatch.setattr(
        answer_smoke, "build_answer_chain", lambda model: chain.runnable
    )

    assert answer_smoke.main(["--live"]) == 0

    assert chain.calls == 2
    assert KEY not in capsys.readouterr().out
