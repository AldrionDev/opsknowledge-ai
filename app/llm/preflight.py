"""Environment preflight for commands that make real LLM provider calls.

A live command must refuse to run when the shell environment would send the API
key and the prompts somewhere else than the configured provider, or ship them to
an external tracing service. `find_unsafe_environment` reports such variables
by name only, never their values.

- endpoint redirection: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_URL`,
  `LANGSMITH_GATEWAY`, `LANGSMITH_GATEWAY_API_KEY`: the installed
  `langchain-anthropic` would send the key and the prompts to that endpoint;
- tracing: `LANGSMITH_TRACING`, `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`,
  `LANGCHAIN_TRACING`, unless unset, blank or an explicit false value (`false`,
  `0`, `no`, `off`): it would ship prompts and outputs to an external service.

The names are those read by the locked `langchain-anthropic` / `anthropic`
stack; this is not a general network policy. HTTP(S) proxy variables are not
blocked, and tracing switched on in code is not detected.
"""

from collections.abc import Mapping

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
