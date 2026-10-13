import os

import pytest

from app.llm import config


@pytest.fixture(autouse=True)
def clean_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep LLM tests independent of the developer environment."""
    for name in list(os.environ):
        if name == config.API_KEY_ENV or name.startswith("LLM_"):
            monkeypatch.delenv(name)
