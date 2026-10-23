import pytest

from tests.fake_llm import block_network, clear_llm_environment


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every evaluation unit test runs without credentials and without network."""
    clear_llm_environment(monkeypatch)
    block_network(monkeypatch)
