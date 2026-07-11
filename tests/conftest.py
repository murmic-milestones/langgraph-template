import pytest

from fakes import FakeLLM


@pytest.fixture
def fake(monkeypatch) -> FakeLLM:
    """Replace the LLM factory with a recording fake.

    Patches ``get_llm`` where agents look it up — new agents must fetch
    their model via ``BaseAgent.llm`` for this seam to keep working.
    """

    fake = FakeLLM()
    monkeypatch.setattr("app.agents.base.get_llm", lambda temperature=0.3: fake)
    return fake
