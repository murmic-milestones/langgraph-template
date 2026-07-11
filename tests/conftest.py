import pytest

from fakes import FakeLLM


@pytest.fixture
def fake(monkeypatch) -> FakeLLM:
    """Replace the LLM factory with a recording fake.

    Patches ``get_llm`` where agents look it up — new agents must fetch
    their model via ``BaseAgent.llm`` for this seam to keep working. The
    ``*args, **kwargs`` signature tolerates future ``get_llm`` parameters
    (temperature, per-agent model overrides, ...).
    """

    fake = FakeLLM()
    monkeypatch.setattr("app.agents.base.get_llm", lambda *args, **kwargs: fake)
    return fake
