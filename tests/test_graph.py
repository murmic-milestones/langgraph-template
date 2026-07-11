"""End-to-end graph test with a fake LLM — no API key or network needed.

Shows the pattern for testing LangGraph apps: monkeypatch the LLM factory
where agents look it up (``app.agents.base.get_llm``), then drive whole
conversation turns through the compiled graph and assert on state.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.greeter import NameCheck
from app.graph import build_graph


class FakeLLM:
    """Stands in for ChatOpenAI in both plain and structured-output modes."""

    def __init__(self) -> None:
        self.structured_result: NameCheck | None = None
        self.reply_text = "Hello!"

    def with_structured_output(self, schema):
        return _FakeStructured(self)

    def invoke(self, messages) -> AIMessage:
        return AIMessage(content=self.reply_text)


class _FakeStructured:
    def __init__(self, parent: FakeLLM) -> None:
        self._parent = parent

    def invoke(self, messages):
        return self._parent.structured_result


def test_onboarding_then_chat(monkeypatch) -> None:
    fake = FakeLLM()
    monkeypatch.setattr("app.agents.base.get_llm", lambda temperature=0.3: fake)

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    # Turn 1: no name in the conversation — the greeter asks and the run ends.
    fake.structured_result = NameCheck(name=None, reply="What's your first name?")
    state = graph.invoke({"messages": [HumanMessage(content="hi")]}, config)

    assert state["messages"][-1].content == "What's your first name?"
    assert not state.get("profile", {}).get("name")

    # Turn 2: the user answers — name is stored and the chat node replies.
    fake.structured_result = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    state = graph.invoke({"messages": [HumanMessage(content="I'm Paul")]}, config)

    assert state["profile"]["name"] == "Paul"
    assert state["messages"][-1].content == "Hello Paul!"

    # Turn 3: onboarding is idempotent — no extraction call, straight to chat.
    fake.structured_result = None  # would break if the greeter ran again
    fake.reply_text = "Nice weather, Paul."
    state = graph.invoke({"messages": [HumanMessage(content="how's things?")]}, config)

    assert state["messages"][-1].content == "Nice weather, Paul."
