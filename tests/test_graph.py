"""End-to-end graph tests with a fake LLM — no API key or network needed.

Shows the pattern for testing LangGraph apps: monkeypatch the LLM factory
where agents look it up (``app.agents.base.get_llm``), then drive whole
conversation turns through the compiled graph and assert on state.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.greeter import NameCheck
from app.graph import build_graph
from app.visualization import to_mermaid


class FakeLLM:
    """Stands in for ChatOpenAI in both plain and structured-output modes.

    Records the messages of every call so tests can assert on the prompts
    the agents actually built.
    """

    def __init__(self) -> None:
        self.structured_result: NameCheck | None = None
        self.reply_text = "Hello!"
        self.chat_calls: list[list] = []
        self.structured_calls: list[list] = []

    def with_structured_output(self, schema):
        return _FakeStructured(self)

    def invoke(self, messages) -> AIMessage:
        self.chat_calls.append(list(messages))
        return AIMessage(content=self.reply_text)


class _FakeStructured:
    def __init__(self, parent: FakeLLM) -> None:
        self._parent = parent

    def invoke(self, messages):
        self._parent.structured_calls.append(list(messages))
        return self._parent.structured_result


@pytest.fixture
def fake(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr("app.agents.base.get_llm", lambda temperature=0.3: fake)
    return fake


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _onboard_paul(graph, fake: FakeLLM, thread: str = "test-thread") -> dict:
    """Run the standard two onboarding turns; return the state after turn 2."""

    fake.structured_result = NameCheck(name=None, reply="What's your first name?")
    graph.invoke({"messages": [HumanMessage(content="hi")]}, _config(thread))

    fake.structured_result = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    return graph.invoke(
        {"messages": [HumanMessage(content="I'm Paul")]}, _config(thread)
    )


def test_onboarding_then_chat(fake) -> None:
    graph = build_graph(checkpointer=InMemorySaver())

    # Turns 1–2: ask for the name, then store it and reply via the chat node.
    state = _onboard_paul(graph, fake)
    assert state["profile"]["name"] == "Paul"
    assert state["messages"][-1].content == "Hello Paul!"

    # Turn 3: onboarding is idempotent — no extraction call, straight to chat.
    fake.structured_result = None  # would break if the greeter ran again
    fake.reply_text = "Nice weather, Paul."
    state = graph.invoke(
        {"messages": [HumanMessage(content="how's things?")]}, _config("test-thread")
    )
    assert state["messages"][-1].content == "Nice weather, Paul."


def test_thread_isolation(fake) -> None:
    """Sessions are keyed by thread_id — one thread's profile must not leak."""

    graph = build_graph(checkpointer=InMemorySaver())
    _onboard_paul(graph, fake, thread="thread-a")

    # A fresh thread starts onboarding from scratch.
    fake.structured_result = NameCheck(name=None, reply="Who are you?")
    state_b = graph.invoke(
        {"messages": [HumanMessage(content="hello")]}, _config("thread-b")
    )
    assert not state_b.get("profile", {}).get("name")
    assert state_b["messages"][-1].content == "Who are you?"
    assert len(state_b["messages"]) == 2  # its own history, not thread-a's

    # And thread-a's checkpoint is untouched.
    state_a = graph.get_state(_config("thread-a")).values
    assert state_a["profile"]["name"] == "Paul"


def test_profile_reaches_chat_prompt(fake) -> None:
    """The collected name must be formatted into the chat system prompt."""

    graph = build_graph(checkpointer=InMemorySaver())
    _onboard_paul(graph, fake)

    system_message = fake.chat_calls[-1][0]
    assert isinstance(system_message, SystemMessage)
    assert "Paul" in system_message.content


def test_message_history_accumulates_in_order(fake) -> None:
    """add_messages appends across turns: no drops, dupes, or reordering."""

    state = _onboard_paul(build_graph(checkpointer=InMemorySaver()), fake)

    assert [(type(m), m.content) for m in state["messages"]] == [
        (HumanMessage, "hi"),
        (AIMessage, "What's your first name?"),
        (HumanMessage, "I'm Paul"),
        (AIMessage, "Hello Paul!"),
    ]


def test_greeter_fallback_question(fake) -> None:
    """An empty structured reply falls back to the canned question."""

    graph = build_graph(checkpointer=InMemorySaver())
    fake.structured_result = NameCheck(name=None, reply="")
    state = graph.invoke({"messages": [HumanMessage(content="hi")]}, _config("t"))

    assert (
        state["messages"][-1].content
        == "Hi! Before we start, what's your first name?"
    )
    assert not state.get("profile", {}).get("name")


def test_studio_entry_point_runs_statelessly(fake) -> None:
    """The module-level graph used by langgraph.json compiles and runs."""

    from app.graph import graph as studio_graph

    fake.structured_result = NameCheck(name="Zoe", reply="")
    fake.reply_text = "Hi Zoe!"
    state = studio_graph.invoke({"messages": [HumanMessage(content="I'm Zoe")]})

    assert state["profile"]["name"] == "Zoe"
    assert state["messages"][-1].content == "Hi Zoe!"


def test_graph_wiring_renders_all_nodes() -> None:
    """Mermaid export names every node — guards accidental rewiring."""

    mermaid_src = to_mermaid(build_graph())
    for node in ("collect_name", "chat"):
        assert node in mermaid_src
