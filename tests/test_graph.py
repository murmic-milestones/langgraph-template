"""End-to-end graph tests with a fake LLM — no API key or network needed.

Shows the pattern for testing LangGraph apps: monkeypatch the LLM factory
where agents look it up (the ``fake`` fixture in ``conftest.py``), then
drive whole conversation turns through the compiled graph and assert on
state. Nodes are async, so calls go through ``ainvoke`` via ``run()``.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.greeter import NameCheck
from app.graph import build_graph
from app.visualization import to_mermaid
from fakes import config, onboard_paul, run


def test_onboarding_then_chat(fake) -> None:
    graph = build_graph(checkpointer=InMemorySaver())

    # Turns 1–2: ask for the name, then store it and reply via the chat node.
    state = run(onboard_paul(graph, fake))
    assert state["profile"]["name"] == "Paul"
    assert state["messages"][-1].content == "Hello Paul!"

    # Turn 3: onboarding is idempotent — no extraction call, straight to chat.
    fake.structured_results.clear()  # any greeter call would now fail
    fake.reply_text = "Nice weather, Paul."
    state = run(
        graph.ainvoke({"messages": [HumanMessage(content="how's things?")]}, config())
    )
    assert state["messages"][-1].content == "Nice weather, Paul."


def test_thread_isolation(fake) -> None:
    """Sessions are keyed by thread_id — one thread's profile must not leak."""

    graph = build_graph(checkpointer=InMemorySaver())
    run(onboard_paul(graph, fake, thread="thread-a"))

    # A fresh thread starts onboarding from scratch.
    fake.structured_results[NameCheck] = NameCheck(name=None, reply="Who are you?")
    state_b = run(
        graph.ainvoke({"messages": [HumanMessage(content="hello")]}, config("thread-b"))
    )
    assert not state_b.get("profile", {}).get("name")
    assert state_b["messages"][-1].content == "Who are you?"
    assert len(state_b["messages"]) == 2  # its own history, not thread-a's

    # And thread-a's checkpoint is untouched.
    state_a = graph.get_state(config("thread-a")).values
    assert state_a["profile"]["name"] == "Paul"


def test_profile_reaches_chat_prompt(fake) -> None:
    """The collected name must be formatted into the chat system prompt."""

    graph = build_graph(checkpointer=InMemorySaver())
    run(onboard_paul(graph, fake))

    system_message = fake.chat_calls[-1][0]
    assert isinstance(system_message, SystemMessage)
    assert "Paul" in system_message.content


def test_message_history_accumulates_in_order(fake) -> None:
    """add_messages appends across turns: no drops, dupes, or reordering."""

    state = run(onboard_paul(build_graph(checkpointer=InMemorySaver()), fake))

    assert [(type(m), m.content) for m in state["messages"]] == [
        (HumanMessage, "hi"),
        (AIMessage, "What's your first name?"),
        (HumanMessage, "I'm Paul"),
        (AIMessage, "Hello Paul!"),
    ]


def test_greeter_fallback_question(fake) -> None:
    """An empty structured reply falls back to the canned question."""

    graph = build_graph(checkpointer=InMemorySaver())
    fake.structured_results[NameCheck] = NameCheck(name=None, reply="")
    state = run(graph.ainvoke({"messages": [HumanMessage(content="hi")]}, config("t")))

    assert (
        state["messages"][-1].content == "Hi! Before we start, what's your first name?"
    )
    assert not state.get("profile", {}).get("name")


def test_tool_calling_loop(fake) -> None:
    """chat → tools → chat: requested tools run and results reach the model."""

    graph = build_graph(checkpointer=InMemorySaver())
    run(onboard_paul(graph, fake))

    fake.replies = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_current_time", "args": {}, "id": "call_1"}],
        ),
        AIMessage(content="It is 12:00 UTC."),
    ]
    state = run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="what time is it?")]}, config()
        )
    )

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["get_current_time"]
    assert state["messages"][-1].content == "It is 12:00 UTC."
    # The tool result was sent back to the model on the second chat call.
    assert any(isinstance(m, ToolMessage) for m in fake.chat_calls[-1])


def test_history_trimming_bounds_the_prompt(fake, monkeypatch) -> None:
    """Only recent messages reach the model; full history stays in state."""

    monkeypatch.setattr("app.agents.chat.MAX_HISTORY_MESSAGES", 4)
    graph = build_graph(checkpointer=InMemorySaver())
    run(onboard_paul(graph, fake))

    for i in range(5):
        run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=f"message {i}")]}, config()
            )
        )

    sent = fake.chat_calls[-1]
    assert isinstance(sent[0], SystemMessage)
    assert len(sent) - 1 <= 4  # system prompt + trimmed window

    full_history = graph.get_state(config()).values["messages"]
    assert len(full_history) > 4  # state keeps everything


def test_studio_entry_point_runs_statelessly(fake) -> None:
    """The module-level graph used by langgraph.json compiles and runs."""

    from app.graph import graph as studio_graph

    fake.structured_results[NameCheck] = NameCheck(name="Zoe", reply="")
    fake.reply_text = "Hi Zoe!"
    state = run(studio_graph.ainvoke({"messages": [HumanMessage(content="I'm Zoe")]}))

    assert state["profile"]["name"] == "Zoe"
    assert state["messages"][-1].content == "Hi Zoe!"


def test_graph_wiring_renders_all_nodes() -> None:
    """Mermaid export names every node — guards accidental rewiring."""

    mermaid_src = to_mermaid(build_graph())
    for node in ("collect_name", "chat", "tools"):
        assert node in mermaid_src
