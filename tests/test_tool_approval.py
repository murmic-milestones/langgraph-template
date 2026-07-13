"""Tests for the tool-approval example (interrupt-gated tool calls).

Delete this file together with ``examples/tool_approval.py``.

The fake LLM's ``replies`` queue scripts the model's tool requests, so
these drive the full chat → review → tools loop offline.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from examples.tool_approval import build_tool_approval_graph
from fakes import config, run

DANGEROUS_CALL = AIMessage(
    content="",
    tool_calls=[{"name": "delete_backup", "args": {"name": "q3-2025"}, "id": "c1"}],
)
SAFE_CALL = AIMessage(
    content="",
    tool_calls=[{"name": "list_backups", "args": {}, "id": "c2"}],
)


def test_dangerous_tool_pauses_for_approval(fake) -> None:
    graph = build_tool_approval_graph()
    fake.replies = [DANGEROUS_CALL]

    result = run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="delete q3-2025")]}, config("pause")
        )
    )

    payload = result["__interrupt__"][0].value
    assert payload["tool"] == "delete_backup"
    assert payload["args"] == {"name": "q3-2025"}
    # The tool must NOT have run yet: no ToolMessage in the history.
    assert not any(m.type == "tool" for m in result["messages"])


def test_approval_executes_the_tool(fake) -> None:
    graph = build_tool_approval_graph()
    fake.replies = [DANGEROUS_CALL, AIMessage(content="Done — backup deleted.")]

    run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="delete q3-2025")]}, config("approve")
        )
    )
    result = run(graph.ainvoke(Command(resume="yes"), config("approve")))

    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert tool_outputs == ["Backup 'q3-2025' deleted."]
    assert result["messages"][-1].text == "Done — backup deleted."


def test_denial_skips_the_tool_and_informs_the_model(fake) -> None:
    graph = build_tool_approval_graph()
    fake.replies = [DANGEROUS_CALL, AIMessage(content="Understood, leaving it.")]

    run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="delete q3-2025")]}, config("deny")
        )
    )
    result = run(graph.ainvoke(Command(resume="no"), config("deny")))

    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert tool_outputs == ["Tool call denied by the user. Do not retry."]
    assert "deleted" not in " ".join(tool_outputs)
    assert result["messages"][-1].text == "Understood, leaving it."


def test_safe_tools_run_without_interruption(fake) -> None:
    graph = build_tool_approval_graph()
    fake.replies = [SAFE_CALL, AIMessage(content="Here are your backups.")]

    result = run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="what backups exist?")]}, config("safe")
        )
    )

    assert "__interrupt__" not in result
    tool_outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert tool_outputs == ["q1-2026, q2-2026, q3-2025 (stale)"]
