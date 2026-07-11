"""Test for the standalone interrupt() example.

Delete this file together with ``examples/human_approval.py``.
"""

from __future__ import annotations

from langgraph.types import Command

from examples.human_approval import build_approval_graph


def test_interrupt_pauses_then_resumes() -> None:
    graph = build_approval_graph()
    cfg = {"configurable": {"thread_id": "approval-test"}}

    # The run pauses inside request_approval and surfaces the payload.
    result = graph.invoke({"action": "deploy v2"}, cfg)
    assert "Approve 'deploy v2'?" in result["__interrupt__"][0].value["question"]
    assert "result" not in result  # act() has not run yet

    # Resuming re-enters the node at the interrupt point.
    result = graph.invoke(Command(resume="yes"), cfg)
    assert result["approved"] is True
    assert result["result"] == "deploy v2: executed"


def test_interrupt_rejection() -> None:
    graph = build_approval_graph()
    cfg = {"configurable": {"thread_id": "rejection-test"}}

    graph.invoke({"action": "drop database"}, cfg)
    result = graph.invoke(Command(resume="no"), cfg)
    assert result["approved"] is False
    assert result["result"] == "drop database: cancelled"
