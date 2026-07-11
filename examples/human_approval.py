"""Standalone ``interrupt()`` demo. [OPTIONAL FEATURE: delete this folder]

The main app pauses for user input by *ending the run* and re-entering
from START next turn (the gate pattern — see README). LangGraph's other
human-in-the-loop mechanism is ``interrupt()``: the graph pauses *inside*
a node, and a later ``Command(resume=...)`` continues from that exact
point. Prefer it when re-running earlier nodes would be wasteful or
wrong (approvals before irreversible actions, mid-task confirmation).

This example is self-contained — no LLM, no shared code with ``app/`` —
so it can be deleted (with its test in ``tests/test_examples.py``)
without touching anything else.

Run it::

    python examples/human_approval.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class ApprovalState(TypedDict, total=False):
    action: str
    approved: bool
    result: str


def request_approval(state: ApprovalState) -> dict:
    """Pause the run and wait for a human decision.

    ``interrupt()`` raises internally: the run stops here and the payload
    surfaces to the caller under ``"__interrupt__"``. When the caller
    resumes with ``Command(resume=value)``, this node re-runs and
    ``interrupt()`` *returns* that value.
    """

    decision = interrupt({"question": f"Approve '{state['action']}'? [yes/no]"})
    return {"approved": str(decision).strip().lower() == "yes"}


def act(state: ApprovalState) -> dict:
    outcome = "executed" if state["approved"] else "cancelled"
    return {"result": f"{state['action']}: {outcome}"}


def build_approval_graph(checkpointer=None):
    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("act", act)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "act")
    builder.add_edge("act", END)
    # interrupt() requires a checkpointer — the pause is a saved state.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def main() -> None:
    graph = build_approval_graph()
    config = {"configurable": {"thread_id": "approval-demo"}}

    result = graph.invoke({"action": "delete 3 old backups"}, config)
    question = result["__interrupt__"][0].value["question"]

    answer = input(f"{question} ")
    result = graph.invoke(Command(resume=answer), config)
    print(result["result"])


if __name__ == "__main__":
    main()
