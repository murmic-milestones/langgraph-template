"""Tests for the cross-thread memory (Store API) example.

Delete this file together with ``examples/long_term_memory.py``.
"""

from __future__ import annotations

from examples.long_term_memory import build_memory_graph


def _config(thread: str, user: str) -> dict:
    return {"configurable": {"thread_id": thread, "user_id": user}}


def test_memory_survives_across_threads() -> None:
    graph = build_memory_graph()

    graph.invoke({"note": "favourite colour is blue"}, _config("t1", "alice"))
    # A brand-new thread: the checkpointer has no state for it, but the
    # store still holds alice's facts.
    state = graph.invoke({}, _config("t2", "alice"))

    assert state["recalled"] == ["favourite colour is blue"]


def test_memories_are_namespaced_per_user() -> None:
    graph = build_memory_graph()

    graph.invoke({"note": "favourite colour is blue"}, _config("t1", "alice"))
    state = graph.invoke({}, _config("t2", "bob"))

    assert state["recalled"] == []  # bob must never see alice's facts


def test_facts_accumulate_for_one_user() -> None:
    graph = build_memory_graph()

    graph.invoke({"note": "favourite colour is blue"}, _config("t1", "alice"))
    graph.invoke({"note": "allergic to peanuts"}, _config("t2", "alice"))
    state = graph.invoke({}, _config("t3", "alice"))

    assert sorted(state["recalled"]) == [
        "allergic to peanuts",
        "favourite colour is blue",
    ]
