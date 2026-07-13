"""Checkpoint history, replay, and forking. [OPTIONAL FEATURE: delete
this file + tests/test_time_travel.py]

Compiling with a checkpointer buys more than session resume: every step
of every run is kept as a checkpoint, and the thread's history is
queryable and re-enterable.

* ``graph.get_state_history(config)`` — every checkpoint of the thread,
  newest first: the state values, and which node was next to run.
* Invoking with a ``checkpoint_id`` in the config **forks** the thread
  from that point: the new run branches off the old state, and both
  timelines remain in the history. Nothing is overwritten.

This is the mechanism behind "rewind the conversation and try a
different answer" features and LangGraph Studio's step-through
debugging — and it comes free with the checkpointer the template
already uses.

This example is self-contained (no LLM, no shared code with ``app/``).
Run it::

    python examples/time_travel.py
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class TripState(TypedDict, total=False):
    stop: str  # this turn's input
    itinerary: Annotated[list[str], operator.add]  # accumulates across turns


def add_stop(state: TripState) -> dict:
    return {"itinerary": [state["stop"]]}


def build_trip_graph(checkpointer=None):
    builder = StateGraph(TripState)
    builder.add_node("add_stop", add_stop)
    builder.add_edge(START, "add_stop")
    builder.add_edge("add_stop", END)
    # The checkpointer is what makes history/forking possible.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def main() -> None:
    graph = build_trip_graph()
    config = {"configurable": {"thread_id": "trip"}}

    for stop in ("Paris", "Berlin", "Prague"):
        graph.invoke({"stop": stop}, config)
    print("itinerary:", graph.get_state(config).values["itinerary"])

    # Walk the thread's past: one snapshot per step, newest first.
    history = list(graph.get_state_history(config))
    print(f"{len(history)} checkpoints recorded")

    # Rewind to just after Berlin (a finished turn: nothing left to run)
    # and fork: same thread, different continuation.
    after_berlin = next(
        s
        for s in history
        if s.values.get("itinerary") == ["Paris", "Berlin"] and not s.next
    )
    forked = graph.invoke({"stop": "Rome"}, after_berlin.config)
    print("forked itinerary:", forked["itinerary"])

    # Both timelines coexist in the history — nothing was overwritten.
    endings = {
        tuple(s.values["itinerary"])
        for s in graph.get_state_history(config)
        if len(s.values.get("itinerary", [])) == 3
    }
    print("timelines:", sorted(endings))


if __name__ == "__main__":
    main()
