"""Tests for the checkpoint history / forking example.

Delete this file together with ``examples/time_travel.py``.
"""

from __future__ import annotations

from examples.time_travel import build_trip_graph

CONFIG = {"configurable": {"thread_id": "trip-test"}}


def _three_turn_graph():
    graph = build_trip_graph()
    for stop in ("Paris", "Berlin", "Prague"):
        graph.invoke({"stop": stop}, CONFIG)
    return graph


def test_history_records_every_step() -> None:
    graph = _three_turn_graph()

    snapshots = list(graph.get_state_history(CONFIG))
    itineraries = [tuple(s.values.get("itinerary", [])) for s in snapshots]

    assert graph.get_state(CONFIG).values["itinerary"] == ["Paris", "Berlin", "Prague"]
    # Earlier states are all still reachable, newest first.
    assert ("Paris", "Berlin") in itineraries
    assert ("Paris",) in itineraries


def test_forking_from_a_checkpoint_branches_the_thread() -> None:
    graph = _three_turn_graph()

    after_berlin = next(
        s
        for s in graph.get_state_history(CONFIG)
        if s.values.get("itinerary") == ["Paris", "Berlin"] and not s.next
    )
    forked = graph.invoke({"stop": "Rome"}, after_berlin.config)

    assert forked["itinerary"] == ["Paris", "Berlin", "Rome"]
    # Both timelines coexist — forking never overwrites the original.
    endings = {
        tuple(s.values["itinerary"])
        for s in graph.get_state_history(CONFIG)
        if len(s.values.get("itinerary", [])) == 3
    }
    assert endings == {
        ("Paris", "Berlin", "Prague"),
        ("Paris", "Berlin", "Rome"),
    }
