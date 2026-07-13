"""Tests for the Send-API map-reduce example.

Delete this file together with ``examples/parallel_fanout.py``.
"""

from __future__ import annotations

from examples.parallel_fanout import ASPECTS, build_fanout_graph
from fakes import run


def test_one_llm_call_per_aspect(fake) -> None:
    fake.reply_text = "Draft text."
    graph = build_fanout_graph()

    run(graph.ainvoke({"topic": "green tea"}))

    assert len(fake.chat_calls) == len(ASPECTS)
    # Each branch got its own aspect-specific system prompt.
    prompts = [messages[0].content for messages in fake.chat_calls]
    for aspect in ASPECTS:
        assert any(aspect in prompt for prompt in prompts)


def test_reducer_collects_every_section(fake) -> None:
    fake.reply_text = "Draft text."
    graph = build_fanout_graph()

    state = run(graph.ainvoke({"topic": "green tea"}))

    assert len(state["sections"]) == len(ASPECTS)
    assert all("Draft text." in section for section in state["sections"])


def test_join_runs_once_with_all_branches_merged(fake) -> None:
    fake.reply_text = "Draft text."
    graph = build_fanout_graph()

    state = run(graph.ainvoke({"topic": "green tea"}))

    assert state["report"].startswith("# Green Tea")
    for aspect in ASPECTS:
        assert f"## {aspect.title()}" in state["report"]
