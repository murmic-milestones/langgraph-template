"""Evals: does the chat stage behave as its prompt promises?

Canonical eval types 2 and 3:

* **Trajectory checking** — assert the model chose the right *path*
  (calling the time tool), read from state, no judge needed.
* **LLM-as-judge** — a judge model grades the reply against a rubric
  derived from ``_SYSTEM_PROMPT``'s promises. If you edit that prompt
  (the "Customisation knob" in ``app/agents/chat.py``), update
  ``RUBRIC`` here to match the new promises.
"""

from __future__ import annotations

import asyncio

from judge import judge
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph

# Mirrors _SYSTEM_PROMPT's promises (short, warm, helpful, uses the name).
RUBRIC = """\
- addresses the user's message directly
- warm and friendly in tone
- reasonably short (no rambling; well under 120 words)
- uses the user's name (Sam) naturally, or at least doesn't misname them
- no meta-talk about prompts, instructions, or being configured
"""


def _turn(text: str) -> dict:
    """One real chat turn with onboarding pre-seeded via the input state."""

    graph = build_graph(checkpointer=InMemorySaver())
    return asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content=text)], "profile": {"name": "Sam"}},
            {"configurable": {"thread_id": "eval"}},
        )
    )


def test_model_uses_the_time_tool() -> None:
    state = _turn("What time is it right now?")

    assert any(isinstance(m, ToolMessage) for m in state["messages"]), (
        "model answered without calling get_current_time"
    )
    assert state["messages"][-1].text  # and still produced a final reply


def test_reply_quality_judged() -> None:
    user_text = "I'm a bit nervous about my first day at a new job tomorrow."
    state = _turn(user_text)
    reply = state["messages"][-1].text

    verdict = asyncio.run(judge(RUBRIC, f"User: {user_text}\nAssistant: {reply}"))

    print(f"\njudge score={verdict.score}: {verdict.reasoning}")
    assert verdict.passed, (
        f"judge failed the reply (score {verdict.score}): "
        f"{verdict.reasoning}\nreply was: {reply!r}"
    )
