"""Graph assembly.

Flow (one run == one chat turn)::

    START → collect_name ──(name set?)──> chat → END
                     │
                     └────── False ────────────> END

Turns where onboarding is incomplete end early; the checkpointer persists
the state per ``thread_id``, so the next invoke resumes where the
conversation left off. Add stages by inserting a node + gate pair between
``collect_name`` and ``chat``.

Two entry points:

* :func:`build_graph` — call with a checkpointer when you own the runtime
  (CLI, Flask/FastAPI, tests).
* ``graph`` — a compiled instance *without* a checkpointer, referenced by
  ``langgraph.json``. LangGraph Studio / ``langgraph dev`` / the LangGraph
  platform inject their own persistence, and warn if you bring your own.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.agents.chat import ChatAgent
from app.agents.greeter import GreeterAgent
from app.state import AppState

# Retry transient LLM/API failures (rate limits, timeouts) with exponential
# backoff before surfacing the error to the caller.
_LLM_RETRY = RetryPolicy(max_attempts=3)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the application graph.

    Args:
        checkpointer: Persistence backend, e.g. ``InMemorySaver`` for local
            dev or ``PostgresSaver`` for production. Leave as ``None`` when
            the runtime provides persistence (LangGraph Studio/platform) —
            but note that without one, state is NOT kept between invokes.
    """

    greeter = GreeterAgent()
    chat = ChatAgent()

    builder = StateGraph(AppState)

    builder.add_node("collect_name", greeter.collect_name, retry_policy=_LLM_RETRY)
    builder.add_node("chat", chat.respond, retry_policy=_LLM_RETRY)

    builder.add_edge(START, "collect_name")
    builder.add_conditional_edges(
        "collect_name",
        greeter.is_name_set,
        {True: "chat", False: END},
    )
    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)


# Entry point for langgraph.json (Studio / `langgraph dev` / platform).
graph = build_graph()
