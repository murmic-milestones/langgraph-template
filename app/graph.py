"""Graph assembly.

Flow (one run == one chat turn)::

    START → collect_name ──(name set?)──> chat ──(tool calls?)──> END
                     │                     ↑  └──> tools ──┘
                     └────── False ──────> END

Turns where onboarding is incomplete end early; the checkpointer persists
the state per ``thread_id``, so the next invoke resumes where the
conversation left off. Add stages by inserting a node + gate pair between
``collect_name`` and ``chat``.

The chat ⇄ tools cycle is the standard tool-calling loop: when the model
requests tool calls, ``ToolNode`` executes them and control returns to
``chat`` with the results; otherwise the turn ends. Lines marked
``[tools]`` below are removable — see ``app/tools.py`` for the steps.

Two entry points:

* :func:`build_graph` — call with a checkpointer when you own the runtime
  (CLI, Flask/FastAPI, tests).
* ``graph`` — a compiled instance *without* a checkpointer, referenced by
  ``langgraph.json``. LangGraph Studio / ``langgraph dev`` / the LangGraph
  platform inject their own persistence, and warn if you bring your own.

Because ``graph = build_graph()`` runs at import time, everything
``build_graph`` constructs must be side-effect free until first use: no
filesystem writes, no network calls, no reading env vars into baked-in
values you'd want tests to override. If a dependency needs those (an
output file, a client), make it lazy — initialise on first use, not in
``__init__``.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition  # [tools]
from langgraph.types import RetryPolicy

from app.agents.chat import ChatAgent
from app.agents.greeter import GreeterAgent
from app.state import AppState
from app.tools import TOOLS  # [tools]

# Retry transient LLM/API failures with exponential backoff before
# surfacing the error to the caller. Scope (langgraph's default_retry_on):
# retries connection errors, HTTP 5xx, and unknown exceptions; does NOT
# retry programming errors (ValueError/TypeError/...) — note that
# structured-output parse failures (OutputParserException) subclass
# ValueError and therefore surface immediately rather than retrying.
_LLM_RETRY = RetryPolicy(max_attempts=3)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the application graph.

    Args:
        checkpointer: Persistence backend, e.g. ``InMemorySaver`` for local
            dev or a SQLite/Postgres saver for production. Leave as ``None``
            when the runtime provides persistence (LangGraph Studio /
            platform) — but note that without one, state is NOT kept
            between invokes.
    """

    greeter = GreeterAgent()
    chat = ChatAgent()

    builder = StateGraph(AppState)

    # Node/gate contract (async nodes, sync gate predicates) is enforced
    # by tests/test_template_invariants.py.
    builder.add_node("collect_name", greeter.collect_name, retry_policy=_LLM_RETRY)
    builder.add_node("chat", chat.respond, retry_policy=_LLM_RETRY)
    builder.add_node("tools", ToolNode(TOOLS))  # [tools]

    builder.add_edge(START, "collect_name")
    builder.add_conditional_edges(
        "collect_name",
        greeter.is_name_set,
        {True: "chat", False: END},
    )
    # [tools] Route to the tool node when the model requested calls,
    # otherwise end the turn. To remove tool calling, replace these two
    # statements with: builder.add_edge("chat", END)
    builder.add_conditional_edges(
        "chat",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "chat")  # [tools]

    return builder.compile(checkpointer=checkpointer)


# Entry point for langgraph.json (Studio / `langgraph dev` / platform) —
# must stay checkpointer-free: the runtime injects its own persistence.
graph = build_graph()
