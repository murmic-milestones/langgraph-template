"""Cross-thread memory with the Store API. [OPTIONAL FEATURE: delete
this file + tests/test_long_term_memory.py]

The checkpointer remembers everything *within* one thread — that is the
"checkpointer + thread id = sessions" pattern. But each thread starts
blank: nothing collected in one conversation is visible in the next. The
second persistence layer, a **store** (``BaseStore``), holds facts that
survive across threads, namespaced by user.

The division of labour to copy:

* checkpointer → conversation state, keyed by ``thread_id`` (a session);
* store → durable facts, keyed by a namespace you choose — here
  ``("memories", user_id)`` (a user's long-term memory).

Nodes receive the store by declaring a keyword-only ``store`` parameter;
the instance passed to ``compile(store=...)`` is injected at runtime.
Like the checkpointer, ``InMemoryStore`` is for demos — production wants
a durable backend (e.g. ``PostgresStore``) owned by the process driver.

SECURITY: the namespace is the only isolation between users' memories —
derive ``user_id`` from an authenticated identity, exactly like
``thread_id`` (see README "Security").

This example is self-contained (no LLM, no shared code with ``app/``).
Run it::

    python examples/long_term_memory.py
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class MemoryState(TypedDict, total=False):
    note: str  # a fact to remember this turn (optional input)
    recalled: list[str]  # every fact known about the user (output)


def _user_id(config: RunnableConfig) -> str:
    # Like thread_id: server-derived from the authenticated caller.
    return config["configurable"]["user_id"]


def remember(state: MemoryState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Node: persist this turn's fact under the user's namespace."""

    if state.get("note"):
        store.put(("memories", _user_id(config)), str(uuid4()), {"note": state["note"]})
    return {}


def recall(state: MemoryState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Node: read back everything stored for this user — any thread."""

    items = store.search(("memories", _user_id(config)))
    return {"recalled": [item.value["note"] for item in items]}


def build_memory_graph(store: BaseStore | None = None, checkpointer=None):
    builder = StateGraph(MemoryState)
    builder.add_node("remember", remember)
    builder.add_node("recall", recall)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "recall")
    builder.add_edge("recall", END)
    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )


def main() -> None:
    graph = build_memory_graph()

    def config(thread: str) -> dict:
        return {"configurable": {"thread_id": thread, "user_id": "alice"}}

    # Session 1: alice states a fact.
    graph.invoke({"note": "favourite colour is blue"}, config("monday-chat"))

    # Session 2: a brand-new thread — the checkpointer knows nothing
    # about it, but the store still holds alice's facts.
    state = graph.invoke({}, config("friday-chat"))
    print("recalled in a fresh thread:", state["recalled"])


if __name__ == "__main__":
    main()
