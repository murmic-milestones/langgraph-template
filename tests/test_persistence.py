"""Durable-session test for the optional SQLite checkpointer. [sqlite]

Delete this file together with the ``--db`` feature (see ``main.py``).
Skips automatically if ``langgraph-checkpoint-sqlite`` is not installed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.greeter import NameCheck
from app.graph import build_graph
from fakes import config, run

pytest.importorskip("langgraph.checkpoint.sqlite.aio")
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402


def test_sqlite_sessions_survive_restart(fake, tmp_path) -> None:
    """Two separate saver instances on one file = a process restart."""

    db_path = str(tmp_path / "chat.db")

    async def first_process() -> None:
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            graph = build_graph(checkpointer=saver)
            fake.structured_result = NameCheck(name="Paul", reply="")
            fake.reply_text = "Hello Paul!"
            await graph.ainvoke(
                {"messages": [HumanMessage(content="I'm Paul")]}, config()
            )

    async def second_process() -> dict:
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            graph = build_graph(checkpointer=saver)
            fake.structured_result = None  # greeter must not run again
            fake.reply_text = "Welcome back, Paul!"
            return await graph.ainvoke(
                {"messages": [HumanMessage(content="hi again")]}, config()
            )

    run(first_process())
    state = run(second_process())

    assert state["profile"]["name"] == "Paul"
    assert state["messages"][-1].content == "Welcome back, Paul!"
    assert len(state["messages"]) == 4  # both turns' history survived
