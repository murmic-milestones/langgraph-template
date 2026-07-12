"""Local CLI entry point.

Usage::

    python main.py                 # interactive chat (in-memory sessions)
    python main.py --db chat.db    # durable sessions in a SQLite file
    python main.py --graph         # print the graph as Mermaid source

Each iteration of the loop is one graph run on the same ``thread_id``;
the checkpointer carries the conversation state between runs. A web
front-end works the same way — derive the thread id from the user's
session and run the graph per incoming message.

Functions are ordered inner → outer: one turn (``run_turn``), the chat
loop (``chat_loop``), the runtime wrapper (``amain``), then the entry
point (``main``) — read top to bottom; execution starts at the bottom.
Streaming behaviour is documented on ``run_turn``; the optional
``[sqlite]`` feature on ``amain``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.env import EnvironmentCheckError, check_environment
from app.graph import build_graph
from app.log import configure_logging
from app.visualization import to_mermaid

_logger = logging.getLogger(__name__)

# Nodes whose LLM tokens are printed as they arrive. A new LLM-calling
# node stays silent until added here; never add nodes that emit
# structured-output JSON or raw tool traffic.
STREAMING_NODES = {"chat"}


async def run_turn(graph, config: dict, text: str) -> None:
    """Run one graph turn, streaming the reply to stdout.

    ``stream_mode="messages"`` yields chunks from *every* LLM call in
    the graph — including the greeter's structured-output extraction
    (raw JSON deltas) and tool traffic, which are not user-facing — so
    ``STREAMING_NODES`` whitelists what reaches the console. Turns that
    end in a non-streaming node print the final message from the
    checkpointed state instead — which assumes every turn ends with a
    user-facing message. A new node that can end a turn on a tool or
    structured-output message would print that raw; give such turns a
    user-facing closing message instead.
    """

    start = time.perf_counter()
    print("\nAssistant: ", end="", flush=True)
    streamed = False

    async for chunk, metadata in graph.astream(
        {"messages": [HumanMessage(content=text)]},
        config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") in STREAMING_NODES and chunk.text:
            print(chunk.text, end="", flush=True)
            streamed = True

    if not streamed:
        # Turn ended in a non-streaming node (e.g. the onboarding
        # question) — read the reply back from the checkpointed state.
        state = await graph.aget_state(config)
        print(state.values["messages"][-1].text, end="")

    print("\n")
    _logger.info(
        "turn complete: duration_ms=%.0f streamed=%s",
        (time.perf_counter() - start) * 1000,
        streamed,
        extra={"thread_id": config["configurable"]["thread_id"]},
    )


async def chat_loop(graph) -> None:
    config = {"configurable": {"thread_id": "local-cli"}}
    print("Hello-world LangGraph chat — type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        try:
            await run_turn(graph, config, user_input)
        except Exception:
            # Log the traceback, tell the user, keep the session alive.
            _logger.exception("turn failed")
            print("\n[error] That turn failed — details in the log. Try again.\n")


async def amain(db_path: str | None) -> None:
    """Build the graph with the chosen checkpointer and run the loop.

    ``--db`` is the optional ``[sqlite]`` feature (needs
    ``langgraph-checkpoint-sqlite``, in the dev extra). To remove it:
    delete the ``[sqlite]``-marked blocks in this file,
    ``tests/test_persistence.py``, and the dependency.
    """

    if db_path:  # [sqlite] durable sessions
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            await chat_loop(build_graph(checkpointer=saver))
    else:
        await chat_loop(build_graph(checkpointer=InMemorySaver()))


def main() -> None:
    load_dotenv()
    configure_logging()  # drivers configure; app/ modules only emit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", action="store_true", help="print Mermaid source and exit"
    )
    parser.add_argument(  # [sqlite]
        "--db", metavar="PATH", help="persist sessions to a SQLite file"
    )
    args = parser.parse_args()

    if args.graph:
        print(to_mermaid(build_graph()))
        return

    # check_environment raises so each driver picks its reaction; for the
    # CLI that is a clean exit with the fix-it message, no traceback.
    try:
        check_environment()
    except EnvironmentCheckError as error:
        sys.exit(str(error))
    asyncio.run(amain(args.db))


if __name__ == "__main__":
    main()
