"""Local CLI entry point.

Run ``python main.py`` for an interactive chat loop, or
``python main.py --graph`` to print the graph as Mermaid source.

Each iteration of the loop is one graph run on the same ``thread_id``;
the checkpointer carries the conversation state between runs. A web
front-end works the same way — derive the thread id from the user's
session and run the graph per incoming message.

Replies stream token-by-token via ``stream_mode="messages"``. Only nodes
listed in ``STREAMING_NODES`` are streamed: the greeter's LLM call is a
structured-output extraction, so its raw deltas are not user-facing —
for those turns the final message is printed from the checkpointed state
instead.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.visualization import to_mermaid

# Nodes whose LLM tokens are printed as they arrive.
STREAMING_NODES = {"chat"}


def run_turn(graph, config: dict, text: str) -> None:
    """Run one graph turn, streaming the reply to stdout."""

    print("\nAssistant: ", end="", flush=True)
    streamed = False

    for chunk, metadata in graph.stream(
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
        state = graph.get_state(config)
        print(state.values["messages"][-1].text, end="")

    print("\n")


def main() -> None:
    load_dotenv()
    graph = build_graph(checkpointer=InMemorySaver())

    if "--graph" in sys.argv:
        print(to_mermaid(graph))
        return

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

        run_turn(graph, config, user_input)


if __name__ == "__main__":
    main()
