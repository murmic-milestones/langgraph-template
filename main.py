"""Local CLI entry point.

Usage::

    python main.py                 # interactive chat (in-memory sessions)
    python main.py --db chat.db    # durable sessions in a SQLite file
    python main.py --graph         # print the graph as Mermaid source

Each iteration of the loop is one graph run on the same ``thread_id``;
the checkpointer carries the conversation state between runs. A web
front-end works the same way — derive the thread id from the user's
session and run the graph per incoming message.

Replies stream token-by-token via ``stream_mode="messages"``. Only nodes
listed in ``STREAMING_NODES`` are streamed: every LLM call in the graph
emits chunks, including the greeter's structured-output extraction (raw
JSON deltas) and tool results — neither is user-facing. For turns that
end in a non-streaming node (e.g. the onboarding question) the final
message is printed from the checkpointed state instead.

``--db`` is an optional feature: it needs the ``langgraph-checkpoint-
sqlite`` package (in the ``dev`` extra). To remove it, delete the two
``[sqlite]`` blocks below and the dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib.util import find_spec

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.llm import DEFAULT_MODEL
from app.visualization import to_mermaid

# Nodes whose LLM tokens are printed as they arrive.
STREAMING_NODES = {"chat"}

# Supported model providers: import package, install hint, API-key env var
# (None = no key needed). MODEL_NAME uses init_chat_model's
# "provider:model" form; a bare model name is treated as OpenAI.
# To support another provider, add a row here, an extra in pyproject.toml,
# and an example in .env.example.
_PROVIDERS = {
    "openai": ("langchain_openai", 'pip install -e "."', "OPENAI_API_KEY"),
    "anthropic": (
        "langchain_anthropic",
        'pip install -e ".[anthropic]"',
        "ANTHROPIC_API_KEY",
    ),
    "google_genai": (
        "langchain_google_genai",
        'pip install -e ".[google]"',
        "GOOGLE_API_KEY",
    ),
    "ollama": ("langchain_ollama", 'pip install -e ".[ollama]"', None),
}


def check_environment() -> None:
    """Fail fast with guidance instead of a mid-chat traceback."""

    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    provider = model.split(":", 1)[0] if ":" in model else "openai"

    entry = _PROVIDERS.get(provider)
    if entry is None:
        return  # Unknown provider — let init_chat_model report it.
    package, install_hint, key_var = entry

    if find_spec(package) is None:
        sys.exit(
            f"MODEL_NAME={model} needs the {package.replace('_', '-')} "
            f"package.\nInstall it with: {install_hint}"
        )
    if key_var and not os.getenv(key_var):
        sys.exit(
            f"Missing {key_var} (required by MODEL_NAME={model}).\n"
            "Copy .env.example to .env and fill in your key, "
            f"or export {key_var} in your shell."
        )


async def run_turn(graph, config: dict, text: str) -> None:
    """Run one graph turn, streaming the reply to stdout."""

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

        await run_turn(graph, config, user_input)


async def amain(db_path: str | None) -> None:
    if db_path:  # [sqlite] durable sessions
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            await chat_loop(build_graph(checkpointer=saver))
    else:
        await chat_loop(build_graph(checkpointer=InMemorySaver()))


def main() -> None:
    load_dotenv()

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

    check_environment()
    asyncio.run(amain(args.db))


if __name__ == "__main__":
    main()
