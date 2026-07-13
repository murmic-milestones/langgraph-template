"""Human approval for dangerous tool calls. [OPTIONAL FEATURE: delete
this file + tests/test_tool_approval.py]

The main app's tool loop executes *every* call the model requests. That
is fine for read-only tools, but the security notes in ``app/tools.py``
say irreversible actions must be gated behind human approval — this
example is that gate. A review node sits between the model's request and
``ToolNode``: calls to tools named in ``APPROVAL_REQUIRED`` pause the run
with ``interrupt()``; the caller resumes with ``Command(resume=...)``.
Denied calls are answered with a ToolMessage ("denied by the user") so
the model can respond gracefully instead of executing.

``examples/human_approval.py`` shows the bare interrupt() mechanism with
no LLM; this example is the production wiring — the pattern to copy when
a real tool deletes, pays, or publishes.

Run it (needs a configured model, see .env.example)::

    python -m examples.tool_approval
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.agents.base import BaseAgent

_logger = logging.getLogger(__name__)


@tool
def delete_backup(name: str) -> str:
    """Delete the named backup. IRREVERSIBLE."""

    # Stand-in for a real destructive action — the point of the example
    # is that this line is only ever reached after explicit approval.
    _logger.info("tool executed: delete_backup")
    return f"Backup '{name}' deleted."


@tool
def list_backups() -> str:
    """List the backups that currently exist."""

    _logger.info("tool executed: list_backups")
    return "q1-2026, q2-2026, q3-2025 (stale)"


EXAMPLE_TOOLS = [list_backups, delete_backup]

# Tool names that pause the run for human approval before executing.
# Everything else runs straight through, exactly like the main app.
APPROVAL_REQUIRED = {"delete_backup"}

_SYSTEM_PROMPT = """\
You are a careful ops assistant managing backups.
Use the provided tools; never claim to have done something you didn't.
"""


class ApprovalState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


class OpsAgent(BaseAgent):
    """Chat agent bound to this example's tools."""

    async def respond(self, state: ApprovalState) -> dict:
        llm = self.llm.bind_tools(EXAMPLE_TOOLS)
        reply = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [reply]}


def review_tool_calls(state: ApprovalState) -> Command[Literal["tools", "chat"]]:
    """Gate node: pause for approval before any dangerous tool runs.

    On approval the run continues to the tool node unchanged. On denial,
    every requested call is answered with a "denied" ToolMessage (the
    model needs a response for each tool_call id) and control returns to
    the chat node so the model can acknowledge the refusal.
    """

    request = state["messages"][-1]
    for call in request.tool_calls:
        if call["name"] not in APPROVAL_REQUIRED:
            continue
        decision = interrupt(
            {
                "tool": call["name"],
                "args": call["args"],
                "question": f"Allow {call['name']}({call['args']})? [yes/no]",
            }
        )
        if str(decision).strip().lower() != "yes":
            _logger.info("tool call denied by user: %s", call["name"])
            denied = [
                ToolMessage(
                    content="Tool call denied by the user. Do not retry.",
                    tool_call_id=c["id"],
                    name=c["name"],
                )
                for c in request.tool_calls
            ]
            return Command(goto="chat", update={"messages": denied})
    return Command(goto="tools")


def build_tool_approval_graph(checkpointer=None):
    agent = OpsAgent()
    builder = StateGraph(ApprovalState)
    builder.add_node("chat", agent.respond)
    builder.add_node("review", review_tool_calls)
    builder.add_node("tools", ToolNode(EXAMPLE_TOOLS))
    builder.add_edge(START, "chat")
    # Same routing as the main graph, with "review" spliced in front of
    # the tool node; review itself routes via Command.
    builder.add_conditional_edges(
        "chat", tools_condition, {"tools": "review", "__end__": END}
    )
    builder.add_edge("tools", "chat")
    # interrupt() requires a checkpointer — the pause is a saved state.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


async def main() -> None:
    from dotenv import load_dotenv

    from app.env import EnvironmentCheckError, check_environment
    from app.log import configure_logging

    load_dotenv()
    configure_logging()
    try:
        check_environment()
    except EnvironmentCheckError as error:
        sys.exit(str(error))

    graph = build_tool_approval_graph()
    config = {"configurable": {"thread_id": "tool-approval-demo"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Please delete the stale q3-2025 backup.")]},
        config,
    )
    while "__interrupt__" in result:
        answer = input(f"{result['__interrupt__'][0].value['question']} ")
        result = await graph.ainvoke(Command(resume=answer), config)
    print(result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
