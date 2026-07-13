"""Map-reduce fan-out with the Send API. [OPTIONAL FEATURE: delete this
file + tests/test_parallel_fanout.py]

The main app is a straight line with one loop. When one request should
become N independent LLM calls that run *concurrently* — one per
document, per section, per sub-question — LangGraph's answer is
``Send``: a routing function returns a list of ``Send("node", payload)``
objects, the runtime executes one node instance per payload in parallel,
and a reducer (here ``operator.add``) merges their partial results back
into the shared state. A join node then runs once, after every branch
has finished.

The pattern to copy:

* fan-out is a *routing function* (``fan_out``) returning ``Send``s —
  the payload dict becomes that node instance's input state, and may
  carry keys that are not in the graph state (``aspect`` here);
* the collected key (``sections``) needs a reducer, or the parallel
  writes would collide instead of accumulating;
* the join node (``assemble``) needs no special wiring — a plain edge
  from the fanned-out node runs it once when all branches are done.

Run it (needs a configured model, see .env.example)::

    python -m examples.parallel_fanout "green tea"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import operator
import sys
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from app.agents.base import BaseAgent

_logger = logging.getLogger(__name__)

# One parallel branch per aspect. In a real project this list often
# comes from an earlier planning node instead of a constant.
ASPECTS = ("history", "how it works today", "outlook")

_SECTION_PROMPT = """\
You are writing one section of a short briefing about "{topic}".
Cover only this aspect: {aspect}. Two or three sentences, plain prose.
"""


class FanoutState(TypedDict, total=False):
    """Shared graph state."""

    topic: str
    # Reducer required: parallel branches each return {"sections": [one]}
    # and operator.add concatenates them instead of overwriting.
    sections: Annotated[list[str], operator.add]
    report: str


class SectionState(TypedDict):
    """Input carried by each Send — not part of the shared state."""

    topic: str
    aspect: str


class SectionWriter(BaseAgent):
    """Writes one section per branch."""

    async def write_section(self, state: SectionState) -> dict:
        prompt = _SECTION_PROMPT.format(topic=state["topic"], aspect=state["aspect"])
        reply = await self.llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["topic"])]
        )
        return {"sections": [f"## {state['aspect'].title()}\n{reply.text}"]}


def fan_out(state: FanoutState) -> list[Send]:
    """Routing function: one Send per aspect; LangGraph runs them concurrently."""

    return [
        Send("write_section", {"topic": state["topic"], "aspect": aspect})
        for aspect in ASPECTS
    ]


def assemble(state: FanoutState) -> dict:
    """Join node: runs once, after every parallel branch has returned."""

    body = "\n\n".join(state["sections"])
    return {"report": f"# {state['topic'].title()}\n\n{body}"}


def build_fanout_graph():
    writer = SectionWriter()
    builder = StateGraph(FanoutState)
    builder.add_node("write_section", writer.write_section)
    builder.add_node("assemble", assemble)
    # The list argument names the possible Send targets for compilation.
    builder.add_conditional_edges(START, fan_out, ["write_section"])
    builder.add_edge("write_section", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile()  # one-shot run: no checkpointer needed


async def amain(topic: str) -> None:
    graph = build_fanout_graph()
    state = await graph.ainvoke({"topic": topic})
    print(state["report"])


def main() -> None:
    from dotenv import load_dotenv

    from app.env import EnvironmentCheckError, check_environment
    from app.log import configure_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="what the briefing is about")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    try:
        check_environment()
    except EnvironmentCheckError as error:
        sys.exit(str(error))

    asyncio.run(amain(args.topic))


if __name__ == "__main__":
    main()
