"""Test doubles and helpers shared across test modules."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.agents.greeter import NameCheck


class FakeLLM:
    """Stands in for the chat model in plain, structured, and tool modes.

    Records the messages of every call so tests can assert on the prompts
    the agents actually built.

    * Plain/tool mode: replies come from the ``replies`` queue when
      non-empty (for scripted multi-call turns like tool loops), otherwise
      ``reply_text`` is returned.
    * Structured mode: set one result per schema class in
      ``structured_results`` (works with any number of agents/schemas). A
      structured call for a schema with **no** entry fails the test — so
      asserting "this node did not run" is just not queueing its schema.
    """

    def __init__(self) -> None:
        self.structured_results: dict[type[BaseModel], BaseModel] = {}
        self.reply_text = "Hello!"
        self.replies: list[AIMessage] = []
        self.chat_calls: list[list] = []
        self.structured_calls: dict[type[BaseModel], list[list]] = {}

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema: type[BaseModel]) -> _FakeStructured:
        return _FakeStructured(self, schema)

    async def ainvoke(self, messages) -> AIMessage:
        self.chat_calls.append(list(messages))
        if self.replies:
            return self.replies.pop(0)
        return AIMessage(content=self.reply_text)


class _FakeStructured:
    def __init__(self, parent: FakeLLM, schema: type[BaseModel]) -> None:
        self._parent = parent
        self._schema = schema

    async def ainvoke(self, messages) -> BaseModel:
        self._parent.structured_calls.setdefault(self._schema, []).append(
            list(messages)
        )
        result = self._parent.structured_results.get(self._schema)
        assert result is not None, (
            f"unexpected structured call for {self._schema.__name__}"
        )
        return result


def run(coro):
    """Drive an async graph call from a sync test."""

    return asyncio.run(coro)


def config(thread: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread}}


async def onboard_paul(graph, fake: FakeLLM, thread: str = "test-thread") -> dict:
    """Run the standard two onboarding turns; return the state after turn 2."""

    fake.structured_results[NameCheck] = NameCheck(
        name=None, reply="What's your first name?"
    )
    await graph.ainvoke({"messages": [HumanMessage(content="hi")]}, config(thread))

    fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    return await graph.ainvoke(
        {"messages": [HumanMessage(content="I'm Paul")]}, config(thread)
    )
