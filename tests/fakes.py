"""Test doubles and helpers shared across test modules."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.greeter import NameCheck


class FakeLLM:
    """Stands in for the chat model in plain, structured, and tool modes.

    Records the messages of every call so tests can assert on the prompts
    the agents actually built. Replies come from the ``replies`` queue
    when non-empty (for scripted multi-call turns like tool loops),
    otherwise ``reply_text`` is returned.
    """

    def __init__(self) -> None:
        self.structured_result: NameCheck | None = None
        self.reply_text = "Hello!"
        self.replies: list[AIMessage] = []
        self.chat_calls: list[list] = []
        self.structured_calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _FakeStructured(self)

    async def ainvoke(self, messages) -> AIMessage:
        self.chat_calls.append(list(messages))
        if self.replies:
            return self.replies.pop(0)
        return AIMessage(content=self.reply_text)


class _FakeStructured:
    def __init__(self, parent: FakeLLM) -> None:
        self._parent = parent

    async def ainvoke(self, messages):
        self._parent.structured_calls.append(list(messages))
        return self._parent.structured_result


def run(coro):
    """Drive an async graph call from a sync test."""

    return asyncio.run(coro)


def config(thread: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread}}


async def onboard_paul(graph, fake: FakeLLM, thread: str = "test-thread") -> dict:
    """Run the standard two onboarding turns; return the state after turn 2."""

    fake.structured_result = NameCheck(name=None, reply="What's your first name?")
    await graph.ainvoke({"messages": [HumanMessage(content="hi")]}, config(thread))

    fake.structured_result = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    return await graph.ainvoke(
        {"messages": [HumanMessage(content="I'm Paul")]}, config(thread)
    )
