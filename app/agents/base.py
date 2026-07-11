"""Shared LLM plumbing for graph agents.

Agents subclass :class:`BaseAgent` and expose two kinds of methods:

* **node methods** — async, take the state, return a partial state update
  (registered with ``builder.add_node``);
* **gate methods** — sync predicates over the state, return a routing
  value (registered with ``builder.add_conditional_edges``).

Node methods are ``async def`` so the graph can be served concurrently
(FastAPI, LangGraph platform) without a rewrite; call the model with
``ainvoke``/``astream`` accordingly.

Structured output is delegated to the model provider via
``with_structured_output``: the Pydantic schema is enforced server-side,
so no manual JSON parsing, jsonschema validation, or retry loop is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel

from app.llm import get_llm

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseAgent:
    """Base class providing LLM access helpers to agent subclasses."""

    def __init__(self, temperature: float = 0.3) -> None:
        self._temperature = temperature

    @property
    def llm(self) -> BaseChatModel:
        return get_llm(self._temperature)

    async def query_structured(
        self,
        system_prompt: str,
        messages: Sequence[AnyMessage],
        schema: type[SchemaT],
    ) -> SchemaT:
        """Run the conversation through the LLM, returning a validated model."""

        structured_llm = self.llm.with_structured_output(schema)
        return await structured_llm.ainvoke(
            [SystemMessage(content=system_prompt), *messages]
        )
