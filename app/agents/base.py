"""Shared LLM plumbing for graph agents.

Agents subclass :class:`BaseAgent` and expose two kinds of methods:

* **node methods** — take the state, return a partial state update
  (registered with ``builder.add_node``);
* **gate methods** — take the state, return a routing value
  (registered with ``builder.add_conditional_edges``).

Structured output is delegated to the model provider via
``with_structured_output``: the Pydantic schema is enforced server-side,
so no manual JSON parsing, jsonschema validation, or retry loop is needed.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel

from app.llm import get_llm

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseAgent:
    """Base class providing LLM access helpers to agent subclasses."""

    def __init__(self, temperature: float = 0.3) -> None:
        self._temperature = temperature

    @property
    def llm(self):
        return get_llm(self._temperature)

    def query_structured(
        self,
        system_prompt: str,
        messages: Sequence[AnyMessage],
        schema: type[SchemaT],
    ) -> SchemaT:
        """Run the conversation through the LLM, returning a validated model."""

        structured_llm = self.llm.with_structured_output(schema)
        return structured_llm.invoke(
            [SystemMessage(content=system_prompt), *messages]
        )
