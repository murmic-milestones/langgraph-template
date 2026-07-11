"""Shared LLM plumbing for graph agents.

Agents subclass :class:`BaseAgent` and expose two kinds of methods:

* **node methods** — async, take the state, return a partial state update
  (registered with ``builder.add_node``);
* **gate methods** — sync predicates over the state, return a routing
  value (registered with ``builder.add_conditional_edges``).

Node methods are ``async def`` so the graph can be served concurrently
(FastAPI, LangGraph platform) without a rewrite; call the model with
``ainvoke``/``astream`` accordingly.

Per-agent models: pass ``model_env="MY_AGENT_MODEL"`` to the constructor
and that env variable (any ``"provider:model"`` string) overrides the
shared ``MODEL_NAME`` for this agent only — different graph stages can
run different models as pure configuration. Remember to name the same
variable in the driver's ``check_environment(extra_model_vars=...)``
call so it is validated at startup.

Structured output is delegated to the model provider via
``with_structured_output``: the Pydantic schema is enforced server-side,
so no manual JSON parsing, jsonschema validation, or retry loop is needed.

Image input: :func:`image_message` builds the provider-agnostic content
blocks for one image + prompt; :meth:`BaseAgent.query_image_structured`
combines it with structured output. The exact block format below is the
part you cannot verify offline — it is exercised by the unit tests, but
run one real call against your provider before trusting a new one.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.llm import get_llm

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Refuse to embed images larger than this — provider request limits are
# lower still, and a runaway file would balloon memory and token costs.
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def image_message(text: str, image_path: str | Path) -> HumanMessage:
    """Build a human message carrying a prompt plus one base64 image block.

    Works with any vision-capable provider resolved by ``init_chat_model``
    (the ``{"type": "image", "source_type": "base64", ...}`` block is the
    LangChain standard form).

    Security: only call this with paths your application chose. Passing
    unvalidated user-supplied paths would let a user read arbitrary local
    files and send their contents to the model provider.
    """

    path = Path(image_path)
    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path} is {len(raw)} bytes; refusing to embed more than "
            f"{MAX_IMAGE_BYTES} (see MAX_IMAGE_BYTES)"
        )
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(raw).decode("ascii")
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image",
                "source_type": "base64",
                "data": data,
                "mime_type": mime_type,
            },
        ]
    )


class BaseAgent:
    """Base class providing LLM access helpers to agent subclasses.

    Args:
        temperature: sampling temperature for this agent's model.
        model_env: name of an env variable holding a per-agent model
            override; when unset or empty the shared ``MODEL_NAME``
            default applies.
    """

    def __init__(self, temperature: float = 0.3, model_env: str | None = None) -> None:
        self._temperature = temperature
        self._model_env = model_env

    @property
    def llm(self) -> BaseChatModel:
        model = os.getenv(self._model_env) if self._model_env else None
        return get_llm(self._temperature, model or None)

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

    async def query_image_structured(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | Path,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Run one image + prompt through the LLM, returning a validated model.

        The configured model must support vision input.
        """

        return await self.query_structured(
            system_prompt, [image_message(user_text, image_path)], schema
        )
