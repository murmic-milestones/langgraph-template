"""Single factory for chat-model instances.

Centralising model construction means every agent picks up configuration
from the environment, and switching providers is a *config* change, not a
code change: ``init_chat_model`` resolves ``"provider:model"`` strings
(``openai:...``, ``anthropic:...``, ``google_genai:...``, ``ollama:...``)
as soon as the matching integration package is installed — extras for
each are defined in ``pyproject.toml``, and the supported-provider table
lives in ``app/env.py`` (``PROVIDERS``).

Agents that need a different model from the rest of the graph pass an
explicit ``model`` string (see ``BaseAgent``'s ``model_env``); everything
else falls back to the ``MODEL_NAME`` env variable.

Resolution is uniform: the environment is re-read on **every** call, and
instances are cached per *resolved* (model, temperature) pair — so both
``MODEL_NAME`` and per-agent override variables behave identically, and
repeated calls with the same configuration reuse one instance.
"""

from __future__ import annotations

import os
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "openai:gpt-4o-mini"


@cache
def _build_llm(model: str, temperature: float) -> BaseChatModel:
    return init_chat_model(model, temperature=temperature)


def get_llm(temperature: float = 0.3, model: str | None = None) -> BaseChatModel:
    """Return a (cached) chat model configured from the environment.

    Args:
        temperature: sampling temperature for the model.
        model: explicit ``"provider:model"`` override; ``None`` uses the
            ``MODEL_NAME`` env variable (falling back to
            ``DEFAULT_MODEL``).
    """

    return _build_llm(model or os.getenv("MODEL_NAME", DEFAULT_MODEL), temperature)
