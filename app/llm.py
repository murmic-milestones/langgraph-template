"""Single factory for chat-model instances.

Centralising model construction means every agent picks up configuration
from the environment, and switching providers is a *config* change, not a
code change: ``init_chat_model`` resolves ``"provider:model"`` strings
(``openai:...``, ``anthropic:...``, ``google_genai:...``, ``ollama:...``)
as soon as the matching integration package is installed — extras for
each are defined in ``pyproject.toml``, and the supported-provider table
lives in ``main.py`` (``_PROVIDERS``).

Note: instances are cached, so changing ``MODEL_NAME`` mid-process has no
effect — restart instead.
"""

from __future__ import annotations

import os
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "openai:gpt-4o-mini"


@cache
def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """Return a cached chat model configured from the environment."""

    return init_chat_model(
        os.getenv("MODEL_NAME", DEFAULT_MODEL),
        temperature=temperature,
    )
