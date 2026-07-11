"""Single factory for chat-model instances.

Centralising model construction means every agent picks up configuration
(model name, API key) from the environment, and swapping providers is a
one-file change.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a cached chat model configured from the environment."""

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
        temperature=temperature,
    )
