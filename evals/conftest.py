"""Eval-run guards. [OPTIONAL FEATURE: evals — delete evals/ +
.github/workflows/evals.yml to remove]

Evals call REAL models: they need an API key, cost money, and are
stochastic. They are deliberately outside pytest's ``testpaths``, so
the default ``pytest`` (and the Stop hook, and CI's test matrix) never
runs them — invoke explicitly with ``pytest evals``. Without a key for
the configured provider, every eval skips with an explanation instead
of failing.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from app.env import PROVIDERS
from app.llm import DEFAULT_MODEL

load_dotenv()


def _missing_key_reason() -> str | None:
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    provider_name = model.split(":", 1)[0] if ":" in model else "openai"
    provider = PROVIDERS.get(provider_name)
    if provider and provider.key_var and not os.getenv(provider.key_var):
        return (
            f"{provider.key_var} is not set — evals call real models "
            f"(MODEL_NAME={model})"
        )
    return None


def pytest_report_header(config) -> str:
    return "evals: REAL model calls — these cost money and can flake"


def pytest_collection_modifyitems(config, items) -> None:
    reason = _missing_key_reason()
    if reason:
        marker = pytest.mark.skip(reason=reason)
        for item in items:
            item.add_marker(marker)
