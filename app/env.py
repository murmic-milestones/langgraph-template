"""Provider registry and startup environment checking.

This module is the single source of truth (in code) for which model
providers the project supports. ``main.py`` calls :func:`check_environment`
before the first graph run so a missing package, missing API key, or dead
local server fails fast with guidance instead of a mid-run traceback —
any other driver you write (batch job, FastAPI app, worker) should do the
same.

Each :class:`Provider` row carries an optional ``preflight`` callable for
checks beyond package + key (e.g. pinging a local server). The extras in
``pyproject.toml`` and the table in ``.env.example`` must list the same
providers — when you add a row here, update both.

``MODEL_NAME`` uses ``init_chat_model``'s ``"provider:model"`` form; a
bare model name is treated as OpenAI. Agents may name additional env
variables holding per-agent model overrides (see ``BaseAgent``); pass
those variable names in ``extra_model_vars`` so their models are checked
too.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.util import find_spec

from app.llm import DEFAULT_MODEL

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class Provider:
    """What a model provider needs before the first call can succeed."""

    package: str  # import name of the integration package
    install_hint: str  # pip command shown when the package is missing
    key_var: str | None = None  # API-key env var (None = no key needed)
    preflight: Callable[[], str | None] | None = None  # extra check -> error


def _ollama_preflight() -> str | None:
    """Verify the local Ollama server is reachable.

    The scheme is validated before opening the URL: ``urlopen`` would
    happily fetch ``file://`` and other schemes, and config values should
    never get that power.
    """

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    if urllib.parse.urlparse(base_url).scheme not in ("http", "https"):
        return (
            f"OLLAMA_BASE_URL must be an http(s) URL, got: {base_url}\n"
            f"Example: {DEFAULT_OLLAMA_URL}"
        )
    try:
        with urllib.request.urlopen(base_url + "/api/tags", timeout=3):
            return None
    except (urllib.error.URLError, OSError):
        return (
            f"Cannot reach the Ollama server at {base_url}.\n"
            "Start it with: ollama serve"
        )


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("langchain_openai", 'pip install -e "."', "OPENAI_API_KEY"),
    "anthropic": Provider(
        "langchain_anthropic", 'pip install -e ".[anthropic]"', "ANTHROPIC_API_KEY"
    ),
    "google_genai": Provider(
        "langchain_google_genai", 'pip install -e ".[google]"', "GOOGLE_API_KEY"
    ),
    # Vertex AI authenticates via Application Default Credentials (a
    # service account on GCP, `gcloud auth application-default login`
    # locally) — no API-key env variable to check.
    "google_vertexai": Provider(
        "langchain_google_vertexai", 'pip install -e ".[vertexai]"'
    ),
    # preflight is late-bound via lambda so tests can monkeypatch
    # _ollama_preflight on the module.
    "ollama": Provider(
        "langchain_ollama",
        'pip install -e ".[ollama]"',
        preflight=lambda: _ollama_preflight(),
    ),
}


def configured_models(extra_model_vars: Iterable[str] = ()) -> set[str]:
    """Every model string the process is configured to use."""

    models = {os.getenv("MODEL_NAME", DEFAULT_MODEL)}
    for var in extra_model_vars:
        if os.getenv(var):
            models.add(os.environ[var])
    return models


def check_environment(extra_model_vars: Iterable[str] = ()) -> None:
    """Fail fast with guidance instead of a mid-run traceback.

    Args:
        extra_model_vars: names of env variables holding per-agent model
            overrides (e.g. ``("SUMMARISER_MODEL",)``); their models are
            validated alongside ``MODEL_NAME``.
    """

    for model in sorted(configured_models(extra_model_vars)):
        provider_name = model.split(":", 1)[0] if ":" in model else "openai"

        provider = PROVIDERS.get(provider_name)
        if provider is None:
            continue  # Unknown provider — let init_chat_model report it.

        if find_spec(provider.package) is None:
            sys.exit(
                f"Model {model} needs the "
                f"{provider.package.replace('_', '-')} package.\n"
                f"Install it with: {provider.install_hint}"
            )
        if provider.key_var and not os.getenv(provider.key_var):
            sys.exit(
                f"Missing {provider.key_var} (required by model {model}).\n"
                "Copy .env.example to .env and fill in your key, "
                f"or export {provider.key_var} in your shell."
            )
        if provider.preflight and (error := provider.preflight()):
            sys.exit(error)
