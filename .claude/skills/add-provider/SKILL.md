---
name: add-provider
description: Add support for a new model provider (e.g. Mistral, Groq,
  Bedrock). Use when asked to support a new LLM vendor or make a new
  MODEL_NAME prefix work.
---

Add a model provider. Providers are pure configuration — no agent or
graph code changes. Three files must stay in sync (an invariant test,
`test_providers_stay_in_sync_across_config_files`, fails until they do):

1. **`app/env.py`**: add a `Provider` row to `PROVIDERS` — import
   package name, exact install hint, API-key env var (or `None` if the
   provider authenticates another way; add a `preflight` callable if a
   local server or credential needs checking, like Ollama's).
2. **`pyproject.toml`**: add an extra named in the install hint, pinned
   to the integration package's current major
   (`mistral = ["langchain-mistralai>=X,<X+1"]` — check PyPI for X).
3. **`.env.example`**: add a row to the provider table and a
   `MODEL_NAME=` example. The provider prefix must be one
   `init_chat_model` understands.
4. **Test**: extend `tests/test_environment.py` only if the provider has
   novel behaviour (no key, custom preflight); the sync invariant test
   covers the rest.
5. **Docs**: add a row to the README's "Model providers" table.
6. **Verify**: `pytest`, then a live smoke test with a real key if
   available (structured output + tool calling must both work — note
   any caveats in the README row, as done for Ollama).
