# Project: LangGraph Starter Template

Hello-world LangGraph 1.x agent (collect name → chat with tools) meant
to be copied as the skeleton for new projects. Keep it minimal: it
exists to demonstrate patterns, not accumulate features. README.md
documents each pattern — update it whenever the pattern it describes
changes.

## Commands

```
pip install -e ".[dev]"   # deps live in pyproject.toml (no requirements.txt)
python main.py            # async CLI chat loop (needs .env)
python main.py --db x.db  # durable sessions (SQLite)
python main.py --graph    # print graph as Mermaid source
pytest                    # fake-LLM tests, no API key needed
ruff check . && ruff format .   # CI enforces both
langgraph dev             # LangGraph Studio
```

## Architecture

- `app/graph.py` — graph wiring, RetryPolicy, two entry points (see
  gotcha below). Nodes/edges are registered here only.
- `app/state.py` — TypedDict state; `messages` uses the `add_messages`
  reducer (append), `profile` overwrites. Nodes return partial updates.
- `app/agents/` — one class per stage: **async** node methods return
  state updates, sync gate methods route conditional edges. Agents are
  shared across sessions — keep them stateless.
- `app/llm.py` — the only place a model is constructed
  (`init_chat_model`, provider comes from `MODEL_NAME` env). Supported
  providers (OpenAI, Anthropic, Gemini, Ollama) are defined in three
  places that must stay in sync: `_PROVIDERS` in `main.py`, the extras
  in `pyproject.toml`, and the table in `.env.example`/README.
- `app/tools.py` — tool list for the chat ⇄ tools loop.
- One graph run per chat turn; incomplete onboarding ends the run and
  the next invoke re-enters from START, so nodes must be idempotent.

## Optional features

Tool calling `[tools]`, prompt trimming `[trim]`, SQLite sessions
`[sqlite]`, and the `examples/` interrupt demo are deliberately
loosely coupled. Code blocks belonging to each are marked with the
bracketed tag, and removal steps live in the feature's home file
(`app/tools.py`, `app/agents/chat.py`, `main.py`, `examples/`). When
editing near a marked block, preserve the tag comments and keep the
feature removable; when asked to remove a feature, follow its
documented steps including deleting the matching tests.

## Watch out for

- Two graph entry points: module-level `graph` (no checkpointer — used
  by langgraph.json / Studio, which inject persistence) vs
  `build_graph(checkpointer=...)` for CLI/tests/servers. Don't add a
  default checkpointer back to `build_graph`.
- Nodes are async — call the graph with `ainvoke`/`astream`. Sync
  `invoke` raises. Tests wrap calls in `run()` from `tests/fakes.py`.
- `STREAMING_NODES` in `main.py` whitelists which nodes stream tokens.
  Any new LLM-calling node is silent until added there; never add nodes
  that emit structured-output JSON or raw tool traffic.
- Tests monkeypatch `app.agents.base.get_llm` — new agents must fetch
  the model via `BaseAgent.llm`, not construct one directly.
- `get_llm` is cached; changing `MODEL_NAME` mid-process won't take.
- Trimming affects only what is *sent* to the model
  (`MAX_HISTORY_MESSAGES` in `chat.py`); state keeps the full history.
- Keep dependency pins to major 1.x ranges (verified against langgraph
  1.2.9). CI runs ruff check + format + pytest on 3.10/3.12/3.14 —
  run all three locally before pushing.
- License is 0BSD — no attribution required; don't add license headers
  to source files.
