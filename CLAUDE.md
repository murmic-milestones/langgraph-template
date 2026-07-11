# Project: LangGraph Starter Template

Hello-world LangGraph 1.x chatbot (collect name → chat) meant to be
copied as the skeleton for new projects. Keep it minimal: it exists to
demonstrate patterns, not accumulate features. README.md documents each
pattern — update it whenever the pattern it describes changes.

## Commands

```
python main.py            # CLI chat loop (streams tokens; needs .env)
python main.py --graph    # print graph as Mermaid source
pytest                    # fake-LLM test, no API key needed
langgraph dev             # LangGraph Studio (needs requirements-dev.txt)
```

## Architecture

- `app/graph.py` — graph wiring, RetryPolicy, two entry points (see
  gotcha below). Nodes/edges are registered here only.
- `app/state.py` — TypedDict state; `messages` uses the `add_messages`
  reducer (append), `profile` overwrites. Nodes return partial updates.
- `app/agents/` — one class per stage: node methods return state
  updates, gate methods are pure predicates for conditional edges.
  Agents are shared across sessions — keep them stateless.
- `app/llm.py` — the only place a model is constructed.
- One graph run per chat turn; incomplete onboarding ends the run and
  the next invoke re-enters from START, so nodes must be idempotent.

## Watch out for

- Two graph entry points: module-level `graph` (no checkpointer — used
  by langgraph.json / Studio, which inject persistence) vs
  `build_graph(checkpointer=...)` for CLI/tests/servers. Don't add a
  default checkpointer back to `build_graph` — Studio warns on it.
- `STREAMING_NODES` in `main.py` whitelists which nodes stream tokens.
  Any new LLM-calling node is silent until added there; never add nodes
  that emit structured-output JSON (their raw deltas would leak).
- Tests monkeypatch `app.agents.base.get_llm` — new agents must fetch
  the model via `BaseAgent.llm`, not import ChatOpenAI directly.
- `get_llm` is lru_cached; changing env vars mid-process won't take.
- Keep requirements pinned to major 1.x ranges; this template is
  verified against langgraph 1.2.9.
- License is 0BSD — no attribution required; don't add license headers
  to source files.
