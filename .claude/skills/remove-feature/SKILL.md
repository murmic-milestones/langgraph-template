---
name: remove-feature
description: Cleanly remove one of the template's optional features
  (tool calling, history trimming, SQLite sessions, the interrupt demo,
  the Agent Engine adapter). Use when asked to strip, remove, or slim
  down a template feature.
---

Remove an optional feature completely — code, tests, deps, and docs.
Each feature is tagged in code comments and documents its own removal
steps where it lives:

| Feature | Tag | Removal steps live in |
|---|---|---|
| Tool calling | `[tools]` | `app/tools.py` docstring |
| History trimming | `[trim]` | `app/agents/chat.py` docstring |
| SQLite sessions | `[sqlite]` | `main.py` docstring |
| interrupt() demo | — | delete `examples/human_approval.py` + `tests/test_examples.py` |
| Agent Engine adapter | — | delete `examples/agent_engine_app.py` + `tests/test_agent_engine.py` + the `[vertexai]` extra |

Procedure:

1. Search the codebase for the feature's tag and follow the documented
   steps exactly — they are kept current.
2. Delete the feature's test file(s) or the specific tests exercising
   it (e.g. `test_tool_calling_loop` for `[tools]`).
3. Remove its dependency/extra from `pyproject.toml` if no other
   feature uses it.
4. Sweep the docs: README structure tree + optional-features table +
   the pattern section describing it; CLAUDE.md's optional-features
   paragraph; `.env.example` if it had variables; this skill's table.
5. **Verify**: `pytest` must be green and `ruff check .` clean; run
   `python main.py --graph` to confirm the graph still wires up.
