---
name: remove-feature
description: Cleanly remove one of the template's optional features
  (tool calling, history trimming, SQLite sessions, the interrupt demo,
  the Agent Engine adapter). Use when asked to strip, remove, or slim
  down a template feature.
---

Remove an optional feature completely — code, tests, deps, and docs.

The **README "Optional features — how to add or remove" table is the
authoritative list** of every removable feature and exactly what to
delete (it is kept complete). Read that table first rather than relying
on a copy here. In-tree features are also tagged in code comments and
document their own removal steps where they live:

- Tool calling `[tools]` → `tools/__init__.py` docstring
- History trimming `[trim]` → `app/agents/chat.py` docstring
- SQLite sessions `[sqlite]` → `main.py` docstring
- each `examples/<name>.py` demo → its own module docstring; delete it
  together with `tests/test_<name>.py` (and any extra it owns)

Procedure:

1. Look up the feature in the README table (and any `[tag]` in code);
   follow the documented steps exactly — they are kept current.
2. Delete the feature's test file(s) or the specific tests exercising
   it (e.g. `test_tool_calling_loop` for `[tools]`).
3. Remove its dependency/extra from `pyproject.toml` if no other
   feature uses it.
4. Sweep the docs: README structure tree + optional-features table +
   the pattern section describing it; CLAUDE.md's optional-features
   paragraph; `.env.example` if it had variables; this skill's table.
5. **Verify**: `pytest` must be green and `ruff check .` clean; run
   `python main.py --graph` to confirm the graph still wires up.
