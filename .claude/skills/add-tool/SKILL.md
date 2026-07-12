---
name: add-tool
description: Give the chat agent a new tool (a Python function the model
  can call). Use when asked to add a capability, integration, lookup, or
  action the assistant should be able to perform.
---

Add a tool the model can call during chat. `get_current_time` in
`app/tools.py` is the reference — the whole feature is one file.

1. **Write the function** in `app/tools.py`, decorated with `@tool`:
   - the **docstring is the model's documentation** — state what it
     does, its parameters, and what it returns, precisely;
   - type-hint every parameter; return JSON-serialisable values;
   - **security**: the model chooses the arguments from a conversation a
     user can steer via prompt injection — treat every argument as
     attacker-controlled. Validate/whitelist paths, URLs, and ids inside
     the tool; keep it least-privilege (no secrets, filesystem, or
     internal network unless essential); gate irreversible actions
     behind human approval. See the SECURITY note in `app/tools.py`.
2. **Register it**: append to the `TOOLS` list. Nothing else needs
   wiring — the chat agent and the graph's `ToolNode` both read `TOOLS`.
3. **Test**: copy `test_tool_calling_loop` in `tests/test_graph.py` —
   queue a fake `AIMessage` with `tool_calls=[{"name": ..., "args":
   {...}, "id": "call_1"}]`, run one turn, assert the `ToolMessage`
   appeared and reached the model's next call.
4. **Verify**: `pytest`, then ask the running bot (`python main.py`) a
   question that should trigger the tool.
5. If the tool needs a secret, add it to `.env.example` (placeholder
   only, never a real value) and read it with `os.getenv` inside the
   function, not at import time.
