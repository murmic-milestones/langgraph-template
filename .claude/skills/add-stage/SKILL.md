---
name: add-stage
description: Add a new onboarding stage to the graph (a fact to collect
  before the main chat, like the user's name). Use when asked to collect
  a new piece of information from the user, add an onboarding step, or
  insert a stage before chat.
---

Add an onboarding stage that collects `<fact>` before the chat stage.
`app/agents/greeter.py` is the reference implementation — mirror it.

1. **State**: add the field to `Profile` in `app/state.py`.
2. **Agent**: create `app/agents/<fact>.py` with a class extending
   `BaseAgent`:
   - a Pydantic schema (like `NameCheck`): the extracted value
     (`| None`) + a `reply` question for when it's missing;
   - an async node method: return `{}` if the fact is already in
     `profile` (idempotency — the graph re-runs every turn); otherwise
     `await self.query_structured(...)`, then return either the updated
     profile or a question message;
   - a sync gate method: `bool(profile.get("<fact>"))`.
3. **Wire it** in `app/graph.py`, inserted into the chain:
   `add_node("collect_<fact>", agent.node, retry_policy=_LLM_RETRY)`,
   then repoint the previous stage's gate `{True: "collect_<fact>"}` and
   add `add_conditional_edges("collect_<fact>", agent.gate,
   {True: "chat", False: END})`.
4. **Test**: copy the shape of `test_onboarding_then_chat` in
   `tests/test_graph.py`; queue the new schema on the fake
   (`fake.structured_results[YourSchema] = ...`).
5. **Verify**: `python main.py --graph` (wiring), `pytest` (the
   invariant tests enforce async nodes / sync gates automatically).
6. **Docs**: update the flow diagrams in `app/graph.py`, README, and
   CLAUDE.md if the flow shape changed.
