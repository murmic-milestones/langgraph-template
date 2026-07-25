"""Main conversation agent — the hello-world payload of the template.

Replace or extend this agent with your project's real behaviour. It shows
the simplest node shape: build a system prompt from collected state, ask
the model, return the reply. Timing, logging, tool binding, and prompt
trimming all live in ``BaseAgent.query_chat`` (``lib/agent.py``) so the
node body stays pure logic.

Two optional features are single arguments below, each removable
independently:

* ``[tools]`` — ``tools=TOOLS`` binds the tools in the ``tools/`` package
  so the model can request calls; the graph's tool node executes them.
  Removal steps are documented in ``tools/__init__.py``.
* ``[trim]`` — ``max_messages=`` bounds the prompt window (full history
  stays in state — trimming affects only what is *sent*). To remove:
  delete that argument.
"""

from __future__ import annotations

from app.state import AppState
from lib.agent import BaseAgent
from tools import TOOLS  # [tools]

# [trim] Customisation knob — tune freely. Prompt window size, counted in
# messages (not tokens). For token-based trimming, pass the model as
# token_counter in query_chat instead of len.
MAX_HISTORY_MESSAGES = 40

# Customisation knob — edit freely; this is the template's personality
# dial. Keep the {name} placeholder (respond() fills it from the profile),
# and keep RUBRIC in evals/test_chat_quality.py in sync with the promises
# you make here.
_SYSTEM_PROMPT = """\
You are a friendly hello-world assistant.
The user's name is {name}; occasionally address them by it.
Use the provided tools when they help answer the question.
Keep replies short, warm, and helpful.
"""


class ChatAgent(BaseAgent):
    """Produces the assistant reply for a fully onboarded user."""

    async def respond(self, state: AppState) -> dict:
        """Node: append one assistant reply built from recent history."""

        name = state.get("profile", {}).get("name", "there")
        reply = await self.query_chat(
            _SYSTEM_PROMPT.format(name=name),
            state["messages"],
            tools=TOOLS,  # [tools]
            max_messages=MAX_HISTORY_MESSAGES,  # [trim]
        )
        return {"messages": [reply]}
