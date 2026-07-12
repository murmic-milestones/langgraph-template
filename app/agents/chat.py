"""Main conversation agent — the hello-world payload of the template.

Replace or extend this agent with your project's real behaviour. It shows
the simplest node shape: build a system prompt from collected state, run
the (trimmed) message history through the model, append the reply.

Two optional features live here, each removable independently:

* ``[tools]`` — the model is bound to the tools in ``app/tools.py`` so it
  can request calls; the graph's tool node executes them. Removal steps
  are documented in ``app/tools.py``.
* ``[trim]`` — only the most recent ``MAX_HISTORY_MESSAGES`` messages are
  sent to the model (full history stays in state — trimming affects the
  prompt only). To remove: delete the ``trim_messages`` call and pass
  ``state["messages"]`` directly.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, trim_messages

from app.agents.base import BaseAgent
from app.state import AppState
from app.tools import TOOLS  # [tools]

# [trim] Customisation knob — tune freely. Prompt window size, counted in
# messages (not tokens). For token-based trimming, pass the model as
# token_counter instead of len.
MAX_HISTORY_MESSAGES = 40

# Customisation knob — edit freely; this is the template's personality
# dial. Keep the {name} placeholder (respond() fills it from the profile).
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

        # [trim] Keep the prompt bounded on long conversations.
        # start_on="human" avoids sending orphaned tool results.
        recent = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human",
        )

        llm = self.llm.bind_tools(TOOLS)  # [tools]
        reply = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT.format(name=name)), *recent]
        )
        return {"messages": [reply]}
