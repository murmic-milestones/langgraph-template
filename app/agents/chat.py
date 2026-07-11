"""Main conversation agent — the hello-world payload of the template.

Replace or extend this agent with your project's real behaviour. It shows
the simplest node shape: build a system prompt from collected state, run
the message history through the model, append the reply.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from app.agents.base import BaseAgent
from app.state import AppState

_SYSTEM_PROMPT = """\
You are a friendly hello-world assistant.
The user's name is {name}; occasionally address them by it.
Keep replies short, warm, and helpful.
"""


class ChatAgent(BaseAgent):
    """Produces the assistant reply for a fully onboarded user."""

    def respond(self, state: AppState) -> dict:
        """Node: append one assistant reply built from the full history."""

        name = state.get("profile", {}).get("name", "there")
        reply = self.llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT.format(name=name)),
                *state["messages"],
            ]
        )
        return {"messages": [reply]}
