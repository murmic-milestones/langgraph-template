"""Onboarding agent: collect the user's name before the main chat.

Demonstrates the *gated onboarding* pattern:

* ``collect_name`` (node) — no-op if the name is already known; otherwise
  scans the conversation with a structured-output call. If the user has
  stated their name it is written to ``profile``; if not, the agent asks
  for it and the gate below ends the turn so the user can answer.
* ``is_name_set`` (gate) — routes to the next stage once the fact has
  been collected, otherwise to ``END``.

Because the graph is compiled with a checkpointer, ending the turn is
enough: the next ``invoke`` on the same thread resumes with everything
collected so far, and completed stages pass straight through.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agents.base import BaseAgent
from app.state import AppState
from pydantic import BaseModel, Field

_EXTRACTION_PROMPT = """\
You are the onboarding step of an assistant. Scan the conversation and work
out the user's first name. Only use a name the user has actually stated
about themselves; give more weight to recent messages. Do not guess.

If they have not given their name, write one short, friendly question
asking for their first name.
"""


class NameCheck(BaseModel):
    """Structured result of scanning the conversation for the user's name."""

    name: str | None = Field(
        default=None,
        description="The user's first name if they have stated it, else null.",
    )
    reply: str = Field(
        default="",
        description=(
            "If no name was found: a short, friendly question asking for "
            "their first name. Empty string otherwise."
        ),
    )


class GreeterAgent(BaseAgent):
    """Collects the user's name into ``state['profile']``."""

    def collect_name(self, state: AppState) -> dict:
        """Node: set ``profile.name`` from the conversation, or ask for it."""

        profile = state.get("profile", {})
        if profile.get("name"):
            return {}  # Already onboarded — pass straight through.

        result = self.query_structured(
            _EXTRACTION_PROMPT, state["messages"], NameCheck
        )

        if result.name:
            return {"profile": {**profile, "name": result.name}}

        question = result.reply or "Hi! Before we start, what's your first name?"
        return {"messages": [AIMessage(content=question)]}

    def is_name_set(self, state: AppState) -> bool:
        """Gate: ``True`` once the user's name has been collected."""

        return bool(state.get("profile", {}).get("name"))
