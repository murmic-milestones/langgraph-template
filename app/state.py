"""Typed graph state.

The state schema is the contract every node reads from and writes to.
Each key may declare a *reducer* (via ``Annotated``) that controls how a
node's partial return value is combined with the existing state:

* ``messages`` uses LangGraph's built-in :func:`add_messages` reducer, so a
  node returns only the *new* messages and they are appended (and
  de-duplicated by id) rather than overwriting the history.
* ``profile`` has no reducer, so a returned value replaces the old one.
  Nodes therefore return the full (copied + updated) profile dict.

Extend ``Profile``/``AppState`` with your own domain fields; keep them
JSON-serialisable so any checkpointer can persist them.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Profile(TypedDict, total=False):
    """Facts collected about the user during onboarding."""

    name: str


class AppState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    profile: Profile
