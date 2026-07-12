"""Tools the chat agent may call. [OPTIONAL FEATURE: tool calling]

Add tools by defining a ``@tool``-decorated function here and appending it
to ``TOOLS`` — the chat agent and the graph's tool node both read that
list, so nothing else needs wiring.

To REMOVE tool calling from the template entirely:
1. delete this file;
2. in ``app/agents/chat.py`` drop the ``bind_tools`` call (marked
   ``[tools]``);
3. in ``app/graph.py`` delete the three ``[tools]`` lines and restore the
   plain ``builder.add_edge("chat", END)``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.tools import tool

_logger = logging.getLogger(__name__)


@tool
def get_current_time() -> str:
    """Return the current date and time in UTC (ISO-8601)."""

    # Tools are actions — log each execution (INFO) for an audit trail.
    _logger.info("tool executed: get_current_time")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


TOOLS = [get_current_time]
