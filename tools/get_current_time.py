"""get_current_time — return the current UTC time.

One tool per file (see tools/__init__.py for the how-to + security rules).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from langchain_core.tools import tool

_logger = logging.getLogger(__name__)


@tool
def get_current_time() -> str:
    """Return the current date and time in UTC (ISO-8601)."""

    # Tools are actions — log each execution (INFO) for an audit trail.
    _logger.info("tool executed: get_current_time")
    return datetime.now(UTC).isoformat(timespec="seconds")
