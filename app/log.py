"""Logging configuration — the only module allowed to configure logging.

The rule (enforced by tests/test_template_invariants.py):

* **Libraries emit, drivers configure.** Modules under ``app/`` do
  exactly one thing with logging: ``_logger = logging.getLogger(__name__)``
  and emit at standard levels. Handlers, formatters, and levels are set
  only here, and :func:`configure_logging` is called only by drivers
  (``main.py``, an adapter's ``set_up()``, your FastAPI startup) —
  never at import time.

Level conventions used across the template:

* ``DEBUG``   — diagnostics and anything derived from message content
  (counts, sizes). Conversation text and profile values are PII: they
  never appear at INFO or above, and are not logged verbatim at all.
* ``INFO``    — one line per significant lifecycle event (model
  initialised, LLM call completed, tool executed, turn finished).
* ``WARNING`` — degraded but continuing (fallbacks, skipped checks).
* ``ERROR``   — an operation failed; ``_logger.exception`` in handlers.

Vendor integration — this module is the seam, stdlib ``logging`` the
contract. Every vendor ships a ``logging.Handler``:

* Cloud/containers: ``LOG_FORMAT=json`` — one JSON object per line on
  stderr; Docker/Kubernetes/Cloud Logging collectors parse it natively.
* Python-native vendors: ``configure_logging(handlers=[SentryHandler()])``
  (or Datadog, Syslog, ...) — nothing under ``app/`` changes.
* OpenTelemetry: attach ``opentelemetry.sdk._logs.LoggingHandler`` the
  same way for OTLP export to any backend.

(LangSmith, enabled via env vars, covers LLM *tracing*; this system is
for application logs — they complement each other.)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

DEFAULT_LEVEL = "INFO"

# Third-party loggers that emit routine INFO noise (one line per HTTP
# request); capped at WARNING unless you re-raise them in a driver.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai")

# LogRecord attributes that are not user-supplied ``extra`` fields.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, ``extra`` fields included.

    Keeps the schema flat and collector-friendly: timestamp, level,
    logger, message, then any ``extra={...}`` keys (e.g. ``thread_id``).
    """

    # Name of the JSON field carrying the log level — subclasses override
    # it for collectors that key on a different field (GcpJsonFormatter).
    level_key = "level"

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            self.level_key: record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(
            (k, v)
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        )
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class GcpJsonFormatter(JsonFormatter):
    """JsonFormatter for Google Cloud Logging (Cloud Run, GKE, Agent Engine).

    Cloud Logging reads the log level from a JSON field named
    ``severity`` — without it, everything on stderr is ingested as
    ERROR. Python level names map to Cloud severities one-to-one.
    Attach via the handler seam::

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(GcpJsonFormatter())
        configure_logging(handlers=[handler])

    (For sending logs to Cloud Logging from *outside* GCP, use the
    ``google-cloud-logging`` package's ``CloudLoggingHandler`` through
    the same ``handlers=`` seam instead — see the README recipe.)
    """

    level_key = "severity"


def configure_logging(
    level: int | str | None = None,
    json_format: bool | None = None,
    handlers: list[logging.Handler] | None = None,
) -> None:
    """Configure root logging. Call from drivers only — never at import.

    Args:
        level: overrides the ``LOG_LEVEL`` env variable (default INFO).
        json_format: overrides ``LOG_FORMAT`` (``json`` or ``text``).
        handlers: replaces the default stderr handler entirely — the
            vendor seam (pass a Sentry/Datadog/OTel/... handler here).

    Logs go to **stderr** so stdout stays clean for the chat UI; both
    streams reach container log collectors.
    """

    if level is None:
        level = os.getenv("LOG_LEVEL", DEFAULT_LEVEL).upper()
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "text").strip().lower() == "json"

    if handlers is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            JsonFormatter()
            if json_format
            else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
        )
        handlers = [handler]

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = handlers

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
