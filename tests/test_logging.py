"""Tests for app/log.py and the template's logging conventions."""

from __future__ import annotations

import json
import logging

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.greeter import NameCheck
from app.graph import build_graph
from app.log import configure_logging
from fakes import config, onboard_paul, run


@pytest.fixture
def restore_root_logging():
    """Snapshot and restore root logger state around configure_logging tests."""

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_log_level_env_is_respected(monkeypatch, capsys, restore_root_logging):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    configure_logging()

    logging.getLogger("app.test").info("info-line")
    logging.getLogger("app.test").warning("warning-line")

    err = capsys.readouterr().err
    assert "info-line" not in err
    assert "warning-line" in err


def test_json_format_emits_parseable_lines_with_extras(
    monkeypatch, capsys, restore_root_logging
):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging(json_format=True)

    logging.getLogger("app.test").info("hello", extra={"thread_id": "t-1"})

    line = capsys.readouterr().err.strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["level"] == "INFO"
    assert entry["message"] == "hello"
    assert entry["logger"] == "app.test"
    assert entry["thread_id"] == "t-1"  # extras survive into the JSON


def test_lifecycle_events_are_logged(fake, caplog):
    """One INFO line per significant operation (LLM calls, replies)."""

    with caplog.at_level(logging.INFO):
        run(onboard_paul(build_graph(checkpointer=InMemorySaver()), fake))

    messages = [r.getMessage() for r in caplog.records]
    assert any("structured call ok" in m for m in messages)
    assert any("chat reply generated" in m for m in messages)


def test_no_conversation_content_in_logs(fake, caplog):
    """PII rule: user text must never appear in log records, at any level."""

    secret = "XYZZY-SECRET-42"
    graph = build_graph(checkpointer=InMemorySaver())
    fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")

    with caplog.at_level(logging.DEBUG):
        run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=f"I'm Paul, {secret}")]},
                config("pii-test"),
            )
        )

    for record in caplog.records:
        assert secret not in record.getMessage(), (
            f"conversation content leaked into logs via {record.name}"
        )
