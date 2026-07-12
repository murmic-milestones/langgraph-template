"""Offline tests for the Agent Engine adapter.

Delete this file together with ``examples/agent_engine_app.py``.

These verify the platform contract locally (pickling, set_up, the query
round-trip, JSON-serialisable output) with the fake LLM — no GCP project
or vertexai SDK needed. The deployment itself can only be validated
against a real project.
"""

from __future__ import annotations

import json
import pickle

import pytest

from app.agents.greeter import NameCheck
from examples.agent_engine_app import AgentEngineApp


def test_instance_is_pickleable_before_set_up() -> None:
    """The platform pickles the configured instance to ship it."""

    app = AgentEngineApp(model="google_vertexai:gemini-2.5-flash")
    restored = pickle.loads(pickle.dumps(app))
    assert restored._model == "google_vertexai:gemini-2.5-flash"


def test_register_operations_shape() -> None:
    ops = AgentEngineApp().register_operations()
    assert ops[""] == ["query"]
    assert ops["async"] == ["async_query"]


def test_set_up_applies_model_override(monkeypatch, fake) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    app = AgentEngineApp(model="google_vertexai:gemini-2.5-flash")
    app.set_up()
    import os

    assert os.environ["MODEL_NAME"] == "google_vertexai:gemini-2.5-flash"


def test_query_round_trip_is_json_serialisable(fake) -> None:
    """A full turn through the real graph, returned as plain JSON data."""

    app = AgentEngineApp()
    app.set_up()

    fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    result = app.query(message="I'm Paul", thread_id="engine-test")

    assert result["reply"] == "Hello Paul!"
    assert result["profile"]["name"] == "Paul"
    json.dumps(result)  # platform requires JSON-serialisable output


def test_sync_query_drops_cached_models_between_loops(fake, monkeypatch) -> None:
    """query() bridges with asyncio.run, i.e. a fresh event loop per call;
    a model instance cached from an earlier loop would raise "Event loop
    is closed" (see app.llm.reset_llm_cache). The fake LLM has no HTTP
    client to trip over, so assert the cache reset itself.
    """

    import app.llm

    resets: list[bool] = []
    monkeypatch.setattr(app.llm, "reset_llm_cache", lambda: resets.append(True))

    engine = AgentEngineApp()
    engine.set_up()
    fake.structured_results[NameCheck] = NameCheck(name=None, reply="Who are you?")
    engine.query(message="hello", thread_id="loop-test")

    assert resets, "query() must reset_llm_cache() before entering a new loop"


def test_blank_thread_id_is_rejected(fake) -> None:
    """No shared-default session: an empty thread_id must be refused."""

    app = AgentEngineApp()
    app.set_up()
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="thread_id is required"):
            app.query(message="hi", thread_id=bad)


def test_threads_are_isolated_per_conversation(fake) -> None:
    app = AgentEngineApp()
    app.set_up()

    fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")
    fake.reply_text = "Hello Paul!"
    app.query(message="I'm Paul", thread_id="thread-a")

    fake.structured_results[NameCheck] = NameCheck(name=None, reply="Who are you?")
    result_b = app.query(message="hello", thread_id="thread-b")
    assert result_b["reply"] == "Who are you?"
    assert result_b["profile"] == {}
