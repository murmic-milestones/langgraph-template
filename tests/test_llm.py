"""Tests for the model factory's resolution and caching semantics.

These use the real ``get_llm`` (no network — construction only), so the
``fake`` fixture is deliberately not requested.
"""

from __future__ import annotations

import pytest

from app.llm import get_llm


@pytest.fixture(autouse=True)
def openai_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")


def test_model_name_is_reread_on_every_call(monkeypatch):
    first = get_llm()
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4.1-mini")
    second = get_llm()

    assert first.model_name == "gpt-4o-mini"
    assert second.model_name == "gpt-4.1-mini"


def test_same_configuration_reuses_one_instance():
    assert get_llm() is get_llm()


def test_explicit_model_overrides_env():
    llm = get_llm(model="openai:gpt-4.1-mini")
    assert llm.model_name == "gpt-4.1-mini"
