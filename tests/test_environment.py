"""Tests for the provider-aware startup check in main.py."""

from __future__ import annotations

import pytest

import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test with no provider configuration."""

    for var in ("MODEL_NAME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_missing_key_exits_with_guidance(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")
    with pytest.raises(SystemExit) as exc:
        main.check_environment()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_missing_provider_package_exits_with_install_hint(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "anthropic:claude-sonnet-5")
    monkeypatch.setattr(main, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        main.check_environment()
    assert "langchain-anthropic" in str(exc.value)
    assert ".[anthropic]" in str(exc.value)


def test_ollama_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "ollama:llama3.2")
    monkeypatch.setattr(main, "find_spec", lambda name: object())
    main.check_environment()  # must not raise


def test_bare_model_name_is_treated_as_openai(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    with pytest.raises(SystemExit) as exc:
        main.check_environment()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_configured_provider_passes(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    main.check_environment()  # must not raise


def test_unknown_provider_is_deferred_to_init_chat_model(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "mystery:model-x")
    main.check_environment()  # must not raise — init_chat_model reports it
