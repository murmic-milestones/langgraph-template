"""Tests for the provider-aware startup check in app/env.py.

``find_spec`` is always monkeypatched so results do not depend on which
provider packages happen to be installed in the developer's environment.
"""

from __future__ import annotations

import pytest

from app import env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test with no provider configuration and packages 'installed'."""

    for var in ("MODEL_NAME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(env, "find_spec", lambda name: object())


def test_missing_key_exits_with_guidance(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")
    with pytest.raises(SystemExit) as exc:
        env.check_environment()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_missing_provider_package_exits_with_install_hint(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "anthropic:claude-sonnet-5")
    monkeypatch.setattr(env, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        env.check_environment()
    assert "langchain-anthropic" in str(exc.value)
    assert ".[anthropic]" in str(exc.value)


def test_ollama_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "ollama:llama3.2")
    monkeypatch.setattr(env, "_ollama_preflight", lambda: None)
    env.check_environment()  # must not raise


def test_failing_preflight_exits_with_its_message(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "ollama:llama3.2")
    monkeypatch.setattr(env, "_ollama_preflight", lambda: "server down: hint")
    with pytest.raises(SystemExit) as exc:
        env.check_environment()
    assert "server down: hint" in str(exc.value)


def test_ollama_preflight_rejects_non_http_urls(monkeypatch):
    """Config must never hand urlopen a file:// (or other) scheme."""

    monkeypatch.setenv("OLLAMA_BASE_URL", "file:///etc/passwd")
    error = env._ollama_preflight()
    assert error is not None
    assert "http(s)" in error


def test_bare_model_name_is_treated_as_openai(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    with pytest.raises(SystemExit) as exc:
        env.check_environment()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_configured_provider_passes(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    env.check_environment()  # must not raise


def test_unknown_provider_is_deferred_to_init_chat_model(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "mystery:model-x")
    env.check_environment()  # must not raise — init_chat_model reports it


def test_extra_model_vars_are_checked_too(monkeypatch):
    """A broken per-agent override must fail fast even if MODEL_NAME is fine."""

    monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SUMMARISER_MODEL", "anthropic:claude-sonnet-5")
    with pytest.raises(SystemExit) as exc:
        env.check_environment(extra_model_vars=("SUMMARISER_MODEL",))
    assert "ANTHROPIC_API_KEY" in str(exc.value)
