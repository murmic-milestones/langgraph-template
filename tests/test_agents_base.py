"""Tests for the BaseAgent helpers: per-agent models and image input."""

from __future__ import annotations

import base64

import pytest

import app.agents.base as base
from app.agents.base import BaseAgent, image_message

FAKE_JPEG = b"\xff\xd8\xff\xe0-not-really-a-jpeg-"


def test_image_message_builds_base64_block(tmp_path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(FAKE_JPEG)

    message = image_message("describe this", image)

    text_block, image_block = message.content
    assert text_block == {"type": "text", "text": "describe this"}
    assert image_block["type"] == "image"
    assert image_block["source_type"] == "base64"
    assert image_block["mime_type"] == "image/png"
    assert base64.b64decode(image_block["data"]) == FAKE_JPEG


def test_image_message_rejects_oversized_files_without_reading(
    tmp_path, monkeypatch
) -> None:
    """The size guard exists to keep runaway files out of memory, so it
    must fire from stat() alone — reading the file first would defeat it."""

    monkeypatch.setattr(base, "MAX_IMAGE_BYTES", 10)
    image = tmp_path / "big.png"
    image.write_bytes(b"x" * 11)
    monkeypatch.setattr(
        base.Path,
        "read_bytes",
        lambda self: pytest.fail("oversized file was read into memory"),
    )

    with pytest.raises(ValueError, match="refusing to embed"):
        image_message("describe this", image)


def test_agent_without_model_env_uses_shared_default(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        base, "get_llm", lambda temperature=0.3, model=None: seen.update(m=model)
    )

    _ = BaseAgent().llm
    assert seen["m"] is None


def test_model_env_overrides_shared_model(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        base, "get_llm", lambda temperature=0.3, model=None: seen.update(m=model)
    )
    monkeypatch.setenv("SUMMARISER_MODEL", "ollama:llama3.2")

    _ = BaseAgent(model_env="SUMMARISER_MODEL").llm
    assert seen["m"] == "ollama:llama3.2"


def test_unset_model_env_falls_back_to_shared_model(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        base, "get_llm", lambda temperature=0.3, model=None: seen.update(m=model)
    )
    monkeypatch.delenv("SUMMARISER_MODEL", raising=False)

    _ = BaseAgent(model_env="SUMMARISER_MODEL").llm
    assert seen["m"] is None
