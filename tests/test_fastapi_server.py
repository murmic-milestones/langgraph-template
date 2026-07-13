"""Tests for the FastAPI serving example (JSON + SSE endpoints).

Delete this file together with ``examples/fastapi_server.py`` and the
``[serve]`` extra.

Everything runs offline: the app is exercised through httpx's ASGI
transport with a graph backed by the fake LLM. The skip below fires
only when the serve dependencies were removed from the environment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")

import httpx  # noqa: E402  (import after the skip guard)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.agents.greeter import NameCheck  # noqa: E402
from app.graph import build_graph  # noqa: E402
from examples.fastapi_server import create_app  # noqa: E402
from fakes import run  # noqa: E402


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(
        app=create_app(build_graph(checkpointer=InMemorySaver()))
    )
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_chat_endpoint_runs_one_turn(fake) -> None:
    fake.structured_results[NameCheck] = NameCheck(
        name=None, reply="What's your first name?"
    )

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post(
                "/chat", json={"text": "hi"}, headers={"x-user-id": "alice"}
            )

    response = run(scenario())
    assert response.status_code == 200
    assert response.json() == {"reply": "What's your first name?"}


def test_sessions_are_isolated_per_user(fake) -> None:
    """thread_id comes from the (demo-)authenticated user, so two users
    on the same server must not share onboarding state."""

    async def scenario() -> tuple[dict, dict]:
        async with _client() as client:
            fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")
            fake.reply_text = "Hello Paul!"
            first = await client.post(
                "/chat", json={"text": "I'm Paul"}, headers={"x-user-id": "paul"}
            )
            # A different user starts from scratch: onboarding runs again.
            fake.structured_results[NameCheck] = NameCheck(
                name=None, reply="Who are you?"
            )
            second = await client.post(
                "/chat", json={"text": "hello"}, headers={"x-user-id": "mallory"}
            )
            return first.json(), second.json()

    paul, mallory = run(scenario())
    assert paul == {"reply": "Hello Paul!"}
    assert mallory == {"reply": "Who are you?"}


def test_missing_user_header_is_rejected(fake) -> None:
    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post("/chat", json={"text": "hi"})

    assert run(scenario()).status_code == 422  # FastAPI validates the header


def test_stream_endpoint_emits_sse_and_done(fake) -> None:
    """The fake LLM emits no token events, so this exercises the
    fall-back-to-state path: one data event with the reply, then [DONE]."""

    fake.structured_results[NameCheck] = NameCheck(
        name=None, reply="What's your first name?"
    )

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post(
                "/chat/stream", json={"text": "hi"}, headers={"x-user-id": "alice"}
            )

    response = run(scenario())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line]
    assert events[0] == 'data: {"token": "What\'s your first name?"}'
    assert events[-1] == "data: [DONE]"
