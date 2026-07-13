"""Serve the graph over HTTP with FastAPI. [OPTIONAL FEATURE: delete
this file + tests/test_fastapi_server.py + the [serve] extra in
pyproject.toml]

The working version of the README's "Serving over HTTP" section: a JSON
endpoint for request/response clients and an SSE endpoint that streams
tokens as they arrive, using the same node-whitelist rule as the CLI
(``STREAMING_NODES`` in ``main.py``) and the same fall-back-to-state for
turns that end in a non-streaming node.

SECURITY: ``thread_id`` is derived server-side from the authenticated
user — never from the request body or path (see README "Security").
``current_user`` below trusts an ``x-user-id`` header, WHICH IS NOT
AUTHENTICATION — it is a stand-in so the example runs without an auth
stack. Replace it with your real dependency (session cookie, JWT, ...)
before exposing this to anyone.

Run it (needs ``pip install -e ".[serve]"`` and a configured model)::

    python -m examples.fastapi_server

    curl -X POST localhost:8000/chat -H "content-type: application/json" \
         -H "x-user-id: alice" -d "{\"text\": \"hi\"}"
    curl -N -X POST localhost:8000/chat/stream ...   # same shape, SSE
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, Header
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from main import STREAMING_NODES

_logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    text: str


async def current_user(x_user_id: str = Header()) -> str:
    """DEMO ONLY — replace with real authentication.

    A client-supplied header is exactly what the security notes forbid
    as a session key; it stands in here so the example is runnable. The
    contract to keep: this dependency returns an identity the *server*
    verified, and the thread_id is derived from nothing else.
    """

    return x_user_id


def _config(user: str) -> dict:
    return {"configurable": {"thread_id": f"user:{user}"}}


async def _sse_events(graph, config: dict, text: str) -> AsyncIterator[str]:
    """One SSE data event per token; state fallback for silent turns."""

    streamed = False
    async for chunk, metadata in graph.astream(
        {"messages": [HumanMessage(content=text)]}, config, stream_mode="messages"
    ):
        if metadata.get("langgraph_node") in STREAMING_NODES and chunk.text:
            streamed = True
            yield f"data: {json.dumps({'token': chunk.text})}\n\n"

    if not streamed:
        # Turn ended in a non-streaming node (e.g. the onboarding
        # question) — send the reply from the checkpointed state.
        state = await graph.aget_state(config)
        reply = state.values["messages"][-1].text
        yield f"data: {json.dumps({'token': reply})}\n\n"

    yield "data: [DONE]\n\n"


def create_app(graph) -> FastAPI:
    """Build the HTTP app around a compiled graph.

    The graph is injected (not built here) so tests can pass one backed
    by the fake LLM, and the process owner decides the checkpointer.
    """

    app = FastAPI(title="langgraph-template chat")

    @app.post("/chat")
    async def chat(req: ChatRequest, user: str = Depends(current_user)) -> dict:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=req.text)]}, _config(user)
        )
        return {"reply": state["messages"][-1].text}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest, user: str = Depends(current_user)):
        return StreamingResponse(
            _sse_events(graph, _config(user), req.text),
            media_type="text/event-stream",
        )

    return app


def main() -> None:
    from dotenv import load_dotenv
    from langgraph.checkpoint.memory import InMemorySaver

    from app.env import EnvironmentCheckError, check_environment
    from app.graph import build_graph
    from app.log import configure_logging

    load_dotenv()
    configure_logging()
    try:
        check_environment()
    except EnvironmentCheckError as error:
        sys.exit(str(error))

    try:
        import uvicorn
    except ImportError:
        sys.exit('uvicorn is not installed — pip install -e ".[serve]"')

    # InMemorySaver: sessions last as long as the process. Swap in a
    # SQLite/Postgres saver for sessions that survive restarts.
    uvicorn.run(create_app(build_graph(checkpointer=InMemorySaver())), port=8000)


if __name__ == "__main__":
    main()
