"""Adapter for Google's Agent Engine runtime (Gemini Enterprise Agent
Platform). [OPTIONAL FEATURE: delete this file + tests/test_agent_engine.py
+ the [vertexai] extra in pyproject.toml]

The platform deploys any pickle-able Python class that implements
``query()`` (and optionally async/streaming variants):

* ``__init__`` — configuration only; the instance is pickled and shipped,
  so it must hold plain data, never clients or compiled graphs.
* ``set_up()`` — runs once server-side before traffic; this is where the
  graph is built.
* ``register_operations()`` — maps execution modes to method names
  ("" = sync, "async" = coroutine).

The template's own layering does the rest: ``build_graph()`` accepts any
checkpointer, and the model comes from the ``MODEL_NAME`` env variable
(set per deployment via ``env_vars``). ``check_environment()`` is a CLI
concern and is deliberately not called here.

Sessions: the platform passes/derives a thread id per conversation; we
map it straight onto the checkpointer's ``thread_id``. The default
``InMemorySaver`` keeps sessions per-container only — swap in a durable
saver for real deployments (commented below).

Deploy (needs ``pip install -e ".[vertexai]"`` and
``gcloud auth application-default login``)::

    import vertexai
    from vertexai import agent_engines
    from examples.agent_engine_app import AgentEngineApp

    vertexai.init(project="MY_PROJECT", location="us-central1",
                  staging_bucket="gs://MY_BUCKET")
    remote = agent_engines.create(
        AgentEngineApp(model="google_vertexai:gemini-2.5-flash"),
        requirements=[
            "langgraph>=1.2,<2",
            "langchain>=1,<2",
            "langchain-core>=1.4,<2",
            "langchain-google-vertexai>=3,<4",
        ],
        extra_packages=["app", "examples"],  # ship both packages the class needs
    )
    print(remote.query(message="hi", thread_id="user-1"))

Verified offline (contract shape, pickling, query round-trip via the
fake LLM); the deployment itself requires a GCP project — run one real
``remote.query`` before trusting a new deployment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage

_logger = logging.getLogger(__name__)


class AgentEngineApp:
    """Wraps the template graph in the Agent Engine custom-agent contract.

    Args:
        model: optional ``"provider:model"`` string applied as
            ``MODEL_NAME`` during ``set_up()`` — lets one deployment pin
            its model without further env configuration. On the platform
            prefer ``google_vertexai:...`` models, which authenticate via
            the runtime's service account (no API key).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model  # plain data only — instances are pickled

    def set_up(self) -> None:
        """Build the graph server-side (never in __init__)."""

        if self._model:
            os.environ["MODEL_NAME"] = self._model

        import sys

        from langgraph.checkpoint.memory import InMemorySaver

        from app.graph import build_graph
        from app.log import GcpJsonFormatter, configure_logging

        # set_up is this deployment's driver — it owns log configuration.
        # GcpJsonFormatter adds the `severity` field Cloud Logging keys
        # on (plain JSON on stderr would all be ingested as ERROR).
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(GcpJsonFormatter())
        configure_logging(handlers=[handler])

        # InMemorySaver = sessions survive within one container only.
        # For durable sessions use e.g. langchain-google-cloud-sql-pg:
        #   engine = PostgresEngine.from_instance(...)
        #   engine.init_checkpoint_table()
        #   saver = PostgresSaver.create_sync(engine)
        self.graph = build_graph(checkpointer=InMemorySaver())

    async def async_query(
        self, *, message: str, thread_id: str = "default"
    ) -> dict[str, Any]:
        """One chat turn; returns a JSON-serialisable result."""

        start = time.perf_counter()
        state = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            {"configurable": {"thread_id": thread_id}},
        )
        _logger.info(
            "query complete: duration_ms=%.0f",
            (time.perf_counter() - start) * 1000,
            extra={"thread_id": thread_id},
        )
        return {
            "reply": state["messages"][-1].text,
            "profile": dict(state.get("profile", {})),
        }

    def query(self, *, message: str, thread_id: str = "default") -> dict[str, Any]:
        """Sync entry point required by the platform.

        The platform calls this outside any event loop, so bridging the
        async graph with ``asyncio.run`` is safe here.
        """

        return asyncio.run(self.async_query(message=message, thread_id=thread_id))

    def register_operations(self) -> dict[str, list[str]]:
        return {"": ["query"], "async": ["async_query"]}
