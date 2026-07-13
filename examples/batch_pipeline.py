"""Side-effecting batch pipeline. [OPTIONAL FEATURE: delete this file +
tests/test_batch_pipeline.py]

The runnable version of README pattern 16 ("Side-effecting agents"). The
main app's agents are pure — state in, state out, persistence owned by
the checkpointer. This pipeline's agent *writes its own output* (a CSV
row per document), which changes the recipe:

* **One graph run per unit of work** — the driver (``run_batch``) owns
  the file loop, ordering, and ``--limit``; the graph only ever sees a
  single document.
* **Effect dependencies are injected through the constructor** (the
  ``SummaryStore``), never module-level paths — tests point the agent at
  a ``tmp_path`` and the fake-LLM seam stays intact.
* **Nodes are idempotent and outputs double as resume state**: the node
  checks its own store first and no-ops when the row exists, so
  re-running an interrupted batch is always safe — no checkpointer
  needed at all.
* **The store is lazy** — nothing touches the filesystem until first
  use, keeping module-import side-effect free (README pattern 2).

Run it (needs a configured model, see .env.example)::

    python -m examples.batch_pipeline path/to/docs --limit 5

Summarises every ``*.txt`` in the folder into ``summaries.csv`` beside
them; re-running skips documents already summarised.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agents.base import BaseAgent

_logger = logging.getLogger(__name__)

_PROMPT = """\
Summarise the user's document in one plain sentence.
"""


class Summary(BaseModel):
    """Structured summarisation result."""

    summary: str = Field(description="One-sentence summary of the document.")


class BatchState(TypedDict, total=False):
    """State for one unit of work (one document)."""

    source: str  # file name relative to the docs folder
    summary: str
    skipped: bool  # True when the store already had this document


class SummaryStore:
    """CSV-backed output store: one ``(source, summary)`` row per document.

    Lazy on purpose: the constructor stores paths only, and the CSV is
    first read/created on first use — so building the graph (which may
    happen at import time) performs no filesystem I/O.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)
        self._done: set[str] | None = None  # loaded on first use

    def _load(self) -> set[str]:
        if self._done is None:
            self._done = set()
            if self._path.exists():
                with self._path.open(newline="", encoding="utf-8") as handle:
                    self._done = {row["source"] for row in csv.DictReader(handle)}
        return self._done

    def done(self, source: str) -> bool:
        return source in self._load()

    def add(self, source: str, summary: str) -> None:
        new_file = not self._path.exists()
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("source", "summary"))
            if new_file:
                writer.writeheader()
            writer.writerow({"source": source, "summary": summary})
        self._load().add(source)


class SummariserAgent(BaseAgent):
    """Summarises one document and writes the result to its store."""

    def __init__(self, store: SummaryStore, docs_dir: str | Path) -> None:
        super().__init__(temperature=0.1)
        self._store = store
        self._docs = Path(docs_dir)

    async def summarise(self, state: BatchState) -> dict:
        """Node: summarise ``state['source']`` unless already done."""

        source = state["source"]
        if self._store.done(source):
            # Idempotency: the output row doubles as resume state.
            _logger.debug("already summarised — skipping")
            return {"skipped": True}

        text = (self._docs / source).read_text(encoding="utf-8")
        result = await self.query_structured(
            _PROMPT, [HumanMessage(content=text)], Summary
        )
        self._store.add(source, result.summary)
        _logger.info("document summarised", extra={"source": source})
        return {"summary": result.summary, "skipped": False}


def build_batch_graph(agent: SummariserAgent):
    """One-node graph; deliberately no checkpointer (see module docstring)."""

    builder = StateGraph(BatchState)
    builder.add_node("summarise", agent.summarise)
    builder.add_edge(START, "summarise")
    builder.add_edge("summarise", END)
    return builder.compile()


async def run_batch(docs_dir: str | Path, limit: int | None = None) -> int:
    """Summarise up to ``limit`` documents; return how many were new."""

    docs_dir = Path(docs_dir)
    store = SummaryStore(docs_dir / "summaries.csv")
    graph = build_batch_graph(SummariserAgent(store, docs_dir))

    processed = 0
    for name in sorted(p.name for p in docs_dir.glob("*.txt"))[:limit]:
        state = await graph.ainvoke({"source": name})
        if not state.get("skipped"):
            processed += 1
            print(f"summarised {name}")
        else:
            print(f"skipped    {name} (already done)")
    return processed


def main() -> None:
    from dotenv import load_dotenv

    from app.env import EnvironmentCheckError, check_environment
    from app.log import configure_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_dir", help="folder containing *.txt documents")
    parser.add_argument("--limit", type=int, help="process at most N documents")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    try:
        check_environment()
    except EnvironmentCheckError as error:
        sys.exit(str(error))

    processed = asyncio.run(run_batch(args.docs_dir, args.limit))
    print(f"done: {processed} new document(s) summarised")


if __name__ == "__main__":
    main()
