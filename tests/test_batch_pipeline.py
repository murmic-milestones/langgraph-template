"""Tests for the side-effecting batch pipeline example.

Delete this file together with ``examples/batch_pipeline.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from examples.batch_pipeline import (
    SummariserAgent,
    Summary,
    SummaryStore,
    build_batch_graph,
    run_batch,
)
from fakes import run


def _make_docs(tmp_path: Path, count: int = 3) -> Path:
    for i in range(count):
        (tmp_path / f"doc{i}.txt").write_text(f"document number {i}", encoding="utf-8")
    return tmp_path


def _rows(docs_dir: Path) -> list[dict]:
    with (docs_dir / "summaries.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_batch_summarises_every_document(fake, tmp_path) -> None:
    docs = _make_docs(tmp_path)
    fake.structured_results[Summary] = Summary(summary="A short doc.")

    assert run(run_batch(docs)) == 3
    rows = _rows(docs)
    assert [r["source"] for r in rows] == ["doc0.txt", "doc1.txt", "doc2.txt"]
    assert all(r["summary"] == "A short doc." for r in rows)


def test_rerun_is_idempotent_and_calls_no_model(fake, tmp_path) -> None:
    """Outputs double as resume state: a re-run must skip finished work.
    Not queueing the schema proves no LLM call happens (a structured call
    without a queued result fails the test)."""

    docs = _make_docs(tmp_path)
    fake.structured_results[Summary] = Summary(summary="A short doc.")
    run(run_batch(docs))

    fake.structured_results.clear()  # any further structured call would fail
    assert run(run_batch(docs)) == 0
    assert len(_rows(docs)) == 3  # no duplicate rows


def test_limit_bounds_the_work_and_resumes_cleanly(fake, tmp_path) -> None:
    docs = _make_docs(tmp_path)
    fake.structured_results[Summary] = Summary(summary="A short doc.")

    assert run(run_batch(docs, limit=2)) == 2
    assert len(_rows(docs)) == 2
    # The next run picks up the remainder — an interrupted batch resumes.
    assert run(run_batch(docs)) == 1
    assert len(_rows(docs)) == 3


def test_store_and_graph_construction_touch_no_files(fake, tmp_path) -> None:
    """Lazy stores keep import-time graph building side-effect free."""

    store = SummaryStore(tmp_path / "summaries.csv")
    build_batch_graph(SummariserAgent(store, tmp_path))
    assert list(tmp_path.iterdir()) == []  # nothing created yet
