"""Mermaid visualisation helpers for the compiled graph."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph.state import CompiledStateGraph


def to_mermaid(graph: CompiledStateGraph) -> str:
    """Return the graph as Mermaid source (paste into mermaid.live)."""

    return graph.get_graph().draw_mermaid()


def to_png(graph: CompiledStateGraph, output_path: str | Path | None = None) -> bytes:
    """Render the graph to PNG bytes, optionally writing them to disk.

    Uses the mermaid.ink web API under the hood, so it needs network access.
    """

    png_data: bytes = graph.get_graph().draw_mermaid_png()
    if output_path is not None:
        Path(output_path).write_bytes(png_data)
    return png_data
