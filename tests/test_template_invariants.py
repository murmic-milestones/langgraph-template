"""Machine-checked template invariants.

These turn the prose rules in CLAUDE.md into failing tests, so any tool
(AI or human) that violates a pattern finds out from pytest instead of
from review. When one of these fails, fix the code to match the pattern
— only change the test if the pattern itself is deliberately changing
(and then update CLAUDE.md/README in the same commit).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Provider integration packages that must never be imported by agents —
# models are constructed only via the get_llm factory (this seam is what
# lets tests swap in the fake LLM).
_PROVIDER_PACKAGES = (
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_google_vertexai",
    "langchain_ollama",
)


def test_providers_stay_in_sync_across_config_files() -> None:
    """Every PROVIDERS row needs its pyproject extra and .env.example row."""

    from app.env import PROVIDERS

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    for name, provider in PROVIDERS.items():
        extra = re.search(r"\.\[(\w+)\]", provider.install_hint)
        if extra:  # base-install providers (openai) have no extra
            assert re.search(rf"^{extra.group(1)} = \[", pyproject, re.M), (
                f"provider '{name}': extra [{extra.group(1)}] is named in its "
                "install hint but not defined in pyproject.toml"
            )
        assert provider.package.replace("_", "-") in pyproject, (
            f"provider '{name}': package {provider.package} is not pinned "
            "anywhere in pyproject.toml"
        )
        assert name in env_example, (
            f"provider '{name}' is missing from the .env.example table"
        )


def test_agents_never_import_provider_packages() -> None:
    """Agents must get models via BaseAgent.llm, not construct their own."""

    for path in (ROOT / "app" / "agents").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                module = getattr(node, "module", None) or ""
                imported = [module, *(alias.name for alias in node.names)]
                offenders = [
                    name for name in imported if name.startswith(_PROVIDER_PACKAGES)
                ]
                assert not offenders, (
                    f"{path.name} imports {offenders} — construct models via "
                    "BaseAgent.llm / get_llm so the test fake keeps working"
                )


def test_all_graph_nodes_are_async() -> None:
    """Sync nodes would raise at runtime (the graph runs via ainvoke only)."""

    from app.graph import build_graph

    for name, spec in build_graph().builder.nodes.items():
        afunc = getattr(spec.runnable, "afunc", None)
        assert afunc is not None and inspect.iscoroutinefunction(afunc), (
            f"node '{name}' has no async implementation — node methods must "
            "be `async def` (see CLAUDE.md)"
        )


def test_all_gates_are_sync_predicates() -> None:
    """Routing functions run inline; they must be plain sync callables."""

    from app.graph import build_graph

    for source, branches in build_graph().builder.branches.items():
        for branch in branches.values():
            path = getattr(branch.path, "func", None) or branch.path
            assert not inspect.iscoroutinefunction(path), (
                f"gate on '{source}' is async — gate methods must be sync "
                "predicates (see CLAUDE.md)"
            )
