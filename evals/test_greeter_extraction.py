"""Eval: does the greeter extract names correctly from real phrasings?

Canonical eval type 1 — **programmatic scoring**: known inputs, exact
expected outputs, no judge needed. This is the eval that catches a bad
edit to ``_EXTRACTION_PROMPT`` in ``app/agents/greeter.py``.

Cheap and near-deterministic (structured output, low temperature), but
still a real model: if a case proves flaky for *your* model, that is
signal — tighten the prompt or reconsider the case, don't loop retries.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from app.agents.greeter import GreeterAgent

# (user message, expected extracted name or None). The None cases matter
# most: names that belong to someone else, or a refusal, must not be
# extracted.
CASES = [
    ("hi!", None),
    ("I'm Paul", "paul"),
    ("my name is María", "maría"),
    ("everyone calls me JJ", "jj"),
    ("It's Paul-Henri.", "paul-henri"),
    ("paul here 👋", "paul"),
    ("I'd rather not say", None),
    ("My friend's name is Anna", None),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_name_extraction(text: str, expected: str | None) -> None:
    result = asyncio.run(
        GreeterAgent().collect_name(
            {"messages": [HumanMessage(content=text)], "profile": {}}
        )
    )

    extracted = result.get("profile", {}).get("name")
    if expected is None:
        assert extracted is None, f"wrongly extracted {extracted!r} from {text!r}"
        # No name -> the node must ask a question instead.
        assert result.get("messages"), "expected a follow-up question"
    else:
        assert extracted is not None, f"failed to extract a name from {text!r}"
        assert extracted.lower() == expected
