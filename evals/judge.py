"""Reusable LLM-as-judge with a structured verdict.

Dogfoods two template patterns: the verdict is enforced by Pydantic
structured output (no parsing), and the judge model is configurable via
``EVAL_JUDGE_MODEL`` (any ``provider:model`` string), falling back to
``MODEL_NAME`` — best practice is judging a cheap model's answers with
a stronger one. Temperature 0 for maximum verdict stability, but judge
evals still flake occasionally; when one fails, read ``reasoning``
before blaming the code.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm

_JUDGE_PROMPT = """\
You are a strict quality judge for a chatbot's reply. Grade the reply
against the rubric, item by item. Judge only the assistant's reply —
never the user's message. Be literal about the rubric; do not invent
extra criteria.
"""


class Verdict(BaseModel):
    """The judge's structured grade for one reply."""

    passed: bool = Field(description="True only if EVERY rubric item is satisfied.")
    score: int = Field(ge=1, le=5, description="1 = terrible, 5 = flawless.")
    reasoning: str = Field(
        description="One short paragraph: which rubric items passed or failed, why."
    )


async def judge(rubric: str, transcript: str) -> Verdict:
    """Grade ``transcript`` (user + assistant turns) against ``rubric``."""

    model = get_llm(temperature=0.0, model=os.getenv("EVAL_JUDGE_MODEL") or None)
    return await model.with_structured_output(Verdict).ainvoke(
        [
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=f"Rubric:\n{rubric}\n\nTranscript:\n{transcript}"),
        ]
    )
