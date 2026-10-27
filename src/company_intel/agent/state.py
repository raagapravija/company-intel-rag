"""Agent state. Saved at the end of each run, it doubles as the Ragas evaluation record."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from company_intel.retrieval import RetrievedContext, RetrieverName

QuestionType = Literal["lookup", "numeric", "relational", "summary"]


class AgentState(TypedDict, total=False):
    question: str
    question_type: QuestionType
    retrievers: list[RetrieverName]  # chosen by the router; can be several
    contexts: Annotated[list[RetrievedContext], operator.add]  # accumulates across retries
    reranked: list[RetrievedContext]
    answer: str
    citations: list[dict]
    grounded: bool
    missing_info: str  # verifier's note on what to look for on retry
    attempts: int
