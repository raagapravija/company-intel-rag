"""Rerank merged contexts with the Voyage reranker.  [Phase 1]"""

from __future__ import annotations

from company_intel.retrieval import RetrievedContext


def rerank(
    question: str, contexts: list[RetrievedContext], top_k: int = 8
) -> list[RetrievedContext]:
    raise NotImplementedError("Phase 1")
