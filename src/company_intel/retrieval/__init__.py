"""Retrievers. Each returns a list of RetrievedContext so the reranker and Ragas see one shape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetrieverName = Literal["hybrid", "text2cypher", "graph"]


@dataclass
class RetrievedContext:
    text: str
    source: RetrieverName
    score: float = 0.0
    # Citation info: chunk id, filing accession, section item, company ticker, url, cypher...
    metadata: dict = field(default_factory=dict)
