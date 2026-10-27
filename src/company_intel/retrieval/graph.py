"""Graph traversal retrieval for multi-hop questions.  [Phase 3]

Pattern: link entities in the question via the `entity_names` full-text index (or start
from top vector hits), expand 1-2 hops along typed relationships, then return the
relationship facts plus the source chunks they were extracted from.
"""

from __future__ import annotations

from company_intel.retrieval import RetrievedContext


def graph_search(question: str, max_hops: int = 2) -> list[RetrievedContext]:
    raise NotImplementedError("Phase 3")
