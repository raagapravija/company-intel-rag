"""Hybrid search: vector index + full-text index, fused with reciprocal rank fusion.  [Phase 1]"""

from __future__ import annotations

from company_intel.retrieval import RetrievedContext
from company_intel.schema import CHUNK_FULLTEXT_INDEX, VECTOR_INDEX

# Both queries return the chunk plus where it came from, for citations.
VECTOR_QUERY = f"""
CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k, $embedding) YIELD node AS chunk, score
MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(chunk)
WHERE $tickers IS NULL OR co.ticker IN $tickers
RETURN chunk.id AS id, chunk.text AS text, score,
       co.ticker AS ticker, f.accession AS accession, f.fiscal_year AS fiscal_year,
       s.item AS item, s.title AS section
"""

FULLTEXT_QUERY = f"""
CALL db.index.fulltext.queryNodes('{CHUNK_FULLTEXT_INDEX}', $query, {{limit: $k}})
YIELD node AS chunk, score
MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(chunk)
WHERE $tickers IS NULL OR co.ticker IN $tickers
RETURN chunk.id AS id, chunk.text AS text, score,
       co.ticker AS ticker, f.accession AS accession, f.fiscal_year AS fiscal_year,
       s.item AS item, s.title AS section
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists. Returns (id, fused_score), best first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(
    question: str, k: int = 30, tickers: list[str] | None = None
) -> list[RetrievedContext]:
    raise NotImplementedError("Phase 1")
