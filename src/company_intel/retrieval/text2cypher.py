"""Text2Cypher over structured data (financials, holdings, filings).  [Phase 2]

Guardrails:
- Runs with the read-only Neo4j user (db.make_driver(readonly=True))
- Rejects any generated query containing write clauses before executing
- Few-shot examples come from eval questions with reference Cypher
"""

from __future__ import annotations

import re

from company_intel.retrieval import RetrievedContext

WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b"
    r"|\bCALL\s+(apoc\.(create|merge|refactor|periodic)|dbms\.|db\.create)",
    re.IGNORECASE,
)


def is_read_only(cypher: str) -> bool:
    """Cheap pre-check; the read-only DB user is the real safety net."""
    return WRITE_CLAUSES.search(cypher) is None


def text2cypher(question: str) -> list[RetrievedContext]:
    raise NotImplementedError("Phase 2")
