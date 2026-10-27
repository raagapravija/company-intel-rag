"""LLM entity + relationship extraction from chunks.  [Phase 3]

Extract into a fixed schema (see docs/schema.md) rather than free-form, so the graph
stays queryable: Company, Person, Product, Risk and the relationships between them.
Every extracted edge keeps `source_chunk` so answers can cite where a fact came from.
Entity resolution (e.g. "TSMC" == "Taiwan Semiconductor Manufacturing") happens here.
"""

from __future__ import annotations

ENTITY_TYPES = ["Company", "Person", "Product", "Risk"]
RELATIONSHIP_TYPES = [
    "SUPPLIED_BY",  # (Company)->(Company)
    "CUSTOMER_OF",  # (Company)->(Company)
    "COMPETES_WITH",  # (Company)->(Company)
    "EXECUTIVE_OF",  # (Person)->(Company)
    "DIRECTOR_OF",  # (Person)->(Company)
    "SELLS",  # (Company)->(Product)
    "FACES",  # (Company)->(Risk)
]


def extract_from_chunks(chunk_ids: list[str]) -> int:
    raise NotImplementedError("Phase 3")
