"""Graph schema: uniqueness constraints, vector index and full-text indexes.

Run with `make schema` (or `python -m company_intel.schema`). Every statement is
idempotent (IF NOT EXISTS), so it is safe to re-run.

See docs/schema.md for the full node/relationship model.
"""

from __future__ import annotations

from company_intel.config import Settings, get_settings

VECTOR_INDEX = "chunk_embedding"
CHUNK_FULLTEXT_INDEX = "chunk_text"
ENTITY_FULLTEXT_INDEX = "entity_names"

# (label, property) pairs that must be unique.
UNIQUE_KEYS: list[tuple[str, str]] = [
    ("Company", "cik"),
    ("Filing", "accession"),
    ("Section", "id"),
    ("Chunk", "id"),
    ("Person", "id"),
    ("Institution", "cik"),
    ("Risk", "id"),
    ("Product", "id"),
    ("NewsArticle", "url"),
    ("FinancialFact", "id"),
]

# Properties worth a range index for filtering in Cypher.
RANGE_INDEXES: list[tuple[str, str]] = [
    ("Company", "ticker"),
    ("Filing", "fiscal_year"),
    ("Section", "item"),
    ("NewsArticle", "published_at"),
    ("FinancialFact", "concept"),
]


def schema_statements(settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    stmts: list[str] = []

    for label, prop in UNIQUE_KEYS:
        name = f"{label.lower()}_{prop}_unique"
        stmts.append(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )

    for label, prop in RANGE_INDEXES:
        name = f"{label.lower()}_{prop}_idx"
        stmts.append(f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")

    stmts.append(
        f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS "
        f"FOR (c:Chunk) ON c.embedding "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {s.embedding_dimensions}, "
        f"`vector.similarity_function`: 'cosine'}}}}"
    )

    stmts.append(
        f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS "
        f"FOR (c:Chunk) ON EACH [c.text]"
    )

    # Lets exact names, tickers and aliases resolve to entities (entity linking at query time).
    stmts.append(
        f"CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS "
        f"FOR (n:Company|Person|Institution|Product|Risk) ON EACH [n.name]"
    )
    return stmts


def apply_schema(settings: Settings | None = None) -> None:
    from company_intel.db import driver_session, run_write

    with driver_session(settings) as driver:
        for stmt in schema_statements(settings):
            run_write(driver, stmt)
            print(f"ok  {stmt.split(' IF NOT EXISTS')[0]}")


if __name__ == "__main__":
    apply_schema()
