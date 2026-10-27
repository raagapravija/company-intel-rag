"""Write filings, sections and embedded chunks to Neo4j (idempotent MERGEs)."""

from __future__ import annotations

from neo4j import Driver

from company_intel.db import run_read, run_write
from company_intel.ingestion.chunking import Chunk, Section
from company_intel.ingestion.edgar import FilingRef

UPSERT_FILING = """
MERGE (co:Company {cik: $cik})
  SET co.ticker = $ticker, co.name = $company_name, co.sector = $sector
MERGE (f:Filing {accession: $accession})
  SET f.form = $form, f.filed_date = date($filed_date), f.period_of_report = date($period),
      f.fiscal_year = $fiscal_year, f.url = $url
MERGE (co)-[:FILED]->(f)
"""

UPSERT_SECTIONS = """
MATCH (f:Filing {accession: $accession})
UNWIND $sections AS s
MERGE (sec:Section {id: s.id})
  SET sec.item = s.item, sec.title = s.title
MERGE (f)-[:HAS_SECTION]->(sec)
"""

# setNodeVectorProperty stores the embedding as a compact float32 vector.
UPSERT_CHUNKS = """
UNWIND $chunks AS c
MATCH (sec:Section {id: c.section_id})
MERGE (ch:Chunk {id: c.id})
  SET ch.text = c.text, ch.position = c.position, ch.embedding_model = $model
MERGE (sec)-[:HAS_CHUNK]->(ch)
WITH ch, c
CALL db.create.setNodeVectorProperty(ch, 'embedding', c.embedding)
"""

LINK_NEXT = """
MATCH (sec:Section {id: $section_id})-[:HAS_CHUNK]->(ch:Chunk)
WITH ch ORDER BY ch.position
WITH collect(ch) AS chunks
UNWIND range(0, size(chunks) - 2) AS i
WITH chunks[i] AS a, chunks[i + 1] AS b
MERGE (a)-[:NEXT]->(b)
"""

MARK_LOADED = """
MATCH (f:Filing {accession: $accession})
SET f.chunk_count = $chunk_count, f.embedding_model = $model, f.loaded_at = datetime()
"""

IS_LOADED = """
MATCH (f:Filing {accession: $accession})
RETURN f.chunk_count AS chunk_count, f.embedding_model AS model
"""


def chunk_header(ref: FilingRef, section: Section) -> str:
    """Prepended to each chunk's text before embedding, so the vector knows its source."""
    return (
        f"{ref.company_name} ({ref.ticker}) {ref.form} FY{ref.fiscal_year}, "
        f"Item {section.item} {section.title}"
    )


def is_loaded(driver: Driver, accession: str, model: str) -> bool:
    rows = run_read(driver, IS_LOADED, accession=accession)
    return bool(rows) and bool(rows[0]["chunk_count"]) and rows[0]["model"] == model


def load_filing(
    driver: Driver,
    ref: FilingRef,
    sector: str,
    sections: list[Section],
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
    batch_size: int = 200,
) -> None:
    run_write(
        driver,
        UPSERT_FILING,
        cik=ref.cik,
        ticker=ref.ticker,
        company_name=ref.company_name,
        sector=sector,
        accession=ref.accession,
        form=ref.form,
        filed_date=ref.filed_date.isoformat(),
        period=ref.period_of_report.isoformat(),
        fiscal_year=ref.fiscal_year,
        url=ref.primary_doc_url,
    )
    run_write(
        driver,
        UPSERT_SECTIONS,
        accession=ref.accession,
        sections=[{"id": s.id, "item": s.item, "title": s.title} for s in sections],
    )
    rows = [
        {
            "id": c.id,
            "section_id": c.section_id,
            "position": c.position,
            "text": c.text,
            "embedding": e,
        }
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    for i in range(0, len(rows), batch_size):
        run_write(driver, UPSERT_CHUNKS, chunks=rows[i : i + batch_size], model=model)
    for s in sections:
        run_write(driver, LINK_NEXT, section_id=s.id)
    run_write(driver, MARK_LOADED, accession=ref.accession, chunk_count=len(chunks), model=model)
