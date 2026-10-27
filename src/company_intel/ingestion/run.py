"""Phase 1 ingestion: EDGAR -> sections -> chunks -> embeddings -> Neo4j.

    uv run python -m company_intel.ingestion.run                 # everything in companies.yaml
    uv run python -m company_intel.ingestion.run --tickers NVDA AMD
    uv run python -m company_intel.ingestion.run --download-only # just fetch filings

Safe to re-run: filings already loaded with the current embedding model are skipped.
"""

from __future__ import annotations

import argparse
import time

from company_intel.config import get_settings, load_companies
from company_intel.ingestion.chunking import chunk_section, html_to_text, split_sections
from company_intel.ingestion.edgar import download_filing, list_filings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="*", help="subset of tickers from companies.yaml")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="reload filings already in Neo4j")
    args = parser.parse_args()

    s = get_settings()
    cfg = load_companies()
    companies = cfg.companies
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        companies = [c for c in companies if c.ticker in wanted]

    driver = None
    if not args.download_only:
        from company_intel.db import make_driver

        driver = make_driver(s)
        driver.verify_connectivity()

    try:
        for company in companies:
            refs = list_filings(company.ticker, cfg.filings.forms, cfg.filings.years)
            if not refs:
                print(f"{company.ticker}: no {cfg.filings.forms} filings for {cfg.filings.years}")
            for ref in refs:
                label = f"{ref.ticker} {ref.form} FY{ref.fiscal_year} ({ref.accession})"
                html = download_filing(ref)
                if args.download_only:
                    print(f"downloaded {label}")
                    continue

                from company_intel.ingestion.load import chunk_header, is_loaded, load_filing

                if not args.force and is_loaded(driver, ref.accession, s.embedding_model):
                    print(f"skip      {label} (already loaded)")
                    continue

                t0 = time.time()
                sections = split_sections(html_to_text(html), ref.accession)
                found = {sec.item for sec in sections}
                if "1A" not in found or "7" not in found:
                    print(f"warning   {label}: sections found {sorted(found)}; check parsing")
                chunks = [c for sec in sections for c in chunk_section(sec)]
                headers = {sec.id: chunk_header(ref, sec) for sec in sections}

                from company_intel.embeddings import embed_documents

                texts = [f"{headers[c.section_id]}\n\n{c.text}" for c in chunks]
                embeddings = embed_documents(texts)
                load_filing(
                    driver, ref, cfg.sector, sections, chunks, embeddings, s.embedding_model
                )
                print(
                    f"loaded    {label}: {len(sections)} sections, {len(chunks)} chunks "
                    f"in {time.time() - t0:.0f}s"
                )
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
