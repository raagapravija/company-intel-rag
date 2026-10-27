"""Ingestion pipeline: sources -> chunks / entities / structured facts -> Neo4j.

Phase 1: edgar.py, chunking.py, load.py, run.py  (filings -> embedded chunks)
Phase 2: structured.py                           (XBRL financials, 13F holdings)
Phase 3: extract.py                              (LLM entity + relationship extraction)
Phase 4: news.py                                 (news articles)
"""
