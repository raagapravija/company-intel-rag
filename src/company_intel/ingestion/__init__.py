"""Ingestion pipeline: sources -> chunks / entities / structured facts -> Neo4j.

Phase 1: edgar.py, chunking.py, embed.py  (plain vector RAG)
Phase 2: structured.py                    (XBRL financials, 13F holdings)
Phase 3: extract.py                       (LLM entity + relationship extraction)
Phase 4: news.py                          (news articles)
"""
