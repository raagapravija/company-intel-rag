"""Ingest news articles (GDELT or RSS) and link them to companies.  [Phase 4]

(:NewsArticle {url, title, published_at, source})-[:HAS_CHUNK]->(:Chunk)
(:NewsArticle)-[:MENTIONS]->(:Company)
"""

from __future__ import annotations


def ingest_news(tickers: list[str], since_days: int = 30) -> int:
    raise NotImplementedError("Phase 4")
