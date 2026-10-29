"""Hybrid search: vector index + full-text index, fused with reciprocal rank fusion."""

from __future__ import annotations

import re
from typing import Literal

from company_intel.config import CompanyConfig, Settings, get_settings, load_companies
from company_intel.retrieval import RetrievedContext
from company_intel.schema import CHUNK_FULLTEXT_INDEX, VECTOR_INDEX

_RETURN = """
MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(chunk)
WHERE ($tickers IS NULL OR co.ticker IN $tickers)
  AND ($years IS NULL OR f.fiscal_year IN $years)
RETURN chunk.id AS id, chunk.text AS text, score,
       co.ticker AS ticker, co.name AS company, f.form AS form, f.accession AS accession,
       f.fiscal_year AS fiscal_year, f.url AS url, s.item AS item, s.title AS section
ORDER BY score DESC
LIMIT $k
"""

# fetch_k > k so ticker/year filters applied after the index lookup still leave k results.
VECTOR_QUERY = f"""
CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $fetch_k, $embedding)
YIELD node AS chunk, score
{_RETURN}
"""

# Exact cosine over the filtered chunks (one company has ~250), so k results are guaranteed.
EXACT_VECTOR_QUERY = f"""
MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(chunk)
WHERE ($tickers IS NULL OR co.ticker IN $tickers)
  AND ($years IS NULL OR f.fiscal_year IN $years)
WITH chunk, vector.similarity.cosine(chunk.embedding, $embedding) AS score
ORDER BY score DESC
LIMIT $k
{_RETURN}
"""

FULLTEXT_QUERY = f"""
CALL db.index.fulltext.queryNodes('{CHUNK_FULLTEXT_INDEX}', $query, {{limit: $fetch_k}})
YIELD node AS chunk, score
{_RETURN}
"""

SearchMode = Literal["vector", "fulltext", "hybrid"]

FULLTEXT_FILTERED_FETCH = 2000

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


def detect_tickers(question: str, companies: CompanyConfig | None = None) -> list[str] | None:
    """Tickers of companies named in the question (by ticker or alias), or None if none."""
    cfg = companies or load_companies()
    q = question.lower()
    found = []
    for c in cfg.companies:
        names = [c.ticker.lower(), *[a.lower() for a in c.aliases]]
        if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names):
            found.append(c.ticker)
    return found or None


_FILING_YEAR = re.compile(r"\bfiscal (?:year )?(20\d{2}) (?:form )?10-K\b", re.IGNORECASE)
_FISCAL_YEAR = re.compile(r"\b(?:fiscal (?:year )?|FY\s?)(20\d{2})\b", re.IGNORECASE)


def detect_fiscal_years(question: str) -> list[int] | None:
    """Filing years a question asks about, or None. "In its fiscal 2025 10-K ... fiscal 2024"
    names the 2025 filing (which also reports 2024), so an explicit filing year wins."""
    years = _FILING_YEAR.findall(question) or _FISCAL_YEAR.findall(question)
    return sorted({int(y) for y in years}) or None


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
    question: str,
    k: int | None = None,
    tickers: list[str] | None = None,
    settings: Settings | None = None,
    mode: SearchMode = "hybrid",
    years: list[int] | None = None,
) -> list[RetrievedContext]:
    """Top-k chunks for a question. mode="vector" is the plain-vector baseline."""
    from company_intel.db import make_driver, run_read
    from company_intel.embeddings import embed_query

    s = settings or get_settings()
    k = k or s.retrieval_k
    if tickers is None:
        tickers = detect_tickers(question)
    if years is None and s.retrieval_year_filter:
        years = detect_fiscal_years(question)
    exact = bool(tickers or years) and s.retrieval_exact_filtered
    fetch_k = k * (5 if tickers else 1) * (2 if years else 1)

    driver = make_driver(s)
    try:
        vec_rows = [] if mode == "fulltext" else run_read(
            driver, EXACT_VECTOR_QUERY if exact else VECTOR_QUERY,
            embedding=embed_query(question, s), k=k, fetch_k=fetch_k, tickers=tickers, years=years,
        )
        ft_rows = [] if mode == "vector" else run_read(
            # Lowercase so words like AND/OR/NOT in a question aren't read as Lucene operators.
            # Full-text lookups are cheap, so read deep enough that the filter leaves k.
            driver, FULLTEXT_QUERY, query=lucene_escape(question.lower()), k=k,
            fetch_k=FULLTEXT_FILTERED_FETCH if exact else fetch_k, tickers=tickers, years=years,
        )
    finally:
        driver.close()

    rows = {r["id"]: r for r in vec_rows + ft_rows}
    # RRF over a single list keeps its order, so vector/fulltext modes pass through unchanged.
    fused = reciprocal_rank_fusion([[r["id"] for r in rs] for rs in (vec_rows, ft_rows) if rs])
    out = []
    for chunk_id, score in fused[:k]:
        r = rows[chunk_id]
        meta = {key: r[key] for key in r if key not in ("text", "score")}
        meta["chunk_id"] = meta.pop("id")
        out.append(RetrievedContext(text=r["text"], source="hybrid", score=score, metadata=meta))
    return out
