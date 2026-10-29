"""Rerank merged contexts. Local bge-reranker by default (free), Voyage optional."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from functools import lru_cache

from company_intel.config import Settings, get_settings
from company_intel.retrieval import RetrievedContext


@lru_cache
def _local_reranker(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, max_length=1024)


def rerank_text(ctx: RetrievedContext, settings: Settings) -> str:
    """The passage as the cross-encoder sees it. The header mirrors the one chunks were
    embedded with (ingestion/load.py chunk_header)."""
    text = " ".join(ctx.text.split()) if settings.rerank_compact_text else ctx.text
    m = ctx.metadata
    if not (settings.rerank_with_header and m.get("ticker")):
        return text
    header = (
        f"{m.get('company')} ({m['ticker']}) {m.get('form')} FY{m.get('fiscal_year')}, "
        f"Item {m.get('item')} {m.get('section')}"
    )
    return f"{header}\n{text}"


def _local_scores(pairs: list[tuple[str, str]], s: Settings) -> list[float]:
    """Cross-encoder scores, computed only for pairs not already cached on disk. The key
    is model + exact input, so changing either rescores. Fully cached calls never load
    the model (~2 GB)."""
    if not s.rerank_cache:
        return [float(v) for v in _local_reranker(s.rerank_model).predict(pairs, batch_size=16)]
    keys = [hashlib.sha256(f"{s.rerank_model}\0{q}\0{p}".encode()).hexdigest() for q, p in pairs]
    path = s.data_dir / "cache" / "rerank.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL)")
        marks = ",".join("?" * len(keys))
        found = dict(db.execute(f"SELECT key, score FROM scores WHERE key IN ({marks})", keys))
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            new = _local_reranker(s.rerank_model).predict([pairs[i] for i in todo], batch_size=16)
            rows = [(keys[i], float(v)) for i, v in zip(todo, new, strict=True)]
            db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?)", rows)
            db.commit()
            found.update(rows)
    return [found[k] for k in keys]


def rerank(
    question: str,
    contexts: list[RetrievedContext],
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedContext]:
    s = settings or get_settings()
    top_k = top_k or s.rerank_top_k
    if not contexts:
        return []
    if s.rerank_provider == "none":
        return contexts[:top_k]

    if s.rerank_provider == "local":
        scores = _local_scores([(question, rerank_text(c, s)) for c in contexts], s)
        ranked = sorted(zip(contexts, scores, strict=True), key=lambda p: p[1], reverse=True)
        out = []
        for ctx, score in ranked[:top_k]:
            ctx.metadata["retrieval_score"] = ctx.score
            ctx.score = float(score)
            out.append(ctx)
        return out

    import voyageai

    if not s.voyage_api_key:
        raise RuntimeError("RERANK_PROVIDER=voyage needs VOYAGE_API_KEY")
    client = voyageai.Client(api_key=s.voyage_api_key.get_secret_value())
    res = client.rerank(
        question, [rerank_text(c, s) for c in contexts], model=s.rerank_model, top_k=top_k
    )
    out = []
    for r in res.results:
        ctx = contexts[r.index]
        ctx.metadata["retrieval_score"] = ctx.score
        ctx.score = float(r.relevance_score)
        out.append(ctx)
    return out
