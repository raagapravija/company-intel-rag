"""Rerank merged contexts. Local bge-reranker by default (free), Voyage optional."""

from __future__ import annotations

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
        scores = _local_reranker(s.rerank_model).predict(
            [(question, rerank_text(c, s)) for c in contexts], batch_size=16
        )
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
