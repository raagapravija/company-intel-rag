"""Embedding models. Local BGE-M3 by default (free), Voyage optional."""

from __future__ import annotations

from functools import lru_cache

from company_intel.config import Settings, get_settings


@lru_cache
def _local_model(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    # BGE-M3 accepts 8192 tokens; our chunks are ~512, so cap for speed and memory.
    model.max_seq_length = 1024
    return model


@lru_cache
def _voyage_client(api_key: str):
    import voyageai

    return voyageai.Client(api_key=api_key)


def _embed(texts: list[str], input_type: str, settings: Settings) -> list[list[float]]:
    if not texts:
        return []
    if settings.embedding_provider == "local":
        vecs = _local_model(settings.embedding_model).encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return [v.tolist() for v in vecs]

    if not settings.voyage_api_key:
        raise RuntimeError("EMBEDDING_PROVIDER=voyage needs VOYAGE_API_KEY")
    client = _voyage_client(settings.voyage_api_key.get_secret_value())
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        res = client.embed(
            texts[i : i + 64],
            model=settings.embedding_model,
            input_type=input_type,
            output_dimension=settings.embedding_dimensions,
        )
        out.extend(res.embeddings)
    return out


def embed_documents(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    return _embed(texts, "document", settings or get_settings())


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    return _embed([text], "query", settings or get_settings())[0]
