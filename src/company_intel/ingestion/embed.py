"""Embed chunks with Voyage and write them to Neo4j.  [Phase 1]

Each Chunk node stores `embedding_model` so a model switch can be detected and
re-embedded instead of silently mixing vector spaces.
"""

from __future__ import annotations

from company_intel.ingestion.chunking import Chunk


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Batch-embed with input_type='document'."""
    raise NotImplementedError("Phase 1")


def embed_query(text: str) -> list[float]:
    """Embed a question with input_type='query'."""
    raise NotImplementedError("Phase 1")


def load_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """MERGE Company-[:FILED]->Filing-[:HAS_SECTION]->Section-[:HAS_CHUNK]->Chunk, plus NEXT."""
    raise NotImplementedError("Phase 1")
