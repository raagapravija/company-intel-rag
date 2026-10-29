"""Settings loaded from environment variables / .env, plus the company list."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# OpenAI-compatible endpoints for free LLM options.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-please")
    neo4j_database: str = "neo4j"
    neo4j_readonly_user: str = "reader"
    neo4j_readonly_password: SecretStr = SecretStr("change-me-too")

    # Embeddings: "local" runs BGE-M3 on your machine (free); "voyage" uses the Voyage API
    embedding_provider: Literal["local", "voyage"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = Field(1024, gt=0, le=4096)
    embedding_batch_size: int = 16

    # Reranking: same choice
    rerank_provider: Literal["local", "voyage", "none"] = "local"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    voyage_api_key: SecretStr | None = None

    # LLM: any OpenAI-compatible endpoint (Groq free tier by default, or Ollama locally)
    llm_base_url: str = GROQ_BASE_URL
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "openai/gpt-oss-120b"
    # Reasoning models spend part of max_tokens thinking before they answer.
    llm_max_tokens: int = 2048
    # "low" keeps gpt-oss fast and within free-tier token limits; empty = don't send it.
    llm_reasoning_effort: str = "low"

    # Ragas judge: use a different model from the one answering, kept fixed across runs
    eval_judge_base_url: str = GEMINI_BASE_URL
    eval_judge_api_key: SecretStr | None = None  # falls back to llm_api_key
    eval_judge_model: str = "gemini-3.8-flash"

    # SEC
    sec_user_agent: str = "company-intel-rag contact@example.com"

    # Retrieval
    retrieval_k: int = 30
    rerank_top_k: int = 8
    # What the cross-encoder reads: chunk text with its source header (company, year,
    # section) and/or with whitespace collapsed (flattened tables put each cell on a line)
    rerank_with_header: bool = True
    rerank_compact_text: bool = False  # measured: no effect
    # Cache cross-encoder scores in data/cache/rerank.sqlite (keyed by model + exact input)
    rerank_cache: bool = True
    # Restrict search to the filing year a question names ("fiscal 2025 10-K")
    retrieval_year_filter: bool = True
    # With a company/year filter, score exactly over just those chunks instead of
    # post-filtering the global vector index (which can leave far fewer than k results)
    retrieval_exact_filtered: bool = True

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    companies_file: Path = PROJECT_ROOT / "config" / "companies.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


class CompanyRef(BaseModel):
    ticker: str
    name: str
    aliases: list[str] = []


class FilingScope(BaseModel):
    forms: list[str] = ["10-K"]
    years: list[int] = []


class CompanyConfig(BaseModel):
    sector: str
    filings: FilingScope = FilingScope()
    companies: list[CompanyRef]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_companies(path: Path | None = None) -> CompanyConfig:
    path = path or get_settings().companies_file
    with open(path) as f:
        return CompanyConfig.model_validate(yaml.safe_load(f))
