"""Settings loaded from environment variables / .env, plus the company list."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-please")
    neo4j_database: str = "neo4j"
    neo4j_readonly_user: str = "reader"
    neo4j_readonly_password: SecretStr = SecretStr("change-me-too")

    # Embeddings + reranking
    voyage_api_key: SecretStr | None = None
    embedding_model: str = "voyage-4-large"
    embedding_dimensions: int = Field(1024, gt=0, le=4096)
    rerank_model: str = "rerank-2.5"

    # LLM
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    eval_judge_model: str = "claude-opus-5-5"

    # SEC
    sec_user_agent: str = "company-intel-rag contact@example.com"

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    companies_file: Path = PROJECT_ROOT / "config" / "companies.yaml"


class CompanyRef(BaseModel):
    ticker: str
    name: str


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
