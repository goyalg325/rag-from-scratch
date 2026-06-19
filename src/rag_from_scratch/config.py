from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    docs_path: Path = Field(default=Path("data/docs"))
    chroma_path: Path = Field(default=Path(".chroma"))
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    rerank_weight_semantic: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_weight_lexical: float = Field(default=0.3, ge=0.0, le=1.0)
    max_query_variants: int = Field(default=4, ge=1, le=10)
    openai_model: str = Field(default="gpt-4o-mini")


def get_settings() -> Settings:
    return Settings()
