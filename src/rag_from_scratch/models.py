from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    source: str
    representation: str
    text: str
    metadata: dict[str, Any]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RetrievalHit(BaseModel):
    chunk_id: str
    source: str
    representation: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    confidence: float
    used_llm: bool
    query_variants: list[str]
    hits: list[RetrievalHit]
    critique: str


class HealthResponse(BaseModel):
    status: str
    docs_indexed: int
    chunks_indexed: int


class ReindexResponse(BaseModel):
    status: str
    docs_indexed: int
    chunks_indexed: int
