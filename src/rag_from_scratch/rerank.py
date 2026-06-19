from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from .text_processing import token_set


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    semantic_score: float
    lexical_score: float
    final_score: float


def rerank_documents(query: str, candidates: list[tuple[Document, float]]) -> list[ScoredDocument]:
    query_tokens = token_set(query)
    ranked: list[ScoredDocument] = []

    for document, semantic_score in candidates:
        doc_tokens = token_set(document.page_content)
        overlap = len(query_tokens & doc_tokens)
        lexical_score = overlap / max(len(query_tokens), 1)
        final_score = 0.7 * semantic_score + 0.3 * lexical_score
        ranked.append(
            ScoredDocument(
                document=document,
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                final_score=final_score,
            )
        )

    ranked.sort(key=lambda item: item.final_score, reverse=True)
    return ranked
