from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .models import RetrievalHit
from .rerank import ScoredDocument, rerank_documents
from .text_processing import normalize_text, token_set


QUERY_EXPANSIONS = {
    "rag": ["retrieval augmented generation", "vector search", "grounded generation"],
    "langchain": ["runnables", "retrievers", "prompt templates"],
    "fastapi": ["api", "rest", "endpoint"],
    "vector store": ["chromadb", "similarity search", "embeddings"],
    "crag": ["corrective retrieval", "query routing", "confidence based retrieval"],
    "self-rag": ["self evaluation", "critic", "reflection"],
}


@dataclass(frozen=True)
class RetrievalPlan:
    route: str
    confidence: float
    critique: str
    query_variants: list[str]
    ranked_documents: list[ScoredDocument]


def build_query_variants(question: str, max_variants: int = 4) -> list[str]:
    question = normalize_text(question)
    variants = [question]
    lower_question = question.lower()

    for needle, expansions in QUERY_EXPANSIONS.items():
        if needle in lower_question:
            for expansion in expansions:
                expanded = f"{question} {expansion}"
                if expanded not in variants:
                    variants.append(expanded)

    keywords = [word for word in question.split() if len(word) > 3]
    if len(keywords) > 2:
        variants.append(" ".join(reversed(keywords)))

    return variants[:max_variants]


def _aggregate_candidates(
    store: Chroma,
    query_variants: list[str],
    candidate_k: int,
) -> list[tuple[Document, float]]:
    aggregated: dict[tuple[str, str], tuple[Document, float]] = {}

    for query in query_variants:
        pairs = store.similarity_search_with_relevance_scores(query, k=candidate_k)
        for document, score in pairs:
            key = (document.metadata.get("chunk_id", document.page_content[:64]), document.page_content)
            existing = aggregated.get(key)
            if existing is None or score > existing[1]:
                aggregated[key] = (document, float(score))

    return list(aggregated.values())


def retrieve_question(
    store: Chroma,
    question: str,
    candidate_k: int,
    top_k: int,
    min_confidence: float,
    max_query_variants: int,
) -> RetrievalPlan:
    query_variants = build_query_variants(question, max_query_variants=max_query_variants)
    candidates = _aggregate_candidates(store, query_variants, candidate_k)
    ranked = rerank_documents(question, candidates)

    if not ranked:
        return RetrievalPlan(
            route="clarify",
            confidence=0.0,
            critique="No indexed documents matched the question.",
            query_variants=query_variants,
            ranked_documents=[],
        )

    top_ranked = ranked[:top_k]
    top_score = top_ranked[0].final_score
    score_spread = top_ranked[0].final_score - top_ranked[-1].final_score if len(top_ranked) > 1 else top_score
    query_tokens = token_set(question)
    coverage = len(query_tokens & token_set(top_ranked[0].document.page_content)) / max(len(query_tokens), 1)
    confidence = max(0.0, min(1.0, 0.6 * top_score + 0.4 * coverage - 0.1 * score_spread))

    if confidence < min_confidence:
        route = "broaden"
        critique = "Retrieved evidence is present but weak; broadened routing is recommended."
    else:
        route = "answer"
        critique = "Retrieval confidence is high enough to answer directly."

    return RetrievalPlan(
        route=route,
        confidence=confidence,
        critique=critique,
        query_variants=query_variants,
        ranked_documents=top_ranked,
    )


def to_hits(ranked_documents: list[ScoredDocument]) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for item in ranked_documents:
        hits.append(
            RetrievalHit(
                chunk_id=item.document.metadata.get("chunk_id", "unknown"),
                source=item.document.metadata.get("source", "unknown"),
                representation=item.document.metadata.get("representation", "raw"),
                score=round(item.final_score, 4),
                text=item.document.page_content,
            )
        )
    return hits
