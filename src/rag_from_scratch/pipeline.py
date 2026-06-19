from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .config import Settings, get_settings
from .embeddings import TfidfEmbeddings
from .generation import generate_answer
from .loader import load_documents
from .models import HealthResponse, QueryResponse, ReindexResponse
from .retrieval import retrieve_question, to_hits
from .text_processing import chunk_text, normalize_text, summarize_chunk
from .vectorstore import build_vector_store


def _build_index_documents(documents: list[Document]) -> list[Document]:
    indexed_documents: list[Document] = []

    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        chunks = chunk_text(document.page_content)
        for position, chunk in enumerate(chunks):
            chunk_id = f"{Path(source).stem}-{position}"
            base_metadata = {
                "source": source,
                "chunk_id": chunk_id,
                "chunk_index": position,
            }
            indexed_documents.append(
                Document(
                    page_content=normalize_text(chunk),
                    metadata={**base_metadata, "representation": "raw"},
                )
            )

            summary = summarize_chunk(chunk)
            if summary:
                indexed_documents.append(
                    Document(
                        page_content=summary,
                        metadata={**base_metadata, "representation": "summary"},
                    )
                )

    return indexed_documents


@dataclass
class RagPipeline:
    settings: Settings
    embeddings: TfidfEmbeddings
    vectorstore: object
    source_documents: list[Document]
    indexed_documents: list[Document]

    @classmethod
    def build(cls, settings: Settings | None = None) -> "RagPipeline":
        settings = settings or get_settings()
        source_documents = load_documents(settings.docs_path)
        indexed_documents = _build_index_documents(source_documents)
        embeddings = TfidfEmbeddings()
        vectorstore = build_vector_store(indexed_documents, settings.chroma_path, embeddings)
        return cls(settings, embeddings, vectorstore, source_documents, indexed_documents)

    def reindex(self) -> ReindexResponse:
        rebuilt = self.build(self.settings)
        self.embeddings = rebuilt.embeddings
        self.vectorstore = rebuilt.vectorstore
        self.source_documents = rebuilt.source_documents
        self.indexed_documents = rebuilt.indexed_documents
        return ReindexResponse(status="ok", docs_indexed=len(self.source_documents), chunks_indexed=len(self.indexed_documents))

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            docs_indexed=len(self.source_documents),
            chunks_indexed=len(self.indexed_documents),
        )

    def query(self, question: str) -> QueryResponse:
        plan = retrieve_question(
            self.vectorstore,
            question,
            candidate_k=self.settings.candidate_k,
            top_k=self.settings.top_k,
            min_confidence=self.settings.min_confidence,
            max_query_variants=self.settings.max_query_variants,
        )
        documents = [item.document for item in plan.ranked_documents]
        answer, used_llm = generate_answer(question, documents, plan.route, plan.confidence, self.settings.openai_model)
        return QueryResponse(
            question=question,
            answer=answer,
            route=plan.route,
            confidence=round(plan.confidence, 4),
            used_llm=used_llm,
            query_variants=plan.query_variants,
            hits=to_hits(plan.ranked_documents),
            critique=plan.critique,
        )


def build_default_pipeline() -> RagPipeline:
    return RagPipeline.build()
