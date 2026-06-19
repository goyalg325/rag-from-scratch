from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import TfidfEmbeddings


def build_vector_store(
    documents: list[Document],
    persist_directory: Path,
    embeddings: TfidfEmbeddings,
) -> Chroma:
    texts = [document.page_content for document in documents]
    embeddings.fit(texts)
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name="rag_from_scratch",
    )
