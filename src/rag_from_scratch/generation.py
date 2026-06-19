from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


def _extractive_answer(question: str, documents: list[Document]) -> str:
    if not documents:
        return "I could not find grounded evidence in the indexed documents."

    bullets = []
    for document in documents[:3]:
        snippet = document.page_content[:280].strip()
        bullets.append(f"- {snippet}")

    joined = "\n".join(bullets)
    return (
        f"Grounded answer for: {question}\n"
        f"Evidence:\n{joined}\n\n"
        "The answer is assembled from the most relevant retrieved passages."
    )


def generate_answer(question: str, documents: list[Document], route: str, confidence: float, model_name: str) -> tuple[str, bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if route == "clarify":
        return (
            "I do not have enough evidence in the indexed corpus to answer this confidently. "
            "Add more source documents or rephrase the question.",
            False,
        )

    if api_key:
        llm = ChatOpenAI(model=model_name, temperature=0)
        prompt = (
            "You are a grounded RAG assistant. Answer only with information supported by the provided context. "
            "If the evidence is weak, say so explicitly.\n\n"
            f"Question: {question}\n"
            f"Confidence: {confidence:.2f}\n"
            "Context:\n"
        )
        context = "\n\n".join(document.page_content for document in documents[:4])
        response = llm.invoke(prompt + context)
        return response.content, True

    return _extractive_answer(question, documents), False
