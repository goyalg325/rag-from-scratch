from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    if not folder.exists():
        return documents

    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "title": path.stem},
                    )
                )
    return documents
