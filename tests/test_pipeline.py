from __future__ import annotations

from pathlib import Path

from rag_from_scratch.config import Settings
from rag_from_scratch.pipeline import RagPipeline


def test_pipeline_answers_with_grounded_context(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text(
        "Retrieval Augmented Generation uses retrieval and generation together. CRAG checks quality.",
        encoding="utf-8",
    )

    settings = Settings(docs_path=docs, chroma_path=tmp_path / "chroma")
    pipeline = RagPipeline.build(settings)

    response = pipeline.query("What does CRAG do in RAG?")

    assert response.question == "What does CRAG do in RAG?"
    assert response.hits
    assert response.confidence >= 0.0
    assert response.route in {"answer", "broaden", "clarify"}


def test_health_reports_index_sizes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("RAG relies on retrieved documents.", encoding="utf-8")

    settings = Settings(docs_path=docs, chroma_path=tmp_path / "chroma")
    pipeline = RagPipeline.build(settings)

    health = pipeline.health()

    assert health.status == "ok"
    assert health.docs_indexed == 1
    assert health.chunks_indexed >= 1
