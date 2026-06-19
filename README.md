# RAG From Scratch

Ground-up Retrieval Augmented Generation system built with Python, LangChain, FastAPI, scikit-learn, and Chroma.

## What it does

- Indexes documents from `data/docs`
- Builds multi-representation chunks for raw text and compact summaries
- Performs semantic retrieval with query expansion and reranking
- Applies CRAG-style confidence checks and Self-RAG-style answer gating
- Exposes a REST API for query, health, and reindex operations

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn rag_from_scratch.api:app --reload
```

