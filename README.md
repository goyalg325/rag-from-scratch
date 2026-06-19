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

## Query the API

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType application/json -Body '{"question":"What is RAG?"}'
```

## Environment variables

- `RAG_DOCS_PATH`: folder containing source documents, default `data/docs`
- `RAG_CHROMA_PATH`: local Chroma persistence folder, default `.chroma`
- `RAG_TOP_K`: final retrieved chunks, default `4`
- `RAG_CANDIDATE_K`: initial candidate count, default `10`
- `RAG_MIN_CONFIDENCE`: minimum confidence to answer directly, default `0.45`
- `OPENAI_API_KEY`: optional, enables OpenAI generation when provided

## GitHub upload commands

```powershell
git init
git add .
git commit -m "Initial RAG from scratch project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
