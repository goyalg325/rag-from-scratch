from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import HealthResponse, QueryRequest, QueryResponse, ReindexResponse
from .pipeline import RagPipeline, build_default_pipeline

app = FastAPI(title="RAG From Scratch", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    app.state.pipeline = build_default_pipeline()


def get_pipeline() -> RagPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")
    return pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_pipeline().health()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return get_pipeline().query(request.question)


@app.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    return get_pipeline().reindex()
