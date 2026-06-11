from __future__ import annotations

from fastapi import FastAPI, File, UploadFile

from on_device_assistant.assistant import RagAssistant
from on_device_assistant.config import Settings
from on_device_assistant.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from on_device_assistant.text import normalize_text, stable_id


def create_app(settings: Settings | None = None) -> FastAPI:
    assistant = RagAssistant.from_settings(settings)
    app = FastAPI(
        title="On-Device Retrieval-Augmented Assistant",
        version="0.1.0",
        description="Local RAG service with Chroma, SQLite metadata, reranking, and ONNX tooling.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        stats = assistant.document_store.stats()
        return HealthResponse(
            status="ok",
            documents=stats.documents,
            chunks=stats.chunks,
            collection_name=stats.collection_name,
        )

    @app.post("/documents", response_model=DocumentIngestResponse)
    def ingest_document(request: DocumentIngestRequest) -> DocumentIngestResponse:
        result = assistant.ingest_text(
            text=request.text,
            source_id=request.source_id,
            title=request.title,
            metadata=request.metadata,
        )
        return DocumentIngestResponse(
            source_id=result.source_id,
            title=result.title,
            chunks=result.chunks,
        )

    @app.post("/documents/files", response_model=DocumentIngestResponse)
    async def ingest_file(file: UploadFile = File(...)) -> DocumentIngestResponse:
        content = await file.read()
        text = normalize_text(content.decode("utf-8", errors="ignore"))
        source_id = stable_id(f"{file.filename}:{text[:2048]}", prefix="doc_")
        result = assistant.ingest_text(
            text=text,
            source_id=source_id,
            title=file.filename or source_id,
            metadata={"filename": file.filename},
        )
        return DocumentIngestResponse(
            source_id=result.source_id,
            title=result.title,
            chunks=result.chunks,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = assistant.query(
            question=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            retrieval_latency_ms=result.retrieval_latency_ms,
            total_latency_ms=result.total_latency_ms,
            sources=[
                SourceResponse(
                    chunk_id=source.chunk_id,
                    source_id=source.source_id,
                    chunk_index=source.chunk_index,
                    score=source.score,
                    text=source.text,
                    metadata=source.metadata,
                )
                for source in result.sources
            ],
        )

    @app.delete("/documents")
    def reset() -> dict[str, str]:
        assistant.reset()
        return {"status": "cleared"}

    return app


app = create_app()
