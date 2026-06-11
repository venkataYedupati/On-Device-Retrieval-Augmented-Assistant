from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[override]
        return default


JsonDict = dict[str, Any]


@dataclass(slots=True)
class RawDocument:
    source_id: str
    title: str
    text: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class IngestResult:
    source_id: str
    title: str
    chunks: int


@dataclass(slots=True)
class QueryResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    retrieval_latency_ms: float
    total_latency_ms: float


@dataclass(slots=True)
class StoreStats:
    documents: int
    chunks: int
    collection_name: str


class DocumentIngestRequest(BaseModel):
    source_id: str | None = None
    title: str | None = None
    text: str = Field(min_length=1)
    metadata: JsonDict = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    source_id: str
    title: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=25)
    use_reranker: bool = True


class SourceResponse(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    score: float
    text: str
    metadata: JsonDict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    retrieval_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    collection_name: str
