from __future__ import annotations

import time
from pathlib import Path

from on_device_assistant.config import Settings
from on_device_assistant.embeddings import EmbeddingBackend, build_embedding_backend
from on_device_assistant.generator import AnswerGenerator, build_generator
from on_device_assistant.reranker import Reranker, build_reranker
from on_device_assistant.schemas import IngestResult, QueryResult, RawDocument
from on_device_assistant.storage import SQLiteDocumentStore
from on_device_assistant.text import chunk_document, load_documents, stable_id
from on_device_assistant.vector_store import VectorStore, build_vector_store


class RagAssistant:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingBackend,
        vector_store: VectorStore,
        document_store: SQLiteDocumentStore,
        reranker: Reranker,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.document_store = document_store
        self.reranker = reranker
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RagAssistant:
        settings = settings or Settings()
        settings.ensure_dirs()
        document_store = SQLiteDocumentStore(settings.sqlite_path, settings.collection_name)
        return cls(
            settings=settings,
            embeddings=build_embedding_backend(settings),
            vector_store=build_vector_store(
                settings.chroma_path,
                settings.collection_name,
                allow_memory_fallback=settings.allow_hashing_fallback,
            ),
            document_store=document_store,
            reranker=build_reranker(settings),
            generator=build_generator(settings),
        )

    def ingest_text(
        self,
        text: str,
        source_id: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResult:
        source_id = source_id or stable_id(text[:2048], prefix="doc_")
        document = RawDocument(
            source_id=source_id,
            title=title or source_id,
            text=text,
            metadata=metadata or {},
        )
        return self.ingest_document(document)

    def ingest_document(self, document: RawDocument) -> IngestResult:
        chunks = chunk_document(
            document,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        old_chunk_ids = self.document_store.get_chunk_ids_for_source(document.source_id)
        if old_chunk_ids:
            self.vector_store.delete(old_chunk_ids)

        embeddings = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        self.document_store.upsert_document(document)
        self.document_store.replace_chunks(document.source_id, chunks)
        self.vector_store.upsert(chunks, embeddings)
        return IngestResult(source_id=document.source_id, title=document.title, chunks=len(chunks))

    def ingest_paths(self, path: Path, recursive: bool = False) -> list[IngestResult]:
        return [self.ingest_document(document) for document in load_documents(path, recursive)]

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> QueryResult:
        total_started = time.perf_counter()
        top_k = top_k or self.settings.default_top_k
        candidate_limit = max(top_k, top_k * 4 if use_reranker else top_k)

        retrieval_started = time.perf_counter()
        query_embedding = self.embeddings.embed_query(question)
        candidates = self.vector_store.query(query_embedding, top_k=candidate_limit)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

        if use_reranker:
            sources = self.reranker.rerank(question, candidates, limit=top_k)
        else:
            sources = candidates[:top_k]

        answer = self.generator.generate(question, sources)
        total_latency_ms = (time.perf_counter() - total_started) * 1000
        self.document_store.log_query_metric(
            question=question,
            top_k=top_k,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
        )
        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    def reset(self) -> None:
        self.vector_store.reset()
        self.document_store.clear()
