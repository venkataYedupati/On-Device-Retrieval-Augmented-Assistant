from __future__ import annotations

import json
import math
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from on_device_assistant.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Path, collection_name: str) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "source_id": chunk.source_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        retrieved: list[RetrievedChunk] = []
        zipped_results = zip(ids, docs, metadatas, distances, strict=False)
        for chunk_id, text, metadata, distance in zipped_results:
            metadata = dict(metadata or {})
            source_id = str(metadata.pop("source_id", "unknown"))
            chunk_index = int(metadata.pop("chunk_index", 0))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    chunk_index=chunk_index,
                    text=text or "",
                    score=max(0.0, 1.0 - float(distance)),
                    metadata=metadata,
                )
            )
        return retrieved

    def reset(self) -> None:
        all_items = self.collection.get(include=[])
        ids = all_items.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)


class SQLiteVectorStore(VectorStore):
    """Small persistent fallback when Chroma is unavailable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        rows = [
            (
                chunk.chunk_id,
                chunk.source_id,
                chunk.chunk_index,
                chunk.text,
                json.dumps(chunk.metadata, sort_keys=True),
                json.dumps(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO vectors
                    (chunk_id, source_id, chunk_index, text, metadata_json, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    chunk_index=excluded.chunk_index,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    embedding_json=excluded.embedding_json
                """,
                rows,
            )

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM vectors WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, source_id, chunk_index, text, metadata_json, embedding_json
                FROM vectors
                """
            ).fetchall()

        scored: list[RetrievedChunk] = []
        for row in rows:
            candidate = json.loads(row["embedding_json"])
            scored.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    chunk_index=int(row["chunk_index"]),
                    text=row["text"],
                    score=_cosine_similarity(embedding, candidate),
                    metadata=json.loads(row["metadata_json"]),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vectors")

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items[chunk.chunk_id] = (chunk, embedding)

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._items.pop(chunk_id, None)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored: list[RetrievedChunk] = []
        for chunk, candidate in self._items.values():
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=_cosine_similarity(embedding, candidate),
                    metadata=dict(chunk.metadata),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def reset(self) -> None:
        self._items.clear()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_vector_store(
    path: Path,
    collection_name: str,
    allow_memory_fallback: bool,
) -> VectorStore:
    try:
        return ChromaVectorStore(path=path, collection_name=collection_name)
    except Exception:
        if allow_memory_fallback:
            fallback_path = path.parent / f"{collection_name}_vectors.sqlite3"
            return SQLiteVectorStore(fallback_path)
        raise
