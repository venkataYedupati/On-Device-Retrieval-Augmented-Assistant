from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from on_device_assistant.schemas import Chunk, RawDocument, StoreStats


class SQLiteDocumentStore:
    def __init__(self, path: Path, collection_name: str = "on_device_rag") -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES documents(source_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_source_id
                    ON chunks(source_id);

                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    retrieval_latency_ms REAL NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def upsert_document(self, document: RawDocument) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (source_id, title, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    title=excluded.title,
                    metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (document.source_id, document.title, json.dumps(document.metadata, sort_keys=True)),
            )

    def replace_chunks(self, source_id: str, chunks: Iterable[Chunk]) -> None:
        rows = [
            (
                chunk.chunk_id,
                chunk.source_id,
                chunk.chunk_index,
                chunk.text,
                json.dumps(chunk.metadata, sort_keys=True),
                len(chunk.text.split()),
            )
            for chunk in chunks
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            conn.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, source_id, chunk_index, text, metadata_json, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_chunk_ids_for_source(self, source_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE source_id = ? ORDER BY chunk_index",
                (source_id,),
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def stats(self) -> StoreStats:
        with self._connect() as conn:
            documents = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
            chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        return StoreStats(
            documents=int(documents),
            chunks=int(chunks),
            collection_name=self.collection_name,
        )

    def log_query_metric(
        self,
        question: str,
        top_k: int,
        retrieval_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_metrics
                    (question, top_k, retrieval_latency_ms, total_latency_ms)
                VALUES (?, ?, ?, ?)
                """,
                (question, top_k, retrieval_latency_ms, total_latency_ms),
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM query_metrics")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
