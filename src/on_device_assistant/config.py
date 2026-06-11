from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local envs
    BaseSettings = object  # type: ignore[assignment,misc]
    SettingsConfigDict = dict  # type: ignore[assignment]

    def Field(default: Any, **_: Any) -> Any:  # type: ignore[override]
        return default


class Settings(BaseSettings):
    """Runtime configuration.

    The app is designed to run fully on device. Paths default to a local `.odra`
    folder so vector, metadata, and model artifacts can be persisted together.
    """

    if BaseSettings is not object:
        model_config = SettingsConfigDict(
            env_prefix="ODRA_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    data_dir: Path = Field(default=Path(".odra"))
    sqlite_path: Path = Field(default=Path(".odra/assistant.sqlite3"))
    chroma_path: Path = Field(default=Path(".odra/chroma"))
    collection_name: str = "on_device_rag"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model: str | None = None
    device: str = "cpu"
    allow_hashing_fallback: bool = True
    normalize_embeddings: bool = True

    chunk_size: int = 650
    chunk_overlap: int = 100
    default_top_k: int = 5

    def __init__(self, **values: Any) -> None:
        if BaseSettings is object:
            values = {**self._env_values(), **values}
            for field_name, default in self._defaults().items():
                setattr(self, field_name, values.get(field_name, default))
            self._coerce_paths()
        else:
            super().__init__(**values)

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "data_dir": Path(".odra"),
            "sqlite_path": Path(".odra/assistant.sqlite3"),
            "chroma_path": Path(".odra/chroma"),
            "collection_name": "on_device_rag",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generator_model": None,
            "device": "cpu",
            "allow_hashing_fallback": True,
            "normalize_embeddings": True,
            "chunk_size": 650,
            "chunk_overlap": 100,
            "default_top_k": 5,
        }

    @staticmethod
    def _env_values() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith("ODRA_"):
                continue
            name = key.removeprefix("ODRA_").lower()
            if value.lower() in {"true", "false"}:
                values[name] = value.lower() == "true"
            elif value.isdigit():
                values[name] = int(value)
            elif value == "":
                values[name] = None
            else:
                values[name] = value
        return values

    def _coerce_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.sqlite_path = Path(self.sqlite_path)
        self.chroma_path = Path(self.chroma_path)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
