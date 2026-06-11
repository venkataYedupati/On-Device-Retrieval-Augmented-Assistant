from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from on_device_assistant.config import Settings
from on_device_assistant.text import tokenize


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashingEmbeddings(EmbeddingBackend):
    """Deterministic local fallback used for tests and no-download demos."""

    def __init__(self, dimensions: int = 384, normalize: bool = True) -> None:
        self.dimensions = dimensions
        self.normalize = normalize

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big", signed=False)
            index = value % self.dimensions
            sign = -1.0 if value & 1 else 1.0
            vector[index] += sign
        if self.normalize:
            norm = math.sqrt(sum(item * item for item in vector))
            if norm:
                vector = [item / norm for item in vector]
        return vector


class SentenceTransformerEmbeddings(EmbeddingBackend):
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedding_backend(settings: Settings) -> EmbeddingBackend:
    if settings.embedding_model.lower() in {"hash", "hashing", "test"}:
        return HashingEmbeddings(normalize=settings.normalize_embeddings)

    try:
        return SentenceTransformerEmbeddings(
            model_name=settings.embedding_model,
            device=settings.device,
            normalize=settings.normalize_embeddings,
        )
    except Exception:
        if settings.allow_hashing_fallback:
            return HashingEmbeddings(normalize=settings.normalize_embeddings)
        raise
