from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from on_device_assistant.config import Settings
from on_device_assistant.schemas import RetrievedChunk
from on_device_assistant.text import tokenize


class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class LexicalReranker:
    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        query_tokens = set(tokenize(question))
        rescored: list[RetrievedChunk] = []
        for candidate in candidates:
            doc_tokens = set(tokenize(candidate.text))
            overlap = len(query_tokens & doc_tokens)
            lexical_score = overlap / max(1, len(query_tokens))
            score = 0.75 * candidate.score + 0.25 * lexical_score
            rescored.append(replace(candidate, score=score))
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(question, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        rescored = [
            replace(candidate, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]


def build_reranker(settings: Settings) -> Reranker:
    if not settings.reranker_model:
        return LexicalReranker()
    try:
        return CrossEncoderReranker(settings.reranker_model, device=settings.device)
    except Exception:
        return LexicalReranker()
