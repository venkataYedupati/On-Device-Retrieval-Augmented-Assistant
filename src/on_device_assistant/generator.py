from __future__ import annotations

import re
from typing import Protocol

from on_device_assistant.config import Settings
from on_device_assistant.schemas import RetrievedChunk
from on_device_assistant.text import tokenize

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class AnswerGenerator(Protocol):
    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class ExtractiveAnswerGenerator:
    """Fast CPU-friendly answer synthesis from retrieved context."""

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return "I could not find relevant local context for that question."

        query_tokens = set(tokenize(question))
        scored_sentences: list[tuple[float, int, str, RetrievedChunk]] = []
        for source_index, source in enumerate(sources):
            for sentence in SENTENCE_RE.split(source.text):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                sentence_tokens = set(tokenize(sentence))
                overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
                scored_sentences.append(
                    (overlap + source.score * 0.05, source_index, sentence, source)
                )

        if not scored_sentences:
            best = sources[0]
            return f"{best.text[:600].strip()} [{best.source_id}:{best.chunk_index}]"

        selected = sorted(scored_sentences, key=lambda item: item[0], reverse=True)[:3]
        selected = sorted(selected, key=lambda item: item[1])
        answer_parts = [
            f"{sentence} [{source.source_id}:{source.chunk_index}]"
            for _, _, sentence, source in selected
        ]
        return " ".join(answer_parts)


class TransformersAnswerGenerator:
    """Optional local model generation for users who provide a seq2seq model."""

    def __init__(self, model_name: str, device: int = -1) -> None:
        from transformers import pipeline

        self.pipeline = pipeline("text2text-generation", model=model_name, device=device)

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return "I could not find relevant local context for that question."
        context = "\n\n".join(source.text for source in sources[:4])
        prompt = (
            "Answer using only this context.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        output = self.pipeline(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
        return str(output).strip()


def build_generator(settings: Settings) -> AnswerGenerator:
    if not settings.generator_model:
        return ExtractiveAnswerGenerator()
    try:
        device_index = 0 if settings.device.startswith("cuda") else -1
        return TransformersAnswerGenerator(settings.generator_model, device=device_index)
    except Exception:
        return ExtractiveAnswerGenerator()
