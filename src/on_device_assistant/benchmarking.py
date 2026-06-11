from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from on_device_assistant.assistant import RagAssistant


def run_benchmark(
    assistant: RagAssistant,
    eval_file: Path | None = None,
    iterations: int = 5,
) -> dict[str, Any]:
    queries = _load_queries(eval_file)
    latencies: list[float] = []
    retrieval_latencies: list[float] = []
    ndcg_scores: list[float] = []

    for item in queries:
        for _ in range(iterations):
            started = time.perf_counter()
            result = assistant.query(item["question"], top_k=item.get("top_k", 5))
            latencies.append((time.perf_counter() - started) * 1000)
            retrieval_latencies.append(result.retrieval_latency_ms)
        if "relevant_source_ids" in item:
            ndcg_scores.append(
                ndcg_at_k(
                    [source.source_id in set(item["relevant_source_ids"]) for source in result.sources],
                    k=len(result.sources),
                )
            )

    return {
        "queries": len(queries),
        "iterations_per_query": iterations,
        "retrieval_latency_ms_p50": _percentile(retrieval_latencies, 50),
        "retrieval_latency_ms_p95": _percentile(retrieval_latencies, 95),
        "total_latency_ms_p50": _percentile(latencies, 50),
        "total_latency_ms_p95": _percentile(latencies, 95),
        "ndcg_at_k_mean": statistics.fmean(ndcg_scores) if ndcg_scores else None,
    }


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    gains = [1.0 if item else 0.0 for item in relevant[:k]]
    dcg = sum(gain / _log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / _log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _load_queries(eval_file: Path | None) -> list[dict[str, Any]]:
    if eval_file is None:
        return [{"question": "How does on-device retrieval stay fast?", "top_k": 5}]
    rows: list[dict[str, Any]] = []
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _log2(value: int) -> float:
    return math.log2(value)
