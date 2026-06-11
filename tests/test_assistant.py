from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from on_device_assistant import RagAssistant, Settings


def _settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        sqlite_path=root / "assistant.sqlite3",
        chroma_path=root / "chroma",
        collection_name="test_collection",
        embedding_model="hashing",
        reranker_model="",
        chunk_size=220,
        chunk_overlap=30,
        default_top_k=3,
    )


class AssistantTests(unittest.TestCase):
    def test_ingest_and_query_returns_cited_local_answer(self) -> None:
        with TemporaryDirectory() as tmp:
            assistant = RagAssistant.from_settings(_settings(Path(tmp)))
            ingest = assistant.ingest_text(
                source_id="edge_latency",
                title="Edge Latency",
                text=(
                    "Retrieval latency is optimized with compact normalized embeddings, "
                    "small chunk windows, and bounded top-k vector search."
                ),
            )

            result = assistant.query("How is retrieval latency optimized?", top_k=2)

            self.assertGreaterEqual(ingest.chunks, 1)
            self.assertTrue(result.sources)
            self.assertEqual(result.sources[0].source_id, "edge_latency")
            self.assertIn("edge_latency", result.answer)
            self.assertGreaterEqual(result.retrieval_latency_ms, 0)

    def test_vector_store_persists_across_assistant_instances(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))
            first = RagAssistant.from_settings(settings)
            first.ingest_text(
                source_id="quantization",
                title="Quantization",
                text="Dynamic INT8 quantization reduces model size and memory bandwidth.",
            )

            second = RagAssistant.from_settings(settings)
            result = second.query("What reduces memory bandwidth?", top_k=1)

            self.assertTrue(result.sources)
            self.assertEqual(result.sources[0].source_id, "quantization")


if __name__ == "__main__":
    unittest.main()
