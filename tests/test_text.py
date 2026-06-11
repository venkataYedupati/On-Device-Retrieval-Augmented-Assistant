from __future__ import annotations

import unittest
from pathlib import Path

from on_device_assistant.schemas import RawDocument
from on_device_assistant.text import chunk_document, source_id_from_path, split_text


class TextTests(unittest.TestCase):
    def test_split_text_uses_overlap_without_empty_chunks(self) -> None:
        text = " ".join(f"token-{index}" for index in range(200))
        chunks = split_text(text, chunk_size=120, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_chunk_document_assigns_stable_chunk_ids(self) -> None:
        document = RawDocument(
            source_id="doc",
            title="Doc",
            text="Latency is optimized with compact embeddings and bounded vector search.",
        )
        chunks = chunk_document(document, chunk_size=80, chunk_overlap=10)
        self.assertTrue(chunks[0].chunk_id.startswith("chk_"))
        self.assertEqual(chunks[0].source_id, "doc")
        self.assertEqual(chunks[0].metadata["title"], "Doc")

    def test_source_id_from_path_is_readable(self) -> None:
        self.assertEqual(source_id_from_path(Path("Edge RAG Overview.md")), "edge_rag_overview")


if __name__ == "__main__":
    unittest.main()
