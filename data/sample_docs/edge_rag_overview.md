# Edge RAG Overview

The assistant indexes local documents with a compact sentence-transformer encoder and stores vectors
in a persistent Chroma collection. Metadata, source titles, chunk offsets, and query timings are kept
in SQLite so the device can restart without losing corpus state.

Retrieval is optimized for low latency by using normalized embeddings, small chunk windows, and a
bounded top-k search. On a constrained device, the retriever should avoid scanning full source files
after ingestion and should return candidate chunks directly from the vector index.

Memory usage is reduced by using compact embedding models, batching document encoding, and exporting
transformer backbones to ONNX for runtime portability. Dynamic INT8 quantization compresses linear
weights and is useful when CPU inference has to run without a GPU.

The reranking layer can run a cross-encoder over the top candidates to improve answer relevance.
When the cross-encoder is unavailable, a lexical fallback blends token overlap with vector similarity.

The default answer generator is extractive. It selects the most relevant sentences from retrieved
chunks and cites the source IDs, which keeps inference predictable on offline hardware.
