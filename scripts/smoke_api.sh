#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -fsS "${BASE_URL}/health"
curl -fsS -X POST "${BASE_URL}/documents" \
  -H "Content-Type: application/json" \
  -d '{"source_id":"smoke","title":"Smoke Test","text":"On-device RAG uses local embeddings, vector retrieval, reranking, and extractive answers."}'
curl -fsS -X POST "${BASE_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does on-device RAG use?","top_k":3}'
