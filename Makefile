.PHONY: install install-dev lint test serve sample benchmark

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest -q

serve:
	uvicorn on_device_assistant.api:app --host 0.0.0.0 --port 8000 --reload

sample:
	odra ingest data/sample_docs --recursive
	odra ask "How does the assistant keep retrieval fast on constrained devices?"

benchmark:
	odra benchmark --eval-file data/eval/eval_queries.jsonl
