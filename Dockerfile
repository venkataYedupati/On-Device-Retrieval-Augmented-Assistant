FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ODRA_DATA_DIR=/app/.odra \
    ODRA_SQLITE_PATH=/app/.odra/assistant.sqlite3 \
    ODRA_CHROMA_PATH=/app/.odra/chroma

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "on_device_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
