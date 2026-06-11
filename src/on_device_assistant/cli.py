from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from on_device_assistant.assistant import RagAssistant
from on_device_assistant.benchmarking import run_benchmark
from on_device_assistant.config import Settings
from on_device_assistant.quantization import (
    export_transformer_encoder_to_onnx,
    quantize_onnx_dynamic,
    quantize_torch_dynamic,
)

app = typer.Typer(help="On-device retrieval-augmented assistant")
console = Console()


def _assistant() -> RagAssistant:
    return RagAssistant.from_settings(Settings())


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest")],
    recursive: Annotated[bool, typer.Option("--recursive", "-r")] = False,
) -> None:
    assistant = _assistant()
    results = assistant.ingest_paths(path, recursive=recursive)
    table = Table("Source ID", "Title", "Chunks")
    for result in results:
        table.add_row(result.source_id, result.title, str(result.chunks))
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from local documents")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=25)] = 5,
    no_reranker: Annotated[bool, typer.Option("--no-reranker")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    result = _assistant().query(question, top_k=top_k, use_reranker=not no_reranker)
    if json_output:
        console.print_json(
            data={
                "question": result.question,
                "answer": result.answer,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "sources": [
                    {
                        "chunk_id": source.chunk_id,
                        "source_id": source.source_id,
                        "chunk_index": source.chunk_index,
                        "score": source.score,
                        "text": source.text,
                        "metadata": source.metadata,
                    }
                    for source in result.sources
                ],
            }
        )
        return

    console.print(f"[bold]Answer[/bold]\n{result.answer}\n")
    table = Table("Score", "Source", "Chunk", "Preview")
    for source in result.sources:
        table.add_row(
            f"{source.score:.3f}",
            source.source_id,
            str(source.chunk_index),
            source.text[:120].replace("\n", " "),
        )
    console.print(table)
    console.print(
        f"retrieval={result.retrieval_latency_ms:.1f}ms total={result.total_latency_ms:.1f}ms"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    import uvicorn

    uvicorn.run("on_device_assistant.api:app", host=host, port=port, reload=reload)


@app.command()
def benchmark(
    eval_file: Annotated[Path | None, typer.Option("--eval-file")] = None,
    iterations: Annotated[int, typer.Option("--iterations", min=1)] = 5,
) -> None:
    metrics = run_benchmark(_assistant(), eval_file=eval_file, iterations=iterations)
    console.print_json(data=metrics)


@app.command()
def reset() -> None:
    _assistant().reset()
    console.print("Local document and vector stores cleared.")


@app.command("export-onnx")
def export_onnx(
    model_name: Annotated[str, typer.Option("--model")] = "sentence-transformers/all-MiniLM-L6-v2",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/encoder.onnx"),
) -> None:
    path = export_transformer_encoder_to_onnx(model_name, output)
    console.print(str(path))


@app.command("quantize-onnx")
def quantize_onnx(
    input_path: Annotated[Path, typer.Argument(help="Input ONNX model path")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/encoder.int8.onnx"),
) -> None:
    path = quantize_onnx_dynamic(input_path, output)
    console.print(str(path))


@app.command("quantize-torch")
def quantize_torch(
    model_name: Annotated[str, typer.Option("--model")] = "sentence-transformers/all-MiniLM-L6-v2",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/encoder.int8.pt"),
) -> None:
    path = quantize_torch_dynamic(model_name, output)
    console.print(str(path))


@app.command()
def stats() -> None:
    assistant = _assistant()
    stats_value = assistant.document_store.stats()
    console.print(json.dumps(stats_value.__dict__, indent=2))
