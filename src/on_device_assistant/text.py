from __future__ import annotations

import hashlib
import re
from pathlib import Path

from on_device_assistant.schemas import Chunk, RawDocument

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def stable_id(value: str, prefix: str = "") -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}" if prefix else digest


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_document(path: Path) -> RawDocument:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in {".html", ".htm"}:
        text = _read_html(path)
    else:
        text = path.read_text(encoding="utf-8")

    normalized = normalize_text(text)
    if not normalized:
        raise ValueError(f"No readable text found in {path}")

    return RawDocument(
        source_id=source_id_from_path(path),
        title=path.stem.replace("-", " ").replace("_", " ").strip() or path.name,
        text=normalized,
        metadata={"path": str(path), "extension": suffix},
    )


def source_id_from_path(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_").lower()
    return slug or stable_id(str(path), prefix="doc_")


def load_documents(path: Path, recursive: bool = False) -> list[RawDocument]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [read_document(path)]
    if not path.is_dir():
        raise FileNotFoundError(path)

    iterator = path.rglob("*") if recursive else path.glob("*")
    documents: list[RawDocument] = []
    for candidate in sorted(iterator):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(read_document(candidate))
    return documents


def chunk_document(document: RawDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    texts = split_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        chunk_id = stable_id(f"{document.source_id}:{index}:{text}", prefix="chk_")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=document.source_id,
                chunk_index=index,
                text=text,
                metadata={**document.metadata, "title": document.title},
            )
        )
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [
            normalize_text(chunk)
            for chunk in splitter.split_text(text)
            if normalize_text(chunk)
        ]
    except ModuleNotFoundError:
        return _fallback_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = normalize_text(text[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap) if chunk_overlap else start + stride
    return chunks


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pypdf to ingest PDF files.") from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return path.read_text(encoding="utf-8")

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")
