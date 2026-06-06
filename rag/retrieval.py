"""Local document ingestion and retrieval for the AskMamma knowledge base."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core import config
from db.database import get_connection, initialize_database, rows_to_dicts, utc_now


DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


def sanitize_filename(file_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file_name).name)
    if Path(safe).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use PDF, txt, markdown, or CSV.")
    return safe


def _read_file(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
    return [(path.read_text(encoding="utf-8"), None)]


def ingest_document(path: Path) -> dict[str, Any]:
    initialize_database()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {path.suffix}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (file_name, path, content_type, uploaded_at)
            VALUES (?, ?, ?, ?)
            """,
            (path.name, str(path), path.suffix.lower(), utc_now()),
        )
        document_id = cursor.lastrowid
        chunk_count = 0
        for text, page_number in _read_file(path):
            for chunk in splitter.split_text(text):
                if chunk.strip():
                    chunk_count += 1
                    connection.execute(
                        """
                        INSERT INTO document_chunks (document_id, chunk_id, page_number, text, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (document_id, chunk_count, page_number, chunk, "{}", utc_now()),
                    )
    return {"document_id": document_id, "file_name": path.name, "chunks": chunk_count}


def reindex_documents() -> dict[str, Any]:
    initialize_database()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("DELETE FROM document_chunks")
        connection.execute("DELETE FROM documents")
    indexed = []
    for path in DOCUMENTS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            indexed.append(ingest_document(path))
    return {"indexed": indexed, "count": len(indexed)}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score(query: str, text: str) -> float:
    query_counts = Counter(_tokens(query))
    text_counts = Counter(_tokens(text))
    if not query_counts or not text_counts:
        return 0.0
    dot = sum(query_counts[token] * text_counts[token] for token in query_counts)
    q_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    t_norm = math.sqrt(sum(value * value for value in text_counts.values()))
    return dot / (q_norm * t_norm) if q_norm and t_norm else 0.0


def document_search(query: str, limit: int = 5) -> dict[str, Any]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.chunk_id, c.page_number, c.text, d.file_name
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
    scored = []
    for row in rows_to_dicts(rows):
        score = _score(query, row["text"])
        if score > 0:
            row["score"] = round(score, 4)
            scored.append(row)
    scored.sort(key=lambda item: item["score"], reverse=True)
    results = scored[:limit]
    if not results:
        return {"found": False, "message": "No relevant document was found.", "results": []}
    return {"found": True, "results": results}


def save_uploaded_document(file_name: str, content: bytes) -> dict[str, Any]:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(file_name)
    path = config.UPLOAD_DIR / safe_name
    path.write_bytes(content)
    return ingest_document(path)
