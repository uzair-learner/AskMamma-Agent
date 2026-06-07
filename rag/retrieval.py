"""Embedding-backed local document ingestion and retrieval for AskMamma."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from core import config
from db.database import get_connection, initialize_database, rows_to_dicts, utc_now


DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings used to keep retrieval offline-friendly."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for index, token in enumerate(tokens):
            bucket = hash((token, index % 5)) % self.dimensions
            vector[bucket] += 1.0
            if index:
                bigram_bucket = hash((tokens[index - 1], token)) % self.dimensions
                vector[bigram_bucket] += 0.5
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


def embeddings_model() -> Embeddings:
    return LocalHashEmbeddings()


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


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


def _iter_document_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in [DOCUMENTS_DIR, config.UPLOAD_DIR]:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(path)
    return sorted(paths)


def _vector_store_files_exist() -> bool:
    return (config.VECTOR_STORE_PATH / "index.faiss").exists() and (config.VECTOR_STORE_PATH / "index.pkl").exists()


def _build_documents_from_database() -> list[Document]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT c.chunk_id, c.page_number, c.text, d.file_name, d.path
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY d.file_name, c.chunk_id
            """
        ).fetchall()
    documents = []
    for row in rows_to_dicts(rows):
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "file_name": row["file_name"],
                    "path": row["path"],
                    "chunk_id": row["chunk_id"],
                    "page_number": row["page_number"],
                },
            )
        )
    return documents


def rebuild_vector_store() -> dict[str, Any]:
    initialize_database()
    config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    documents = _build_documents_from_database()
    if not documents:
        for child in config.VECTOR_STORE_PATH.glob("*"):
            child.unlink()
        return {"indexed_chunks": 0, "path": str(config.VECTOR_STORE_PATH)}

    store = FAISS.from_documents(documents, embeddings_model())
    store.save_local(str(config.VECTOR_STORE_PATH))
    return {"indexed_chunks": len(documents), "path": str(config.VECTOR_STORE_PATH)}


def ingest_document(path: Path, rebuild_index: bool = True) -> dict[str, Any]:
    initialize_database()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {path.suffix}")

    splitter = _splitter()
    with get_connection() as connection:
        existing = connection.execute("SELECT id FROM documents WHERE path = ?", (str(path),)).fetchall()
        for row in existing:
            connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (row["id"],))
        connection.execute("DELETE FROM documents WHERE path = ?", (str(path),))
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
    if rebuild_index:
        rebuild_vector_store()
    return {"document_id": document_id, "file_name": path.name, "chunks": chunk_count}


def reindex_documents() -> dict[str, Any]:
    initialize_database()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("DELETE FROM document_chunks")
        connection.execute("DELETE FROM documents")
    indexed = []
    for path in _iter_document_paths():
        indexed.append(ingest_document(path, rebuild_index=False))
    vector = rebuild_vector_store()
    return {"indexed": indexed, "count": len(indexed), "vector_store": vector}


def _load_vector_store() -> FAISS | None:
    initialize_database()
    if not _vector_store_files_exist():
        if _build_documents_from_database():
            rebuild_vector_store()
        else:
            return None
    return FAISS.load_local(
        str(config.VECTOR_STORE_PATH),
        embeddings_model(),
        allow_dangerous_deserialization=True,
    )


def document_search(query: str, limit: int = 5) -> dict[str, Any]:
    initialize_database()
    query = query.strip()
    if not query:
        return {"found": False, "message": "Query cannot be empty.", "results": []}

    store = _load_vector_store()
    if store is None:
        return {"found": False, "message": "No indexed documents are available.", "results": []}

    matches = store.similarity_search_with_score(query, k=limit)
    results = []
    for doc, score in matches:
        results.append(
            {
                "file_name": doc.metadata.get("file_name"),
                "path": doc.metadata.get("path"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page_number": doc.metadata.get("page_number"),
                "text": doc.page_content,
                "score": round(float(score), 4),
            }
        )

    if not results:
        return {"found": False, "message": "No relevant document was found.", "results": []}
    return {"found": True, "results": results, "retriever": "faiss+local-hash-embeddings"}


def save_uploaded_document(file_name: str, content: bytes) -> dict[str, Any]:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(file_name)
    path = config.UPLOAD_DIR / safe_name
    path.write_bytes(content)
    return ingest_document(path, rebuild_index=True)
