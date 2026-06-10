"""Embedding provider helpers."""

from __future__ import annotations

from inventory_pilot_ai.llm_provider import get_embedding_provider


def get_embeddings():
    provider = get_embedding_provider()
    if provider is None:
        raise RuntimeError("No embedding provider is configured.")
    return provider.embeddings()


def embedding_backend_name() -> str:
    provider = get_embedding_provider()
    if provider is None:
        raise RuntimeError("No embedding provider is configured.")
    return f"{provider.name}-embeddings"
