"""Embedding providers for OpenAI and local fallback."""

from __future__ import annotations

import math
import re

from langchain_core.embeddings import Embeddings

from core.llm_provider import get_embedding_provider


class LocalHashEmbeddings(Embeddings):
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
        return [value / norm for value in vector] if norm else vector


def get_embeddings() -> Embeddings:
    provider = get_embedding_provider()
    return provider.embeddings() if provider is not None else LocalHashEmbeddings()


def embedding_backend_name() -> str:
    provider = get_embedding_provider()
    return f"{provider.name}-embeddings" if provider is not None else "local-hash-embeddings"

