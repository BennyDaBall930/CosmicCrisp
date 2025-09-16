"""Null embeddings provider for tests."""
from __future__ import annotations

from typing import List, Sequence

from ..provider import BaseEmbeddingsProvider


class NullEmbeddingsProvider(BaseEmbeddingsProvider):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [[0.0] * 8 for _ in texts]


__all__ = ["NullEmbeddingsProvider"]
