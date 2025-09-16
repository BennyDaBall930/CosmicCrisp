"""Deterministic on-device embeddings placeholder.

Real MLX/PyTorch-backed embeddings will be wired in a later phase.  For now we
expose a deterministic hashing projection so downstream components can be
exercised without requiring GPU models.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence

from ..provider import BaseEmbeddingsProvider

_DIMENSION = 256


class LocalMLXEmbeddingsProvider(BaseEmbeddingsProvider):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Repeat digest to reach target dimension
            repeated = (digest * ((_DIMENSION // len(digest)) + 1))[:_DIMENSION]
            vector = [byte / 255.0 for byte in repeated]
            vectors.append(vector)
        return vectors


__all__ = ["LocalMLXEmbeddingsProvider"]
