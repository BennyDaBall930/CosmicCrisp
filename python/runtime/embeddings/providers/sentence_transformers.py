"""Local sentence-transformers embeddings provider.

This mirrors the Agent Zero setup by loading a Hugging Face sentence-transformer
model into memory and serving synchronous embeddings calls.
"""
from __future__ import annotations

import asyncio
import threading
from typing import List, Sequence

from ..provider import BaseEmbeddingsProvider


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}


def _load_model(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # type: ignore

    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
        return model


class SentenceTransformersEmbeddingsProvider(BaseEmbeddingsProvider):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        model_name = self.config.model or "sentence-transformers/all-MiniLM-L6-v2"
        model = _load_model(model_name)

        embeddings = await asyncio.to_thread(
            model.encode,
            list(texts),
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in embeddings]


__all__ = ["SentenceTransformersEmbeddingsProvider"]
