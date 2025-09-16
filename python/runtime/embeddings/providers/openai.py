"""OpenAI/LiteLLM embeddings provider."""
from __future__ import annotations

import logging
from typing import List, Sequence

from ..provider import BaseEmbeddingsProvider

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    litellm = None  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsProvider(BaseEmbeddingsProvider):
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if litellm is None:
            raise RuntimeError(
                "litellm is required for the OpenAI embeddings provider. Install it or "
                "select a different provider."
            )
        response = await litellm.aembedding(model=self.model, input=list(texts))
        data = response.get("data") or []
        vectors: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("OpenAI embeddings API returned unexpected format")
            vectors.append([float(x) for x in embedding])
        if len(vectors) != len(texts):
            logger.warning(
                "OpenAI embeddings count mismatch: expected %d got %d", len(texts), len(vectors)
            )
        return vectors


__all__ = ["OpenAIEmbeddingsProvider"]
