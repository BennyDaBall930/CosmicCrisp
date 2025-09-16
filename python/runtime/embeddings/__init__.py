"""Embeddings service facade."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ..config import EmbeddingsConfig, RuntimeConfig, load_runtime_config
from .cache import SQLiteEmbeddingsCache
from .provider import BaseEmbeddingsProvider, gather_in_batches, get_provider, register_provider
from .providers.local_mlx import LocalMLXEmbeddingsProvider
from .providers.null import NullEmbeddingsProvider
from .providers.openai import OpenAIEmbeddingsProvider

logger = logging.getLogger(__name__)

# Register built-in providers
register_provider("openai", lambda cfg: OpenAIEmbeddingsProvider(cfg))
register_provider("local_mlx", lambda cfg: LocalMLXEmbeddingsProvider(cfg))
register_provider("null", lambda cfg: NullEmbeddingsProvider(cfg))


@dataclass
class Embeddings:
    provider: BaseEmbeddingsProvider
    cache: SQLiteEmbeddingsCache
    batch_size: int

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model_name = self.provider.model
        keys = [self.cache.make_cache_key(model_name, text) for text in texts]
        cached = await self.cache.get_many(keys)
        results: List[List[float]] = [[] for _ in texts]
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for idx, key in enumerate(keys):
            vector = cached.get(key)
            if vector is None:
                missing_indices.append(idx)
                missing_texts.append(texts[idx])
            else:
                results[idx] = vector
        if missing_texts:
            logger.debug("Embedding cache miss for %d texts", len(missing_texts))

            async def _embed_batch(batch: Sequence[str]) -> List[List[float]]:
                return await self.provider.embed(batch)

            vectors = await gather_in_batches(missing_texts, self.batch_size, _embed_batch)
            if len(vectors) != len(missing_indices):
                raise RuntimeError("Embeddings provider returned unexpected vector count")
            to_cache = {}
            for idx, vector in zip(missing_indices, vectors):
                key = keys[idx]
                results[idx] = vector
                to_cache[key] = vector
            await self.cache.set_many(to_cache)
        return results


def _resolve_config(config: RuntimeConfig | EmbeddingsConfig | None) -> EmbeddingsConfig:
    if config is None:
        runtime = load_runtime_config()
        return runtime.embeddings
    if isinstance(config, RuntimeConfig):
        return config.embeddings
    return config


def get_embeddings(config: RuntimeConfig | EmbeddingsConfig | None = None) -> Embeddings:
    embed_config = _resolve_config(config)
    cache = SQLiteEmbeddingsCache(Path(embed_config.cache_path))
    provider = get_provider(embed_config)
    return Embeddings(provider=provider, cache=cache, batch_size=embed_config.batch_size)


__all__ = ["Embeddings", "get_embeddings"]
