"""Embeddings provider abstraction and registry."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence, TypeVar

from ..config import EmbeddingsConfig, RuntimeConfig

T = TypeVar("T")


class BaseEmbeddingsProvider(ABC):
    """Abstract provider responsible for producing raw embeddings vectors."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode ``texts`` into embedding vectors."""


ProviderFactory = Callable[[EmbeddingsConfig], BaseEmbeddingsProvider]


_PROVIDERS: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDERS[name] = factory


def get_provider(config: RuntimeConfig | EmbeddingsConfig) -> BaseEmbeddingsProvider:
    if isinstance(config, RuntimeConfig):
        embed_config = config.embeddings
    else:
        embed_config = config
    provider_name = embed_config.provider.lower()
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown embeddings provider '{provider_name}'")
    return _PROVIDERS[provider_name](embed_config)


async def gather_in_batches(
    texts: Sequence[str],
    batch_size: int,
    fn: Callable[[Sequence[str]], Awaitable[List[List[float]]]],
) -> List[List[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        batch_size = len(texts)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results: List[List[List[float]]] = await asyncio.gather(
        *(fn(batch) for batch in batches)
    )
    vectors: List[List[float]] = []
    for batch_vectors in results:
        vectors.extend(batch_vectors)
    return vectors


__all__ = ["BaseEmbeddingsProvider", "register_provider", "get_provider", "gather_in_batches"]
