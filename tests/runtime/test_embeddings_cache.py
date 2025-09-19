"""Tests for the embeddings cache behaviour."""
from __future__ import annotations

import pytest

from python.runtime.config import EmbeddingsConfig
from python.runtime.embeddings import get_embeddings


@pytest.mark.asyncio
async def test_embeddings_cache_hits_and_misses(tmp_path):
    cfg = EmbeddingsConfig(
        provider="null",
        model="null-model",
        batch_size=2,
        cache_path=tmp_path / "embeddings.sqlite",
    )
    embeddings = get_embeddings(cfg)

    texts = ["alpha", "beta", "gamma"]
    initial_vectors = await embeddings.embed(texts)
    assert len(initial_vectors) == len(texts)

    original_embed = embeddings.provider.embed
    calls = {"count": 0}

    async def tracking_embed(batch):
        calls["count"] += 1
        return await original_embed(batch)

    embeddings.provider.embed = tracking_embed  # type: ignore[assignment]

    cached = await embeddings.embed(list(texts))
    assert cached == initial_vectors
    assert calls["count"] == 0  # no provider calls on cache hit

    await embeddings.embed(["alpha", "delta"])
    assert calls["count"] == 1  # new text forces a single provider call


@pytest.mark.asyncio
async def test_embeddings_batching_respects_config(tmp_path):
    cfg = EmbeddingsConfig(
        provider="null",
        model="null-model",
        batch_size=2,
        cache_path=tmp_path / "cache.sqlite",
    )
    embeddings = get_embeddings(cfg)

    batches = []
    original_embed = embeddings.provider.embed

    async def capture_batches(batch):
        batches.append(tuple(batch))
        return await original_embed(batch)

    embeddings.provider.embed = capture_batches  # type: ignore[assignment]

    await embeddings.embed(["a", "b", "c", "d", "e"])
    assert batches == [("a", "b"), ("c", "d"), ("e",)]
