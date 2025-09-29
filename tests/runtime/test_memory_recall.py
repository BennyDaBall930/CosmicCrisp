"""Memory retrieval behaviour with the default stores."""
from __future__ import annotations

import pytest

from python.runtime.memory.fallback import MemoryWithFallback
from python.runtime.memory.null import NullMemory
from python.runtime.memory.schema import MemoryItem
from python.runtime.memory.sqlite_faiss import SQLiteFAISSMemory


class DummyEmbeddings:
    async def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "apple" in lowered else 0.0,
                    1.0 if "banana" in lowered else 0.0,
                    float(len(lowered)),
                ]
            )
        return vectors


@pytest.mark.asyncio
async def test_sqlite_memory_semantic_recall(tmp_path):
    store = SQLiteFAISSMemory(path=str(tmp_path / "memory.sqlite"), embeddings=DummyEmbeddings())
    await store.enter("session-1")
    await store.add(
        MemoryItem(kind="fact", text="Apples are crisp and sweet", tags=[], meta={})
    )
    await store.add(
        MemoryItem(kind="fact", text="Bananas are rich in potassium", tags=[], meta={})
    )

    results = await store.similar("banana recipes", k=1)
    assert len(results) == 1
    assert "Bananas" in results[0]["text"]


class FailingMemory(NullMemory):
    """Stub memory that always raises to trigger the fallback path."""

    async def similar(self, query: str, k: int = 5):  # type: ignore[override]
        raise RuntimeError("primary store unavailable")


@pytest.mark.asyncio
async def test_memory_with_fallback_uses_secondary(tmp_path):
    secondary = SQLiteFAISSMemory(path=str(tmp_path / "secondary.sqlite"), embeddings=DummyEmbeddings())
    await secondary.enter("session-2")
    await secondary.add(
        MemoryItem(kind="note", text="Apple pie techniques", tags=[], meta={})
    )

    memory = MemoryWithFallback(primary=FailingMemory(), secondary=secondary)
    recall = await memory.similar("apple", k=1)
    assert recall and "Apple" in recall[0]["text"]
