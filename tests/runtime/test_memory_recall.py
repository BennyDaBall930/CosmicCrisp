"""Memory retrieval behaviour with local and mem0-backed stores."""
from __future__ import annotations

import pytest

from python.runtime.memory.schema import MemoryItem
from python.runtime.memory.sqlite_faiss import SQLiteFAISSMemory
from python.runtime.memory.stores import mem0_adapter as mem0_module


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


@pytest.mark.asyncio
async def test_mem0_adapter_uses_fallback(monkeypatch, tmp_path):
    fallback = SQLiteFAISSMemory(path=str(tmp_path / "fallback.sqlite"), embeddings=DummyEmbeddings())
    await fallback.enter("session-2")
    await fallback.add(
        MemoryItem(kind="note", text="Apple pie techniques", tags=[], meta={})
    )

    monkeypatch.setattr(mem0_module, "Mem0", None, raising=False)
    adapter = mem0_module.Mem0Adapter(fallback=fallback)

    recall = await adapter.similar("apple", k=1)
    assert recall and "Apple" in recall[0]["text"]
