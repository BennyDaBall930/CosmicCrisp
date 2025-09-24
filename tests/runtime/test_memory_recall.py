"""Memory retrieval behaviour with local and mem0-backed stores."""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

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


@pytest.mark.asyncio
async def test_mem0_adapter_local_mode_logging():
    """Test that mem0 adapter logs initialization mode correctly."""
    with patch.dict(os.environ, {"MEM0_LOCAL_MODE": "true"}):
        with patch('python.runtime.memory.stores.mem0_adapter.logger') as mock_logger:
            adapter = mem0_module.Mem0Adapter(namespace="test_local")
            mock_logger.info.assert_called_with(
                "Mem0Adapter initialized in local mode with namespace: %s", "test_local"
            )


@pytest.mark.asyncio
async def test_mem0_adapter_cloud_mode_logging():
    """Test that mem0 adapter logs cloud mode initialization."""
    with patch.dict(os.environ, {}, clear=True):
        with patch('python.runtime.memory.stores.mem0_adapter.logger') as mock_logger:
            adapter = mem0_module.Mem0Adapter(namespace="test_cloud")
            mock_logger.info.assert_called_with(
                "Mem0Adapter initialized in cloud mode with namespace: %s", "test_cloud"
            )


@pytest.mark.asyncio
async def test_mem0_adapter_with_working_client():
    """Test mem0 adapter operations with a working client."""
    mock_client = MagicMock()
    mock_client.search.return_value = [
        {"text": "Apple Zero is an expert programmer", "meta": {"category": "expertise"}}
    ]

    with patch('python.runtime.memory.stores.mem0_adapter._Mem0Client', return_value=mock_client):
        adapter = mem0_module.Mem0Adapter()

        # Test similar search
        results = await adapter.similar("programming expert", k=3)
        assert len(results) == 1
        assert results[0]["text"] == "Apple Zero is an expert programmer"

        mock_client.search.assert_called_once_with(
            "programming expert",
            limit=3,
            user_id="applezero"
        )


@pytest.mark.asyncio
async def test_mem0_adapter_error_fallback_to_sqlite(tmp_path):
    """Test that mem0 adapter falls back to SQLite when mem0 operations fail."""
    # Setup fallback store
    fallback = SQLiteFAISSMemory(path=str(tmp_path / "fallback.sqlite"), embeddings=DummyEmbeddings())
    await fallback.enter("test-session")
    await fallback.add(
        MemoryItem(kind="fallback", text="Fallback Apple information", tags=[], meta={})
    )

    # Mock failing mem0 client
    with patch('python.runtime.memory.stores.mem0_adapter._Mem0Client') as mock_client_class:
        mock_client_class.return_value.search.side_effect = Exception("Mem0 API error")

        adapter = mem0_module.Mem0Adapter(fallback=fallback)

        # Should fall back to SQLite
        results = await adapter.similar("apple", k=1)
        assert results and "Apple" in results[0]["text"]
        assert results[0]["text"] == "Fallback Apple information"


@pytest.mark.asyncio
async def test_mem0_adapter_namespace_usage():
    """Test that mem0 adapter uses correct namespace for operations."""
    mock_client = MagicMock()
    mock_client.add.return_value = {"id": "test_memory_123"}

    with patch('python.runtime.memory.stores.mem0_adapter._Mem0Client', return_value=mock_client):
        adapter = mem0_module.Mem0Adapter(namespace="custom_namespace")

        # Test add operation uses namespace
        item = MemoryItem(text="Test memory", meta={"test": True})
        await adapter.add(item)

        # Verify add was called with correct user_id
        call_args = mock_client.add.call_args
        assert call_args[1]["user_id"] == "custom_namespace"


@pytest.mark.asyncio
async def test_mem0_adapter_memory_count():
    """Test memory count functionality."""
    mock_client = MagicMock()
    mock_client.get_all.return_value = [
        {"text": "Memory 1"},
        {"text": "Memory 2"},
        {"text": "Memory 3"}
    ]

    with patch('python.runtime.memory.stores.mem0_adapter._Mem0Client', return_value=mock_client):
        with patch('python.runtime.memory.stores.mem0_adapter._client_mode', "client"):
            adapter = mem0_module.Mem0Adapter()
            adapter._client = mock_client
            adapter._client_mode = "client"

            count = await adapter.count()
            assert count == 3
            mock_client.get_all.assert_called_once_with(user_id="applezero")
