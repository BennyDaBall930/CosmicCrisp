"""Tests for local mem0 integration."""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from cosmiccrisp.python.runtime.memory.stores.mem0_adapter import Mem0Adapter
from cosmiccrisp.python.runtime.memory.schema import MemoryItem


class TestMem0Local:
    """Test local mem0 functionality."""

    def test_adapter_initialization_local_mode(self):
        """Test adapter initializes correctly in local mode."""
        with patch.dict(os.environ, {"MEM0_LOCAL_MODE": "true"}):
            adapter = Mem0Adapter(namespace="test")
            assert adapter._local_mode is True
            assert adapter._namespace == "test"

    def test_adapter_initialization_cloud_mode(self):
        """Test adapter initializes correctly in cloud mode."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = Mem0Adapter(namespace="test")
            assert adapter._local_mode is False
            assert adapter._namespace == "test"

    @pytest.mark.asyncio
    async def test_fallback_when_mem0_unavailable(self):
        """Test fallback to SQLite when mem0 is not available."""
        # Mock mem0 imports to raise ImportError
        with patch('python.runtime.memory.stores.mem0_adapter._LegacyMem0', None), \
             patch('python.runtime.memory.stores.mem0_adapter._Mem0Client', None):

            adapter = Mem0Adapter()
            assert adapter._client is None

            # Test operations fall back to SQLite (if fallback provided)
            # This would typically be tested with actual fallback store
            # For now, test that the methods don't crash
            try:
                # These should not raise exceptions if fallback is properly configured
                await adapter.similar("test query", 5)
            except RuntimeError as e:
                assert "fallback configured" in str(e)

    @pytest.mark.asyncio
    async def test_memory_operations_with_mocked_client(self):
        """Test memory operations with mocked mem0 client."""
        mock_client = MagicMock()
        mock_client.add.return_value = {"id": "test_id"}
        mock_client.search.return_value = [
            {"text": "test memory", "meta": {"category": "test"}, "id": "mem_1"}
        ]
        mock_client.recent.return_value = [
            {"text": "recent memory", "meta": {"category": "recent"}, "id": "mem_2"}
        ]
        mock_client.count.return_value = 2

        adapter = Mem0Adapter()
        adapter._client = mock_client
        adapter._client_mode = "legacy"

        # Test add operation
        item = MemoryItem(text="test text", meta={"test": "data"})
        result = await adapter.add(item)
        assert result == "test_id"
        mock_client.add.assert_called_once()

        # Test search operation
        results = await adapter.similar("test query", 5)
        assert len(results) == 1
        assert results[0]["text"] == "test memory"
        mock_client.search.assert_called_once_with("test query", 5)

        # Test recent operation
        results = await adapter.recent(3)
        assert len(results) == 1
        assert results[0]["text"] == "recent memory"
        mock_client.recent.assert_called_once_with(3)

        # Test count operation
        count = await adapter.count()
        assert count == 2
        mock_client.count.assert_called_once()

        # Test error handling
        mock_client.add.side_effect = Exception("API error")
        result = await adapter.add(item)
        if adapter._fallback is None:
            # Should raise if no fallback
            with pytest.raises(RuntimeError):
                await adapter.add(item)

    def test_environment_variable_parsing(self):
        """Test various forms of MEM0_LOCAL_MODE environment variable."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("", False),
            ("anything_else", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MEM0_LOCAL_MODE": env_value}):
                adapter = Mem0Adapter()
                assert adapter._local_mode == expected, f"Failed for {env_value}"

    @pytest.mark.asyncio
    async def test_namespace_handling(self):
        """Test memory operations respect namespace configuration."""
        mock_client = MagicMock()
        mock_client.get_all.return_value = []

        adapter = Mem0Adapter(namespace="custom_namespace", base_url="http://test")
        adapter._client = mock_client
        adapter._client_mode = "client"  # Use new client mode

        # Test that namespace is used in operations
        await adapter.count()
        # Verify get_all was called with user_id parameter
        args, kwargs = mock_client.get_all.call_args
        assert kwargs.get("user_id") == "custom_namespace"

    def test_adapter_cleanup(self):
        """Test adapter resources are properly managed."""
        with patch.dict(os.environ, {"MEM0_LOCAL_MODE": "true"}):
            adapter = Mem0Adapter()
            # Adapter should not hold any resource locks
            assert adapter._client is None or hasattr(adapter._client, 'close')
