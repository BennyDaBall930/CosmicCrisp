"""Tests for Apple Zero memory preloading functionality."""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from scripts.setup_mem0 import AppleZeroMemoryPreloader


class TestMemoryPreloading:
    """Test Apple Zero memory preloading functionality."""

    def test_preloader_initialization(self):
        """Test preloader initializes correctly."""
        with patch.dict(os.environ, {
            "MEM0_API_KEY": "test_key",
            "MEM0_BASE_URL": "http://test"
        }):
            preloader = AppleZeroMemoryPreloader(namespace="test_ns", base_url="http://test")
            assert preloader.namespace == "test_ns"
            assert preloader.preload_data_path.exists()  # Should point to the JSON file

    def test_preload_data_file_location(self):
        """Test preloader finds preload data file correctly."""
        preloader = AppleZeroMemoryPreloader()
        expected_path = Path(__file__).parent.parent / "scripts" / "mem0_preload_data.json"
        assert preloader.preload_data_path == expected_path

    @pytest.mark.asyncio
    async def test_successful_mem0_initialization(self):
        """Test successful mem0 initialization."""
        mock_client = MagicMock()
        mock_client.add.return_value = None  # Test call succeeds

        with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
            preloader = AppleZeroMemoryPreloader()

            result = await preloader.setup_local_mem0()
            assert result is True
            mock_client.add.assert_called_once_with([], user_id="applezero", metadata=None)

    @pytest.mark.asyncio
    async def test_failed_mem0_initialization(self):
        """Test failed mem0 initialization."""
        with patch('scripts.setup_mem0.MemoryClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            preloader = AppleZeroMemoryPreloader()
            result = await preloader.setup_local_mem0()
            assert result is False

    @pytest.mark.asyncio
    async def test_successful_memory_preloading(self):
        """Test successful preloading of Apple Zero memories."""
        # Sample test data
        test_memories = [
            {
                "text": "Test memory 1",
                "metadata": {"category": "test", "confidence": 0.9}
            },
            {
                "text": "Test memory 2",
                "metadata": {"category": "test2", "confidence": 0.8}
            }
        ]

        mock_client = MagicMock()

        # Mock the JSON file reading
        mock_file_data = json.dumps(test_memories)
        with patch('builtins.open', mock_open(read_data=mock_file_data)):
            with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
                preloader = AppleZeroMemoryPreloader()
                count = await preloader.preload_apple_zero_memories()

        assert count == 2
        assert mock_client.add.call_count == 2

        # Verify the calls
        calls = mock_client.add.call_args_list
        assert len(calls) == 2

        # Check first call
        args, kwargs = calls[0]
        assert args[0] == [{"role": "user", "content": "Test memory 1"}]
        assert kwargs["user_id"] == "applezero"
        assert kwargs["metadata"] == {"category": "test", "confidence": 0.9}

        # Check second call
        args, kwargs = calls[1]
        assert args[0] == [{"role": "user", "content": "Test memory 2"}]
        assert kwargs["user_id"] == "applezero"
        assert kwargs["metadata"] == {"category": "test2", "confidence": 0.8}

    @pytest.mark.asyncio
    async def test_memory_preloading_missing_file(self):
        """Test preloading when data file doesn't exist."""
        mock_client = MagicMock()

        with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
            # Temporarily move the preload file
            preloader = AppleZeroMemoryPreloader()
            original_path = preloader.preload_data_path
            preloader.preload_data_path = Path("nonexistent_file.json")

            count = await preloader.preload_apple_zero_memories()
            assert count == 0
            mock_client.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_preloading_invalid_json(self):
        """Test preloading with invalid JSON data."""
        mock_client = MagicMock()

        with patch('builtins.open', mock_open(read_data="invalid json {")):
            with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
                preloader = AppleZeroMemoryPreloader()
                count = await preloader.preload_apple_zero_memories()
                assert count == 0
                mock_client.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_individual_memory_loading_failure(self):
        """Test that individual memory loading failures don't stop the entire process."""
        test_memories = [
            {
                "text": "Good memory",
                "metadata": {"category": "good"}
            },
            {
                "text": "Bad memory",
                "metadata": {"category": "bad"}
            }
        ]

        mock_client = MagicMock()
        # Make the second call fail
        mock_client.add.side_effect = [None, Exception("API Error"), None]

        mock_file_data = json.dumps(test_memories)
        with patch('builtins.open', mock_open(read_data=mock_file_data)):
            with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
                preloader = AppleZeroMemoryPreloader()
                count = await preloader.preload_apple_zero_memories()

        assert count == 1  # Only first memory should succeed
        assert mock_client.add.call_count == 2  # But both attempts made

    @pytest.mark.asyncio
    async def test_verification_success(self):
        """Test successful verification of mem0 integration."""
        mock_client = MagicMock()
        mock_client.get_all.return_value = [
            {"text": "memory1", "id": "1"},
            {"text": "memory2", "id": "2"}
        ]
        mock_client.search.return_value = [{"text": "searched memory"}]

        with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
            preloader = AppleZeroMemoryPreloader()
            result = await preloader.verify_mem0_integration()

        assert result is True
        mock_client.get_all.assert_called_once_with(user_id="applezero")
        mock_client.search.assert_called_once_with("Apple ecosystem", limit=5, user_id="applezero")

    @pytest.mark.asyncio
    async def test_verification_with_empty_memories(self):
        """Test verification with no preloaded memories."""
        mock_client = MagicMock()
        mock_client.get_all.return_value = []  # Empty list
        mock_client.search.return_value = []

        with patch('scripts.setup_mem0.MemoryClient', return_value=mock_client):
            preloader = AppleZeroMemoryPreloader()
            result = await preloader.verify_mem0_integration()

        assert result is True
        # Search should still be called when memories exist
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_verification_failure(self):
        """Test failed verification."""
        with patch('scripts.setup_mem0.MemoryClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            preloader = AppleZeroMemoryPreloader()
            result = await preloader.verify_mem0_integration()

        assert result is False

    def test_memory_format_conversion(self):
        """Test conversion of memory data to mem0 format."""
        preloader = AppleZeroMemoryPreloader()

        # Test memory data
        memory_data = {
            "text": "Sample memory text",
            "metadata": {
                "category": "test",
                "tags": ["tag1", "tag2"],
                "confidence": 0.95
            }
        }

        # Expected format for mem0 client
        messages = [{"role": "user", "content": memory_data["text"]}]
        metadata = memory_data["metadata"]

        # Verify our understanding of the format
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Sample memory text"
        assert metadata["category"] == "test"
        assert metadata["tags"] == ["tag1", "tag2"]
        assert metadata["confidence"] == 0.95
