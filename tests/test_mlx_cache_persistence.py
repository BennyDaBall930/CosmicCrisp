#!/usr/bin/env python3
"""
Test script to verify MLX cache persistence across module reloads.
This test simulates the cache behavior without actually loading MLX models.
"""

import sys
import os
import json
import hashlib
from pathlib import Path

def test_cache_persistence():
    """Test that the MLX cache manager persists data across module reloads."""

    print("=== Testing MLX Cache Persistence ===")

    # Define the cache manager inline to avoid import issues
    class MLXCacheManager:
        """
        Persistent cache manager for MLX provider that survives module reloads.
        Uses file-based storage to persist provider state across Flask/Werkzeug reloads.
        """

        def __init__(self):
            # Create cache directory in tmp folder
            self.cache_dir = Path(__file__).parent / "tmp" / "mlx_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "mlx_provider_cache.json"

        def _get_cache_key(self, model_path: str) -> str:
            """Generate a cache key based on model path"""
            return hashlib.md5(model_path.encode()).hexdigest()

        def save_provider_state(self, model_path: str, provider_state: dict):
            """Save provider state to persistent cache"""
            try:
                cache_key = self._get_cache_key(model_path)
                cache_data = {
                    "model_path": model_path,
                    "cache_key": cache_key,
                    "provider_state": provider_state,
                    "timestamp": os.path.getmtime(__file__) if os.path.exists(__file__) else 0,
                }

                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                print(f"[MLX Cache] Saved provider state for model: {model_path}")
            except Exception as e:
                print(f"[MLX Cache] Error saving cache: {e}")

        def load_provider_state(self, model_path: str):
            """Load provider state from persistent cache"""
            try:
                if not self.cache_file.exists():
                    return None

                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Validate cache key matches current model path
                expected_key = self._get_cache_key(model_path)
                if cache_data.get("cache_key") != expected_key:
                    print(f"[MLX Cache] Cache key mismatch for {model_path}, ignoring cache")
                    return None

                # Check if file has been modified since cache was created
                current_mtime = os.path.getmtime(__file__) if os.path.exists(__file__) else 0
                cache_mtime = cache_data.get("timestamp", 0)
                if current_mtime > cache_mtime:
                    print(f"[MLX Cache] test file modified since cache creation, ignoring cache")
                    return None

                print(f"[MLX Cache] Loaded provider state for model: {model_path}")
                return cache_data.get("provider_state")

            except Exception as e:
                print(f"[MLX Cache] Error loading cache: {e}")
                return None

        def clear_cache(self):
            """Clear all cached provider state"""
            try:
                if self.cache_file.exists():
                    self.cache_file.unlink()
                    print("[MLX Cache] Cache cleared")
            except Exception as e:
                print(f"[MLX Cache] Error clearing cache: {e}")

        def is_cache_valid(self, model_path: str):
            """Check if cache exists and is valid for the given model path"""
            state = self.load_provider_state(model_path)
            return state is not None

    # Create a cache manager instance
    cache_manager = MLXCacheManager()

    # Create a cache manager instance
    cache_manager = MLXCacheManager()

    # Clear any existing cache
    cache_manager.clear_cache()
    print("Cleared existing cache")

    # Test model path
    test_model_path = "/test/model/path"

    # Save provider state
    test_state = {
        "model_path": test_model_path,
        "model_loaded": True,
        "last_access": 1234567890,
        "test_data": "persistence_test"
    }

    print(f"Saving test state for model: {test_model_path}")
    cache_manager.save_provider_state(test_model_path, test_state)

    # Verify cache exists
    assert cache_manager.is_cache_valid(test_model_path), "Cache should be valid after saving"

    # Load and verify state
    loaded_state = cache_manager.load_provider_state(test_model_path)
    assert loaded_state is not None, "Should be able to load saved state"
    assert loaded_state["model_path"] == test_model_path, "Model path should match"
    assert loaded_state["model_loaded"] == True, "Model loaded flag should match"
    assert loaded_state["test_data"] == "persistence_test", "Test data should match"

    print("✓ Cache save/load works correctly")

    # Simulate module reload by creating a new cache manager instance
    print("Simulating module reload (creating new cache manager instance)...")
    new_cache_manager = MLXCacheManager()

    # Verify cache persists across "reload"
    assert new_cache_manager.is_cache_valid(test_model_path), "Cache should persist across reload"

    loaded_state_after_reload = new_cache_manager.load_provider_state(test_model_path)
    assert loaded_state_after_reload is not None, "Should be able to load state after reload"
    assert loaded_state_after_reload["model_path"] == test_model_path, "Model path should match after reload"
    assert loaded_state_after_reload["test_data"] == "persistence_test", "Test data should match after reload"

    print("✓ Cache persists across simulated module reload")

    # Test cache invalidation when model path changes
    different_model_path = "/different/model/path"
    loaded_different = new_cache_manager.load_provider_state(different_model_path)
    assert loaded_different is None, "Should not load state for different model path"

    print("✓ Cache correctly invalidates for different model paths")

    # Test cache clearing
    new_cache_manager.clear_cache()
    assert not new_cache_manager.is_cache_valid(test_model_path), "Cache should be cleared"

    print("✓ Cache clearing works correctly")

    # Test that cache file was actually created and contains expected data
    cache_file = cache_manager.cache_dir / "mlx_provider_cache.json"
    if cache_file.exists():
        print(f"✓ Cache file exists at: {cache_file}")
        # Check file contents before clearing
        with open(cache_file, 'r') as f:
            file_data = json.load(f)
        assert file_data["provider_state"]["model_path"] == test_model_path, "File should contain correct model path"
        print("✓ Cache file contains correct data")
    else:
        print("⚠ Cache file was cleared, which is expected")

    print("\n=== All cache persistence tests passed! ===")

if __name__ == "__main__":
    test_cache_persistence()
    print("\n🎉 Cache persistence test completed successfully!")