#!/usr/bin/env python3

import os
import sys
import importlib

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_reimport():
    """Test what happens when we reimport the models module."""

    print("=== Testing module reimport behavior ===")

    import models
    print("Initial import - calling get_chat_model...")
    provider1 = models.get_chat_model("apple_mlx", "")
    print(f"After first call: _current_mlx_provider exists: {models._current_mlx_provider is not None}")

    # Simulate what might happen in a web server - reimport the module
    print("Reimporting models module...")
    importlib.reload(models)

    print("After reimport - checking global variable...")
    print(f"After reimport: _current_mlx_provider exists: {models._current_mlx_provider is not None}")

    print("After reimport - calling get_chat_model again...")
    provider2 = models.get_chat_model("apple_mlx", "")
    print(f"After second call: _current_mlx_provider exists: {models._current_mlx_provider is not None}")

    if provider1 is provider2:
        print("SUCCESS: Same provider instance returned after reimport")
    else:
        print("ISSUE: Different provider instances after reimport")

if __name__ == "__main__":
    test_module_reimport()