#!/usr/bin/env python3
"""
Simple test script to isolate MLX preloading issue
"""
import sys
import os

# Add the python directory to path
app_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.join(app_dir, "python")
if python_path not in sys.path:
    sys.path.insert(0, python_path)

try:
    # Import required modules
    from python.helpers import settings
    import models

    # Test loading settings
    print("Loading settings...")
    current = settings.get_settings()
    print(f"Chat model provider: {current.get('chat_model_provider', 'NOT SET')}")

    # Test if apple_mlx is configured
    if current.get("chat_model_provider", "").lower() != "apple_mlx":
        print("Apple MLX is not the configured provider")
        sys.exit(0)

    print("Apple MLX is configured, attempting to load model...")

    # Test model loading
    provider = models.get_chat_model("apple_mlx", "")

    # Check if loaded successfully
    if provider and hasattr(provider, '_model_kit') and provider._model_kit is not None:
        print("Apple MLX model preloaded successfully!")
    else:
        print("Failed to preload Apple MLX model.")

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()