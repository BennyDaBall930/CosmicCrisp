#!/usr/bin/env python3
"""Test script for MLX server management functionality."""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    from python.helpers.mlx_server import MLXServerManager
    print("✓ Successfully imported MLXServerManager")

    # Test creating an instance
    manager = MLXServerManager()
    print("✓ Successfully created MLXServerManager instance")

    # Test getting status
    status = manager.get_status()
    print(f"✓ Server status: {status}")

    print("✓ All MLX server tests passed!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)