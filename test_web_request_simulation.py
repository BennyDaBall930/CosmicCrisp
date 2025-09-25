#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import threading

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_separate_processes():
    """Test what happens when we simulate separate processes making requests."""

    print("=== Testing separate process simulation ===")

    # Run multiple separate Python processes that each call get_chat_model
    processes = []
    for i in range(3):
        print(f"Starting process {i}...")
        cmd = [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
import models
print(f"Process {i}: Calling get_chat_model...")
provider = models.get_chat_model("apple_mlx", "")
if provider and hasattr(provider, '_model_kit') and provider._model_kit is not None:
    print(f"Process {i}: SUCCESS - Got provider with model_kit")
    print(f"Process {i}: Global _current_mlx_provider: {{models._current_mlx_provider is not None}}")
else:
    print(f"Process {i}: FAILED - Provider missing or model_kit None")
    print(f"Process {i}: Global _current_mlx_provider: {{models._current_mlx_provider is not None}}")
"""]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)

    # Wait for all processes to complete
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        print(f"Process {i} output:")
        print(stdout)
        if stderr:
            print(f"Process {i} stderr:")
            print(stderr)

def test_module_reimport():
    """Test what happens when we reimport the models module."""

    print("\n=== Testing module reimport simulation ===")

    import models
    print("Initial import - calling get_chat_model...")
    provider1 = models.get_chat_model("apple_mlx", "")
    print(f"After first call: _current_mlx_provider exists: {models._current_mlx_provider is not None}")

    # Simulate what might happen in a web server - reimport the module
    print("Reimporting models module...")
    import importlib
    importlib.reload(models)

    print("After reimport - calling get_chat_model again...")
    provider2 = models.get_chat_model("apple_mlx", "")
    print(f"After second call: _current_mlx_provider exists: {models._current_mlx_provider is not None}")

    if provider1 is provider2:
        print("SUCCESS: Same provider instance returned after reimport")
    else:
        print("ISSUE: Different provider instances after reimport")

if __name__ == "__main__":
    test_separate_processes()
    test_module_reimport()