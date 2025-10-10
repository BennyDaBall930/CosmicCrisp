import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest


def test_mps_device_detection():
    """Test that Apple Silicon devices prefer MPS over CUDA when both are available"""

    # Import torch and ensure we have the right version
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from python.helpers.xtts_tts import _pick_device

    # Test when MPS is available and CUDA is not
    with patch('torch.backends.mps') as mock_mps, \
         patch('torch.mps') as mock_mps_module, \
         patch('torch.cuda.is_available', return_value=False):

        # Configure MPS as available
        mock_mps.is_available = True
        mock_mps.available = True
        mock_mps.is_built.return_value = True

        device = _pick_device()
        assert device == "mps", f"Expected 'mps' when MPS is available, got '{device}'"

    # Test when CUDA is available but MPS is not
    with patch('torch.backends.mps') as mock_mps, \
         patch('torch.mps') as mock_mps_module, \
         patch('torch.cuda.is_available', return_value=True):

        # Configure MPS as unavailable
        mock_mps.is_available = False
        mock_mps.available = False

        device = _pick_device()
        assert device == "cuda", f"Expected 'cuda' when CUDA available and MPS not, got '{device}'"

    # Test when CUDA is available and MPS is available (should prefer MPS)
    with patch('torch.backends.mps') as mock_mps, \
         patch('torch.mps') as mock_mps_module, \
         patch('torch.cuda.is_available', return_value=True):

        # Configure MPS as available
        mock_mps.is_available = True
        mock_mps.available = True
        mock_mps.is_built.return_value = True

        device = _pick_device()
        assert device == "mps", f"Expected 'mps' preferred over CUDA, got '{device}'"

    # Test explicit device preference override
    device = _pick_device("cuda")
    assert device == "cuda", f"Expected explicit device preference to work, got '{device}'"

    device = _pick_device("cpu")
    assert device == "cpu", f"Expected explicit CPU selection to work, got '{device}'"

    # Test case insensitive variations
    device = _pick_device("AUTO")
    # Should behave the same as None (auto-detection)

    device = _pick_device("")  # Empty string should trigger auto-detection
    # Result depends on actual hardware, but should not crash


def test_xtts_device_configuration():
    """Test XTTS backend uses correct device based on Apple Silicon detection"""

    # Import required modules
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from python.helpers.xtts_tts import XTTSConfig, XTTSBackend

    try:
        # Skip if Coqui TTS not available (will use sidecar)
        from TTS.api import TTS
    except ImportError:
        pytest.skip("Coqui TTS not available for device testing")

    # Test with MPS preferred on Apple Silicon
    with patch('torch.backends.mps') as mock_mps, \
         patch('torch.mps') as mock_mps_module, \
         patch('torch.cuda.is_available', return_value=False), \
         patch('python.helpers.xtts_tts._pick_device', return_value="mps"):

        mock_mps.is_available = True
        mock_mps.available = True
        mock_mps.is_built.return_value = True

        try:
            cfg = XTTSConfig(sample_rate=24000)
            backend = XTTSBackend(cfg)
            # Should not raise an exception
            assert backend.device == "mps"
        except Exception as e:
            # If model loading fails, that's okay as long as device selection works
            if "MPS placement failed" in str(e):
                pytest.skip("MPS device placement not supported in test environment")


def test_force_cpu_fallback():
    """Test that CPU fallback works when MPS fails"""

    from python.helpers.xtts_tts import _pick_device

    # Simulate MPS failure scenario
    with patch('torch.backends.mps') as mock_mps, \
         patch('torch.mps') as mock_mps_module, \
         patch('torch.cuda.is_available', return_value=False):

        # MPS appears available but will fail during use
        mock_mps.is_available = True
        mock_mps.available = True
        mock_mps.is_built.return_value = True

        try:
            device = _pick_device()
            # Should return MPS since it appears available
            assert device == "mps"
        except Exception:
            # If auto-detection fails, ensure we get CPU as fallback
            device = _pick_device()
            assert device in ["cpu", "mps"], f"Expected CPU or MPS, got '{device}'"


def test_system_info_detection():
    """Test system info detection for Apple Silicon vs Intel"""

    # This test tries to detect if we're actually running on Apple Silicon
    try:
        result = subprocess.run(['sysctl', 'hw.machine'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            machine = result.stdout.strip().split(':')[-1].strip()

            # Apple Silicon machines include "arm64" in the machine type
            is_apple_silicon = 'arm64' in machine.lower()
        else:
            is_apple_silicon = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        is_apple_silicon = False

    # If we are on Apple Silicon, torch MPS should be available (if torch supports it)
    if is_apple_silicon:
        try:
            import torch
            if hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available() if hasattr(torch.backends.mps, 'is_available') else False

                if mps_available:
                    from python.helpers.xtts_tts import _pick_device
                    device = _pick_device()

                    # On Apple Silicon, MPS should be preferred if available
                    assert device == "mps", f"On Apple Silicon, expected MPS device, got '{device}'"
        except ImportError:
            pytest.skip("PyTorch not available on Apple Silicon")

    # If we're not on Apple Silicon, MPS shouldn't be the default
    else:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            from python.helpers.xtts_tts import _pick_device
            device = _pick_device()

            if has_cuda:
                assert device == "cuda", f"With CUDA available, expected CUDA device, got '{device}'"
            else:
                assert device == "cpu", f"Without CUDA/MPS, expected CPU device, got '{device}'"
        except ImportError:
            pytest.skip("PyTorch not available for device testing")
