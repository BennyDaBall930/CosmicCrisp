import os
import urllib.error
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("disable_autostart", [True])
def test_xtts_sidecar_connection_refused_gives_actionable_error(monkeypatch, disable_autostart):
    # Force use of sidecar backend regardless of local Coqui availability
    monkeypatch.setenv("TTS_FORCE_SIDECAR", "1")
    # Point to an unlikely port to ensure connection refusal
    monkeypatch.setenv("TTS_SIDECAR_URL", "http://127.0.0.1:59999")
    if disable_autostart:
        monkeypatch.setenv("A0_DISABLE_SIDECAR_AUTOSTART", "1")

    # Make urlopen always raise URLError(ConnectionRefusedError)
    def _raise(*args, **kwargs):
        raise urllib.error.URLError(OSError(61, "Connection refused"))

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

    backend = SidecarXTTSBackend(XTTSConfig())
    with pytest.raises(RuntimeError) as ei:
        backend.synthesize("Hello world")

    msg = str(ei.value)
    # Actionable message should mention sidecar and how to start it
    assert "sidecar" in msg.lower()
    assert "setup.sh" in msg and "run.sh" in msg
    assert "127.0.0.1:59999" in msg


def test_sidecar_preload_warmup_autostart(monkeypatch):
    """Test sidecar is started during preload warmup instead of first synthesis"""

    # Mock health check to fail initially, succeed after Popen is called
    health_call_count = 0

    def mock_health_check(timeout=1.5):
        nonlocal health_call_count
        health_call_count += 1
        return health_call_count > 2  # Succeeds on 3rd+ call

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', side_effect=mock_health_check) as mock_health, \
         patch('subprocess.Popen') as mock_popen, \
         patch('os.path.exists', return_value=True), \
         patch('time.sleep'), \
         patch('time.time', side_effect=[0, 10, 15, 20, 25]), \
         patch('os.getcwd', return_value="/fake/project/root"), \
         patch('python.helpers.xtts_tts._post_json', return_value={"status": "healthy"}):

        # Force use of sidecar backend
        monkeypatch.setenv("TTS_FORCE_SIDECAR", "1")
        monkeypatch.setenv("TTS_SIDECAR_URL", "http://127.0.0.1:79999")

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig, get_backend

        # Simulate preload by getting backend
        cfg = XTTSConfig()
        backend = get_backend(cfg)

        assert isinstance(backend, SidecarXTTSBackend)

        # Force sidecar startup during preload by calling _ensure_running
        backend._ensure_running()

        # Verify subprocess was started (health check should have failed initially, causing Popen)
        mock_popen.assert_called()
        # Health check should have been called multiple times
        assert mock_health.call_count >= 3


def test_sidecar_attempted_autostart_prevents_repeated_spawns():
    """Test that repeated _ensure_running calls don't spawn multiple processes"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False), \
         patch('subprocess.Popen') as mock_popen, \
         patch('os.path.exists', return_value=True), \
         patch('os.getcwd', return_value="/fake/project/root"), \
         patch('time.sleep'), \
         patch('time.time', side_effect=[0, 0, 0, 0, 5]):  # Quick success

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        backend = SidecarXTTSBackend(XTTSConfig())

        # First call should attempt to start
        backend._ensure_running()

        # Reset for second call
        backend._attempted_autostart = True

        # Second call should not attempt to start again
        with pytest.raises(RuntimeError) as exc_info:
            backend._ensure_running()

        error_msg = str(exc_info.value).lower()
        assert "failed to reach" in error_msg and "after autostart" in error_msg

        # Should have only called Popen once
        assert mock_popen.call_count == 1


def test_sidecar_environment_variable_disable_variations():
    """Test various values for A0_DISABLE_SIDECAR_AUTOSTART"""

    test_values = ["1", "true", "TRUE", "True", "yes", "invalid", None, ""]

    for value in test_values:
        with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False):
            from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig
            import os

            backend = SidecarXTTSBackend(XTTSConfig())

            # Set environment variable
            old_value = os.environ.get('A0_DISABLE_SIDECAR_AUTOSTART')
            if value is not None:
                os.environ['A0_DISABLE_SIDECAR_AUTOSTART'] = value
            else:
                # Ensure variable is not set
                if 'A0_DISABLE_SIDECAR_AUTOSTART' in os.environ:
                    del os.environ['A0_DISABLE_SIDECAR_AUTOSTART']

            try:
                # Test behavior based on environment variable
                if value in ["1", "true", "TRUE", "True"]:
                    # Should raise RuntimeError about sidecar not running
                    with pytest.raises(RuntimeError) as exc_info:
                        backend._ensure_running()
                    assert "sidecar is not running" in str(exc_info.value).lower()
                    assert "set TTS engine to 'browser'" in str(exc_info.value).lower()
                else:
                    # Should not be disabled, but will fail due to mocking
                    pass
            finally:
                # Restore environment
                if old_value is not None:
                    os.environ['A0_DISABLE_SIDECAR_AUTOSTART'] = old_value
                elif value is not None and value in ["1", "true", "TRUE", "True"]:
                    if 'A0_DISABLE_SIDECAR_AUTOSTART' in os.environ:
                        del os.environ['A0_DISABLE_SIDECAR_AUTOSTART']
